"""
corep_ingest.py — Ingest COREP regulatory DOCX files into the RAG vector database.

Usage:
    python corep_ingest.py --db local                              # custom parser (default)
    python corep_ingest.py --db local --parser docling             # docling parser
    python corep_ingest.py --db prod                               # use DATABASE_URL
    python corep_ingest.py --db local --dry-run                    # parse only, no DB writes
    python corep_ingest.py --db local --file "Capital"             # ingest one matching file
    python corep_ingest.py --db local --dry-run --verbose          # show every chunk

Parser options:
    --parser custom   (default) Hand-crafted style-based parser using python-docx.
                      Reads Word paragraph styles, maintains a heading stack, and builds
                      breadcrumb context prefixes. Full control over COREP-specific styles.
    --parser docling  IBM Docling library. Converts DOCX via its own pipeline, then uses
                      HybridChunker (structure + token-aware). Less tunable but handles
                      edge cases automatically (merged cells, font artefacts, etc.).

Environment variables (in .env):
    DATABASE_URL_LOCAL   — local PostgreSQL connection string (for testing)
    DATABASE_URL         — production PostgreSQL connection string
    OPENAI_API_KEY       — OpenAI API key

Schema changes:
    Adds columns source, annex, breadcrumb to documents table (non-destructive).
    Existing CRR rows are preserved — only COREP rows are deleted before re-ingesting.
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure Unicode output works on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

load_dotenv(override=True)  # .env file takes precedence (same as app.py)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COREP_DIR    = Path("COREP")
EMBED_MODEL  = "text-embedding-3-small"
BATCH_SIZE   = 20   # rows per reconnect — smaller = shorter connection hold time
MAX_CHUNK    = 1200   # chars — flush buffer if it grows beyond this
MIN_CHUNK    = 500    # chars — merge chunks shorter than this with their neighbours
_NOISE_CHUNK = 30    # chars — discard chunks shorter than this (truly empty/noise)

# ---------------------------------------------------------------------------
# Paragraph style classification
#
# Style names discovered by scanning all 19 COREP DOCX files:
#   Heading:    'Instructions Uberschrift 2'  (198x across files)
#               'Heading 2'                   (7x in a few files)
#   Standalone: 'Instructions Text 2'         (330x) — numbered instructions
#   Body:       'Normal'                      (177x)
#               'Instructions Text'           (15x)
#               'Body Text1'                  (50x, Annex XI only)
#               'List Paragraph'              (25x)
#   Skip:       'toc 1/2/3'                   — table of contents, not useful
#               'Titre article'               — French style artifact
#
# Normalisation: lowercase, strip spaces, u-umlaut -> u.
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("ü", "u").replace("Ü", "u")

# Heading styles mapped to hierarchy level (1 = top).
# All heading styles in these files are at one effective level.
_HEADING_MAP = {
    _norm("Instructions Uberschrift 2"): 1,   # actual name (ü normalised above)
    _norm("Instructions berschrift 2"):  1,   # fallback if ü stripped entirely
    _norm("Heading 1"): 1,
    _norm("Heading 2"): 1,
    _norm("Heading 3"): 2,
}

# Styles that flush as their own chunk immediately (numbered general instructions)
_STANDALONE_STYLES = {
    _norm("Instructions Text 2"),
}

# Styles that contribute body text (buffered and merged across paragraphs)
_BODY_STYLES = {
    _norm("Instructions Text"),
    _norm("Normal"),
    _norm("Body Text1"),
    _norm("List Paragraph"),
    _norm("List Paragraph1"),
    _norm("List Bullet"),
    _norm("List"),
    _norm("body"),
    _norm("Table Paragraph"),
}

# Styles to skip entirely (table of contents, language artifacts)
_SKIP_STYLES = {
    _norm("toc 1"),
    _norm("toc 2"),
    _norm("toc 3"),
    _norm("TOC Heading"),
    _norm("Titre article"),
}


def classify_para(para: Paragraph) -> tuple[str, int | None]:
    """
    Returns (kind, heading_level) where kind is one of:
      'heading'     — a section heading
      'standalone'  — a numbered general instruction (flush immediately)
      'body'        — regular body text (buffer and merge)
      'skip'        — empty or noise
    """
    text = para.text.strip()
    if not text or len(text) < 3:
        return "skip", None

    style_name = para.style.name or "Normal"
    style_key  = _norm(style_name)

    if style_key in _SKIP_STYLES:
        return "skip", None
    if style_key in _HEADING_MAP:
        return "heading", _HEADING_MAP[style_key]
    if style_key in _STANDALONE_STYLES:
        return "standalone", None
    return "body", None


# ---------------------------------------------------------------------------
# Filename → metadata
# ---------------------------------------------------------------------------

def parse_filename_meta(filename: str) -> dict:
    """
    Extract annex label and topic from COREP filenames like:
      "2.2 Annex II - Part II - Capital adequacy - Clean.docx"
      "11 Annex XI - Leverage - Clean.docx"
      "7 Annex VII - IP Losses - Clean.docx"
    """
    name = re.sub(r"\s*-\s*Clean\.docx$", "", filename, flags=re.IGNORECASE).strip()
    # Match leading number, then "Annex <roman>", then rest
    m = re.match(r"[\d.]+\s+(Annex\s+[IVXLC]+)\s*[-–]?\s*(.*)", name, re.IGNORECASE)
    if m:
        annex = m.group(1).strip()
        rest  = m.group(2).strip(" -–")
        # Strip "Part II -" or "Part I -" prefix from the topic
        topic = re.sub(r"^Part\s+[IVXLC]+\s*[-–]?\s*", "", rest, flags=re.IGNORECASE).strip()
    else:
        annex = "COREP"
        topic = name
    return {"annex": annex, "topic": topic}


# ---------------------------------------------------------------------------
# Breadcrumb builder
# ---------------------------------------------------------------------------

def make_breadcrumb(annex: str, topic: str, headings: list[str]) -> str:
    """
    Produce a context prefix like:
      [COREP | Annex II | Capital adequacy | C 01.00 - OWN FUNDS | Row definitions]
    """
    parts = ["COREP", annex]
    if topic and topic not in parts:
        parts.append(topic)
    for h in headings:
        if h and h not in parts:
            parts.append(h)
    return "[" + " | ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Chunk post-processing: merge small consecutive text chunks
# ---------------------------------------------------------------------------

def _merge_small_chunks(chunks: list[dict],
                        threshold: int = MIN_CHUNK,
                        max_size: int = MAX_CHUNK) -> list[dict]:
    """
    Post-processing pass: merge consecutive non-table text chunks where at least
    one has body text (content minus the leading breadcrumb line) shorter than
    `threshold`, as long as the combined body stays within `max_size`.

    Table rows (body starts with "[Table]") are never merged — they carry
    independent header context and must stay as individual rows.  They ARE
    preserved even if their body is shorter than `threshold`; only content
    below _NOISE_CHUNK is discarded before this function is called.
    """
    if not chunks:
        return chunks

    def _body(chunk: dict) -> str:
        c = chunk["content"]
        return c.split("\n", 1)[1] if "\n" in c else c

    result: list[dict] = []
    pending = dict(chunks[0])

    for nxt in chunks[1:]:
        p_body = _body(pending)
        n_body = _body(nxt)
        p_tbl  = p_body.lstrip().startswith("[Table]")
        n_tbl  = n_body.lstrip().startswith("[Table]")

        can_merge = (
            not p_tbl and not n_tbl                          # neither is a table row
            and (len(p_body) < threshold
                 or len(n_body) < threshold)                  # at least one is small
            and len(p_body) + 1 + len(n_body) <= max_size    # combined fits
        )

        if can_merge:
            # Prefer the more specific (longer) breadcrumb
            bc = (nxt["breadcrumb"]
                  if len(nxt["breadcrumb"]) > len(pending["breadcrumb"])
                  else pending["breadcrumb"])
            merged_body = p_body + "\n" + n_body
            pending = {
                "content":    f"{bc}\n{merged_body}",
                "breadcrumb": bc,
                "annex":      pending["annex"],
            }
        else:
            result.append(pending)
            pending = dict(nxt)

    result.append(pending)
    return result


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def table_to_chunks(table: Table, breadcrumb: str, annex: str) -> list[dict]:
    """One chunk per content row; first row used as column header context."""
    chunks = []
    rows = table.rows
    if len(rows) < 2:
        return chunks

    # Build header line (deduplicate merged cells)
    seen_cells = set()
    header_parts = []
    for cell in rows[0].cells:
        cell_id = id(cell._tc)  # merged cells share the same _tc element
        if cell_id not in seen_cells:
            seen_cells.add(cell_id)
            t = cell.text.strip()
            if t:
                header_parts.append(t)
    header_line = " | ".join(header_parts)

    for row in rows[1:]:
        seen_cells = set()
        cell_texts = []
        for cell in row.cells:
            cell_id = id(cell._tc)
            if cell_id not in seen_cells:
                seen_cells.add(cell_id)
                t = cell.text.strip()
                if t:
                    cell_texts.append(t)

        row_text = " | ".join(cell_texts)
        if len(row_text) < _NOISE_CHUNK:
            continue

        content = f"{breadcrumb}\n[Table] {header_line}\n{row_text}"
        chunks.append({"content": content, "breadcrumb": breadcrumb, "annex": annex})

    return chunks


# ---------------------------------------------------------------------------
# DOCX parser
# ---------------------------------------------------------------------------

def split_at_sentences(text: str, breadcrumb: str, annex: str) -> list[dict]:
    """
    Split oversized text at sentence boundaries into chunks ≤ MAX_CHUNK chars.
    """
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > MAX_CHUNK:
            if len(current.strip()) >= _NOISE_CHUNK:
                chunks.append({"content": f"{breadcrumb}\n{current.strip()}",
                               "breadcrumb": breadcrumb, "annex": annex})
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current.strip() and len(current.strip()) >= _NOISE_CHUNK:
        chunks.append({"content": f"{breadcrumb}\n{current.strip()}",
                       "breadcrumb": breadcrumb, "annex": annex})
    return chunks


def parse_docx(path: Path, verbose: bool = False) -> list[dict]:
    """
    Parse a COREP DOCX into chunks using paragraph style hierarchy.

    Strategy:
      - Headings update the breadcrumb context (3-level stack)
      - InstructionsText2 paragraphs → each is its own chunk (numbered instructions)
      - Body paragraphs (InstructionsText, Normal, …) → accumulated into a buffer;
        flushed when a heading changes or the buffer exceeds MAX_CHUNK chars
      - Tables → one chunk per content row
    """
    meta  = parse_filename_meta(path.name)
    annex = meta["annex"]
    topic = meta["topic"]

    doc      = Document(str(path))
    headings = ["", "", ""]   # [H1, H2, H3]
    buffer: list[str] = []
    chunks:  list[dict] = []
    style_seen: dict[str, int] = {}  # for verbose reporting

    def current_breadcrumb() -> str:
        return make_breadcrumb(annex, topic, headings)

    def flush(force_breadcrumb: str | None = None):
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer = []
        if len(text) < _NOISE_CHUNK:
            return
        bc = force_breadcrumb or current_breadcrumb()
        if len(text) <= MAX_CHUNK:
            chunks.append({"content": f"{bc}\n{text}", "breadcrumb": bc, "annex": annex})
        else:
            chunks.extend(split_at_sentences(text, bc, annex))

    # Iterate body children in document order (paragraphs + tables interleaved)
    for child in doc.element.body.iterchildren():
        local_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if local_tag == "p":
            para = Paragraph(child, doc)
            kind, level = classify_para(para)
            text = para.text.strip()

            if verbose:
                sname = para.style.name or "Normal"
                style_seen[sname] = style_seen.get(sname, 0) + 1

            if kind == "heading":
                flush()
                idx = level - 1  # 0-based
                headings[idx] = text
                for i in range(idx + 1, len(headings)):
                    headings[i] = ""  # clear sub-headings

            elif kind == "standalone":
                flush()
                bc = current_breadcrumb()
                if len(text) >= _NOISE_CHUNK:
                    chunks.append({"content": f"{bc}\n{text}",
                                   "breadcrumb": bc, "annex": annex})

            elif kind == "body" and text:
                buffer.append(text)
                if len("\n".join(buffer)) >= MAX_CHUNK:
                    flush()

            # kind == "skip" → do nothing

        elif local_tag == "tbl":
            flush()
            table  = Table(child, doc)
            bc     = current_breadcrumb()
            chunks.extend(table_to_chunks(table, bc, annex))

    flush()  # flush any trailing buffer

    if verbose:
        print(f"    Style distribution (top 10):")
        for sname, count in sorted(style_seen.items(), key=lambda x: -x[1])[:10]:
            print(f"      {count:4d}x  {sname}")

    return _merge_small_chunks(chunks)


# ---------------------------------------------------------------------------
# Docling parser (alternative to parse_docx)
# ---------------------------------------------------------------------------

# Docling adds a numeric level prefix to heading strings, e.g. "1 10.1 General remarks".
# Strip the leading digit(s)+space so breadcrumbs look the same as the custom parser.
_DOCLING_HEADING_PREFIX = re.compile(r"^\d+\s+")

# HybridChunker token budget — ~300 tokens ≈ 1 200 chars, matching MAX_CHUNK above.
_DOCLING_MAX_TOKENS = 300


def parse_docx_docling(path: Path) -> list[dict]:
    """
    Parse a COREP DOCX with IBM Docling and return chunks in the same format as
    parse_docx(): list of {"content": ..., "breadcrumb": ..., "annex": ...}.

    Docling pipeline:
      1. DocumentConverter converts the DOCX into a structured DoclingDocument
         (headings, paragraphs, tables all labelled and linked).
      2. HybridChunker splits that document respecting both the heading hierarchy
         (structure-aware) and a token budget (token-aware).
      3. Each chunk carries chunk.meta.headings — the active heading path at that
         point in the document — which we use to build the breadcrumb.
    """
    # Lazy import so the module loads fast when docling is not used.
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    meta  = parse_filename_meta(path.name)
    annex = meta["annex"]
    topic = meta["topic"]

    converter = DocumentConverter()
    doc = converter.convert(str(path)).document

    chunker = HybridChunker(max_tokens=_DOCLING_MAX_TOKENS)
    chunks: list[dict] = []

    for chunk in chunker.chunk(doc):
        text = chunk.text.strip()
        if len(text) < _NOISE_CHUNK:
            continue

        # Build heading list from docling metadata, stripping the numeric prefix
        raw_headings = chunk.meta.headings or []
        clean_headings = [_DOCLING_HEADING_PREFIX.sub("", h).strip() for h in raw_headings]

        bc = make_breadcrumb(annex, topic, clean_headings)
        content = f"{bc}\n{text}"
        chunks.append({"content": content, "breadcrumb": bc, "annex": annex})

    return _merge_small_chunks(chunks)


# ---------------------------------------------------------------------------
# LLM-assisted parser (document map + chunk enrichment)
# ---------------------------------------------------------------------------

_DOCMAP_DIR = COREP_DIR / ".docmaps"

_DOCMAP_PROMPT = """\
You are analysing a COREP regulatory reporting document. Your task is to build a
structured "document map" that captures the section hierarchy and the reporting
templates defined in this document.

Rules:
- Identify ALL logical section headings, even if they look like plain paragraphs
  (e.g. "PART I: GENERAL INSTRUCTIONS", "3. C 47.00 – Leverage ratio calculation").
- For every reporting template mentioned (e.g. C47.00, LRCalc, LR1, LR4, LR5),
  extract its short code(s) and a concise description of what it reports.
- Return ONLY a single valid JSON object — no markdown, no explanation.

Required JSON schema:
{
  "sections": [
    {
      "heading": "<exact heading text>",
      "level": <1 for top-level, 2 for sub-section, etc.>,
      "description": "<one sentence: what this section covers>"
    }
  ],
  "templates": {
    "<short_code>": {
      "name": "<full template name incl. template ID>",
      "description": "<one sentence: what this template reports and its key purpose>",
      "section_heading": "<the section heading under which this template is described>"
    }
  }
}

Document text:
"""

# Regex patterns to detect template codes inside chunk text
_TEMPLATE_CODE_RE = re.compile(
    r'\b(LRCalc|LR1|LR2|LR3|LR4|LR5|LR6'        # Leverage
    r'|C\s*\d+[\w.]*'                              # generic C xx.xx codes
    r'|CA\d|CR\w+|MKR\w+|LE\w+|G\w+SII'           # other COREP codes
    r')\b'
)


def _extract_raw_text(path: Path) -> str:
    """
    Extract a clean plain-text representation of the DOCX for the LLM prompt.
    Paragraphs → text lines; tables → [Table] header / row lines.
    """
    doc = Document(str(path))
    lines: list[str] = []

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                lines.append(text)

        elif tag == "tbl":
            table = Table(child, doc)
            rows = table.rows
            if not rows:
                continue
            # Header row
            seen = set()
            header_parts = []
            for cell in rows[0].cells:
                cid = id(cell._tc)
                if cid not in seen:
                    seen.add(cid)
                    t = cell.text.strip()
                    if t:
                        header_parts.append(t)
            lines.append("[Table] " + " | ".join(header_parts))
            # Data rows (truncate very long tables to save tokens)
            for row in rows[1:min(len(rows), 60)]:
                seen = set()
                parts = []
                for cell in row.cells:
                    cid = id(cell._tc)
                    if cid not in seen:
                        seen.add(cid)
                        t = cell.text.strip()
                        if t:
                            parts.append(t)
                if parts:
                    lines.append("  " + " | ".join(parts))

    return "\n".join(lines)


def _load_or_build_docmap(path: Path, refresh: bool, client: OpenAI) -> dict:
    """
    Load the document map from cache, or call GPT-4o-mini to build it.
    Cache file: COREP/.docmaps/<stem>.json
    """
    import json

    _DOCMAP_DIR.mkdir(exist_ok=True)
    cache_path = _DOCMAP_DIR / (path.stem + ".json")

    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    print(f"    [LLM] Building document map for {path.name} ...")
    raw_text = _extract_raw_text(path)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user", "content": _DOCMAP_PROMPT + raw_text},
        ],
        response_format={"type": "json_object"},
    )

    import json
    raw_json = resp.choices[0].message.content
    try:
        doc_map = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"    [LLM] WARNING: JSON parse failed ({e}); using empty map for {path.name}")
        doc_map = {"sections": [], "templates": {}}

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(doc_map, f, indent=2, ensure_ascii=False)

    print(f"    [LLM] Map cached → {cache_path}")
    return doc_map


def _find_template_for_chunk(content: str, templates: dict) -> dict | None:
    """
    Return the template info dict most relevant to this chunk, or None.
    Matches by short code appearance in the content text.
    """
    # Direct key match first (short codes like LRCalc, LR1, …)
    for code, info in templates.items():
        if re.search(r'\b' + re.escape(code) + r'\b', content):
            return info
    # Fallback: scan for any template-like code and match by name substring
    found = _TEMPLATE_CODE_RE.findall(content)
    for code in found:
        code_clean = code.replace(" ", "").upper()
        for key, info in templates.items():
            if code_clean in key.upper() or code_clean in info.get("name", "").upper():
                return info
    return None


def _find_section_for_chunk(content: str, sections: list[dict]) -> dict | None:
    """
    Return the deepest section whose heading text appears (or is partly found)
    in the chunk content.
    """
    best: dict | None = None
    for sec in sections:
        heading = sec.get("heading", "")
        # Try progressively shorter fragments of the heading
        words = heading.split()
        for length in range(len(words), max(1, len(words) - 3) - 1, -1):
            fragment = " ".join(words[:length])
            if len(fragment) > 8 and fragment.lower() in content.lower():
                if best is None or sec.get("level", 99) >= best.get("level", 0):
                    best = sec
                break
    return best


# Raw-text length below which a chunk is a candidate for merging with its neighbour.
_LLM_MERGE_THRESHOLD = 500   # chars of raw text (breadcrumb/context line excluded)


def _merge_small_llm_chunks(items: list[dict]) -> list[dict]:
    """
    Merge consecutive chunks whose raw_text is below _LLM_MERGE_THRESHOLD,
    as long as the combined raw_text stays within MAX_CHUNK.

    Each item is a dict with keys: breadcrumb, context_line (str|None), raw_text, annex.
    Returns final list with those same keys; caller rebuilds the content string.
    Table rows ([Table] prefix) are never merged into non-table text and vice versa,
    because they have a different semantic structure.
    """
    if not items:
        return items

    result: list[dict] = []
    pending = dict(items[0])

    for nxt in items[1:]:
        p_raw = pending["raw_text"]
        n_raw = nxt["raw_text"]
        p_is_table = p_raw.lstrip().startswith("[Table]")
        n_is_table = n_raw.lstrip().startswith("[Table]")

        can_merge = (
            not p_is_table and not n_is_table          # don't merge table rows
            and (len(p_raw) < _LLM_MERGE_THRESHOLD
                 or len(n_raw) < _LLM_MERGE_THRESHOLD) # at least one is small
            and len(p_raw) + 1 + len(n_raw) <= MAX_CHUNK  # combined fits
        )

        if can_merge:
            pending["raw_text"] = p_raw + "\n" + n_raw
            # Keep the context_line of the first chunk; update breadcrumb if
            # the next chunk has a more specific one (longer = more detail).
            if len(nxt["breadcrumb"]) > len(pending["breadcrumb"]):
                pending["breadcrumb"] = nxt["breadcrumb"]
            if pending["context_line"] is None and nxt["context_line"] is not None:
                pending["context_line"] = nxt["context_line"]
        else:
            result.append(pending)
            pending = dict(nxt)

    result.append(pending)
    return result


def parse_docx_llm(path: Path, refresh_maps: bool = False) -> list[dict]:
    """
    Parse a COREP DOCX using the custom chunker as a base, then enrich each chunk
    with context from a GPT-4o-mini-generated document map.

    Pipeline:
      1. _extract_raw_text()  →  plain-text doc for LLM
      2. _load_or_build_docmap()  →  JSON {sections, templates} (cached on disk)
      3. parse_docx()  →  base chunks (standard breadcrumb + text)
      4. For each chunk: detect referenced template / section, rebuild breadcrumb,
         prepend a Context: line for table/template chunks.
      5. Merge consecutive small (< _LLM_MERGE_THRESHOLD chars) non-table chunks.
    """
    from openai import OpenAI as _OpenAI
    client = _OpenAI()

    meta  = parse_filename_meta(path.name)
    annex = meta["annex"]
    topic = meta["topic"]

    doc_map   = _load_or_build_docmap(path, refresh=refresh_maps, client=client)
    templates = doc_map.get("templates", {})
    sections  = doc_map.get("sections", [])

    # Base chunks from the custom parser (provides text + table structure)
    base_chunks = parse_docx(path)

    # Build intermediate items: (breadcrumb, context_line|None, raw_text, annex)
    items: list[dict] = []
    last_tmpl: dict | None = None   # carry forward: table rows inherit template context

    for chunk in base_chunks:
        content  = chunk["content"]
        # Strip existing breadcrumb prefix to work with the raw text
        raw_text = content.split("\n", 1)[1] if "\n" in content else content

        # Determine best section heading from LLM map
        sec  = _find_section_for_chunk(raw_text, sections)
        tmpl = _find_template_for_chunk(raw_text, templates)

        # Table chunks use {row;col} notation without a template name prefix.
        # Inherit the last seen template so they get context without a code reference.
        if tmpl:
            last_tmpl = tmpl
        elif raw_text.lstrip().startswith("[Table]"):
            tmpl = last_tmpl

        # Build an enriched heading path
        heading_parts: list[str] = []
        if sec:
            heading_parts.append(sec["heading"])
        if tmpl and tmpl.get("section_heading") and tmpl["section_heading"] not in heading_parts:
            heading_parts.append(tmpl["section_heading"])

        bc           = make_breadcrumb(annex, topic, heading_parts)
        context_line = (f"Context: {tmpl['name']} — {tmpl['description']}"
                        if tmpl else None)

        items.append({
            "breadcrumb":   bc,
            "context_line": context_line,
            "raw_text":     raw_text,
            "annex":        annex,
        })

    # Merge consecutive small chunks
    items = _merge_small_llm_chunks(items)

    # Rebuild final content strings
    enriched: list[dict] = []
    for item in items:
        bc   = item["breadcrumb"]
        ctx  = item["context_line"]
        raw  = item["raw_text"]
        content = f"{bc}\n{ctx}\n{raw}" if ctx else f"{bc}\n{raw}"
        enriched.append({"content": content, "breadcrumb": bc, "annex": item["annex"]})

    return enriched


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def migrate_schema(cur):
    """Create documents table if needed and add metadata columns."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          SERIAL PRIMARY KEY,
            content     TEXT,
            embedding   vector(1536),
            source      TEXT DEFAULT 'CRR',
            annex       TEXT,
            breadcrumb  TEXT
        );
    """)
    cur.execute("""
        ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS source      TEXT DEFAULT 'CRR',
            ADD COLUMN IF NOT EXISTS annex       TEXT,
            ADD COLUMN IF NOT EXISTS breadcrumb  TEXT;
    """)


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest COREP DOCX files into the RAG vector database."
    )
    parser.add_argument(
        "--db", choices=["local", "prod"], required=True,
        help="'local' → DATABASE_URL_LOCAL, 'prod' → DATABASE_URL"
    )
    parser.add_argument(
        "--parser", choices=["custom", "docling", "llm"], default="custom",
        help="Chunking backend: 'custom' (default) = hand-crafted style parser; "
             "'docling' = IBM Docling HybridChunker; "
             "'llm' = custom chunks enriched by a GPT-4o-mini document map."
    )
    parser.add_argument(
        "--refresh-maps", action="store_true",
        help="Force regeneration of LLM document maps even if cached (llm parser only)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and report chunk stats without writing to the database."
    )
    parser.add_argument(
        "--file", default=None, metavar="PATTERN",
        help="Only ingest files whose name contains PATTERN (case-insensitive)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print paragraph style distribution and first 3 chunks per file."
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Skip the interactive production confirmation prompt."
    )
    args = parser.parse_args()

    # --- Select DB URL ---
    if args.db == "local":
        db_url = os.getenv("DATABASE_URL_LOCAL")
        if not db_url:
            raise SystemExit(
                "DATABASE_URL_LOCAL is not set.\n"
                "Add it to .env:  DATABASE_URL_LOCAL=postgresql://user:pass@localhost:5432/dbname"
            )
        db_label = "LOCAL"
    else:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise SystemExit("DATABASE_URL is not set.")
        db_label = "PRODUCTION"

    # --- Collect DOCX files ---
    docx_files = sorted(COREP_DIR.glob("*.docx"))
    if args.file:
        pattern = args.file.lower()
        docx_files = [f for f in docx_files if pattern in f.name.lower()]
    if not docx_files:
        raise SystemExit(f"No matching DOCX files found in {COREP_DIR}/")

    if args.parser == "docling":
        parse_fn = lambda p: parse_docx_docling(p)
    elif args.parser == "llm":
        parse_fn = lambda p: parse_docx_llm(p, refresh_maps=args.refresh_maps)
    else:
        parse_fn = lambda p: parse_docx(p, verbose=args.verbose)

    print(f"Target:  {db_label}")
    print(f"Parser:  {args.parser}")
    print(f"Mode:    {'DRY RUN — no DB writes' if args.dry_run else 'LIVE INGEST'}")
    print(f"Files:   {len(docx_files)}")
    print()

    # --- Parse all files ---
    all_chunks: list[dict] = []
    for docx_path in docx_files:
        print(f"  Parsing: {docx_path.name}")
        file_chunks = parse_fn(docx_path)
        print(f"    -> {len(file_chunks)} chunks")

        if args.verbose:
            for i, c in enumerate(file_chunks[:3]):
                print(f"\n    [Sample chunk {i+1}] ({len(c['content'])} chars)")
                preview = c["content"][:300].replace("\n", " | ")
                print(f"    {preview}")

        all_chunks.extend(file_chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    if args.dry_run:
        print("\n--- Dry run complete. No data written. ---")
        return

    # --- Confirm before touching prod ---
    if args.db == "prod" and not args.confirm:
        confirm = input(
            "\nYou are about to write to PRODUCTION. Type 'yes' to continue: "
        ).strip().lower()
        if confirm != "yes":
            raise SystemExit("Aborted.")

    # --- Phase 1: Embed all chunks (no DB connection held during API calls) ---
    client = OpenAI()
    texts = [c["content"] for c in all_chunks]
    all_embeddings: list[list[float]] = []

    print(f"Embedding {len(all_chunks)} chunks (batch size {BATCH_SIZE})...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        all_embeddings.extend(embed_batch(client, batch_texts))
        done = min(i + BATCH_SIZE, len(all_chunks))
        print(f"  {done}/{len(all_chunks)} embedded...")

    # --- Phase 2: Open DB, migrate, delete stale rows, insert — all in one connection ---
    print("\nConnecting to database...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.autocommit = False
    register_vector(conn)
    cur = conn.cursor()

    print("Migrating schema (adding source/annex/breadcrumb columns if needed)...")
    migrate_schema(cur)
    conn.commit()

    cur.execute("DELETE FROM documents WHERE source = 'COREP'")
    deleted = cur.rowcount
    conn.commit()
    print(f"Removed {deleted} existing COREP rows.")

    cur.close()
    conn.close()

    # Insert with per-batch connections + retry on connection drop.
    # execute_values sends one INSERT per batch (single round-trip).
    MAX_RETRIES = 6
    print(f"Inserting {len(all_chunks)} chunks (batch={BATCH_SIZE}, max_retries={MAX_RETRIES})...")
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch_end    = min(batch_start + BATCH_SIZE, len(all_chunks))
        batch_chunks = all_chunks[batch_start:batch_end]
        batch_embs   = all_embeddings[batch_start:batch_end]

        values = [
            (c["content"], emb, c.get("annex"), c.get("breadcrumb"))
            for c, emb in zip(batch_chunks, batch_embs)
        ]

        for attempt in range(MAX_RETRIES):
            try:
                conn = psycopg2.connect(db_url)
                conn.autocommit = False
                register_vector(conn)
                cur = conn.cursor()
                execute_values(
                    cur,
                    """
                    INSERT INTO documents (content, embedding, source, annex, breadcrumb)
                    VALUES %s
                    """,
                    values,
                    template="(%s, %s::vector, 'COREP', %s, %s)",
                )
                conn.commit()
                cur.close()
                conn.close()
                break  # success
            except psycopg2.OperationalError as e:
                try:
                    conn.close()
                except Exception:
                    pass
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"  Connection error at {batch_end} (attempt {attempt + 1}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        print(f"  {batch_end}/{len(all_chunks)} inserted...")

    print("\nDone. COREP documents are indexed and ready to query.")


if __name__ == "__main__":
    main()
