"""
corep_ingest.py — Ingest COREP regulatory DOCX files into the RAG vector database.

Usage:
    python corep_ingest.py --db local                    # use DATABASE_URL_LOCAL
    python corep_ingest.py --db prod                     # use DATABASE_URL
    python corep_ingest.py --db local --dry-run          # parse only, no DB writes
    python corep_ingest.py --db local --file "Capital"   # ingest one matching file
    python corep_ingest.py --db local --dry-run --verbose  # show every chunk

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
from pgvector.psycopg2 import register_vector

load_dotenv(override=False)  # shell env vars take precedence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COREP_DIR    = Path("COREP")
EMBED_MODEL  = "text-embedding-3-small"
BATCH_SIZE   = 50
MAX_CHUNK    = 1200   # chars — flush buffer if it grows beyond this
MIN_CHUNK    = 60     # chars — skip chunks shorter than this

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
        if len(row_text) < MIN_CHUNK:
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
            if len(current.strip()) >= MIN_CHUNK:
                chunks.append({"content": f"{breadcrumb}\n{current.strip()}",
                               "breadcrumb": breadcrumb, "annex": annex})
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current.strip() and len(current.strip()) >= MIN_CHUNK:
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
        if len(text) < MIN_CHUNK:
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
                if len(text) >= MIN_CHUNK:
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

    return chunks


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

    print(f"Target:  {db_label}")
    print(f"Mode:    {'DRY RUN — no DB writes' if args.dry_run else 'LIVE INGEST'}")
    print(f"Files:   {len(docx_files)}")
    print()

    # --- Parse all files ---
    all_chunks: list[dict] = []
    for docx_path in docx_files:
        print(f"  Parsing: {docx_path.name}")
        file_chunks = parse_docx(docx_path, verbose=args.verbose)
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
    if args.db == "prod":
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

    print(f"Inserting {len(all_chunks)} chunks...")
    for i, (chunk, emb) in enumerate(zip(all_chunks, all_embeddings)):
        cur.execute(
            """
            INSERT INTO documents (content, embedding, source, annex, breadcrumb)
            VALUES (%s, %s, 'COREP', %s, %s)
            """,
            (chunk["content"], emb, chunk.get("annex"), chunk.get("breadcrumb"))
        )
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(all_chunks):
            conn.commit()
            print(f"  {i + 1}/{len(all_chunks)} inserted...")

    cur.close()
    conn.close()
    print("\nDone. COREP documents are indexed and ready to query.")


if __name__ == "__main__":
    main()
