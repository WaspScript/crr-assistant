"""
review_crr.py — Dump CRR chunks to text files for review.

Supports both file formats:
  crr.html   — original EUR-Lex HTML (oj-* classes)
  CRR.mhtml  — EUR-Lex MHTML web archive (title-division-* / title-article-norm classes)

Auto-detects the file and format.  Use --file to override.

Produces two files:
  chunks_review_crr_current_<stem>.txt    — as-is (same logic as ingest.py)
  chunks_review_crr_structured_<stem>.txt — structure-aware (Part/Title/Chapter/Section breadcrumbs)

Usage:
    .venv/Scripts/python review_crr.py
    .venv/Scripts/python review_crr.py --file crr.html
    .venv/Scripts/python review_crr.py --file CRR.mhtml
"""

import sys
import re
import email
import argparse
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--file", default=None,
                help="Input file (crr.html or CRR.mhtml). Auto-detected if omitted.")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
if args.file:
    src_path = Path(args.file)
else:
    # Prefer MHTML if it exists
    src_path = Path("CRR.mhtml") if Path("CRR.mhtml").exists() else Path("crr.html")

if not src_path.exists():
    raise SystemExit(f"File not found: {src_path}")

print(f"Source: {src_path}")

# ---------------------------------------------------------------------------
# HTML extraction (handles both plain HTML and MHTML)
# ---------------------------------------------------------------------------

def load_html(path: Path) -> str:
    if path.suffix.lower() == ".mhtml":
        with open(path, "rb") as f:
            msg = email.message_from_bytes(f.read())
        part = next(
            (p for p in msg.walk() if p.get_content_type() == "text/html"),
            None,
        )
        if not part:
            raise SystemExit("No text/html part found in MHTML file.")
        charset = part.get_content_charset() or "utf-8"
        return part.get_payload(decode=True).decode(charset, errors="replace")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
# Two EUR-Lex HTML flavours:
#   "old"  — crr.html:   oj-ti-art, oj-sti-art, oj-ti-section-1, oj-ti-section-2
#   "new"  — CRR.mhtml:  title-article-norm, eli-title (direct), title-division-1/2

class _Fmt:
    """Holds CSS class names for one HTML flavour."""
    def __init__(self, art_cls, sub_cls, sec1_cls, sec2_cls, sub_in_eli_title, sec2_in_eli_title):
        self.art_cls          = art_cls          # article title <p> class
        self.sub_cls          = sub_cls          # article subtitle class (or None)
        self.sec1_cls         = sec1_cls         # section label class (PART ONE, CHAPTER 1…)
        self.sec2_cls         = sec2_cls         # section name class
        self.sub_in_eli_title = sub_in_eli_title # subtitle is inside a div.eli-title
        self.sec2_in_eli_title= sec2_in_eli_title# sec2 is inside a div.eli-title

FMT_OLD = _Fmt(
    art_cls           = "oj-ti-art",
    sub_cls           = "oj-sti-art",
    sec1_cls          = "oj-ti-section-1",
    sec2_cls          = "oj-ti-section-2",
    sub_in_eli_title  = True,
    sec2_in_eli_title = True,
)

FMT_NEW = _Fmt(
    art_cls           = "title-article-norm",
    sub_cls           = None,        # subtitle is the text of div.eli-title directly
    sec1_cls          = "title-division-1",
    sec2_cls          = "title-division-2",
    sub_in_eli_title  = False,
    sec2_in_eli_title = False,
)

# Modification markers in the new format (e.g. ▼M8, ►M17, ▼C2)
_MOD_RE = re.compile(r"[▼►][A-Z]\d*")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _clean(text: str, fmt: _Fmt) -> str:
    """Strip modification markers and normalize whitespace for new format."""
    if fmt is FMT_NEW:
        text = _MOD_RE.sub("", text)
        text = text.replace("\xa0", " ")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
print("Parsing HTML ...")
html = load_html(src_path)
soup = BeautifulSoup(html, "lxml")

# Detect format
fmt = FMT_NEW if soup.find("p", class_="title-article-norm") else FMT_OLD

print(f"Format: {'new (title-division-* / title-article-norm)' if fmt is FMT_NEW else 'old (oj-ti-* classes)'}")

article_divs = [
    div for div in soup.find_all("div", class_="eli-subdivision")
    if div.find("p", class_=fmt.art_cls, recursive=False)
]
print(f"Articles found: {len(article_divs)}")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ---------------------------------------------------------------------------
# Article text extraction (does NOT modify the soup)
# ---------------------------------------------------------------------------

def _article_text(div, fmt: _Fmt) -> tuple[str, str, str]:
    """Returns (title, subtitle, body) without modifying the soup."""
    title_tag = div.find("p", class_=fmt.art_cls, recursive=False)
    title = _clean(title_tag.get_text(), fmt) if title_tag else ""

    if fmt.sub_in_eli_title:
        eli = div.find("div", class_="eli-title", recursive=False)
        sub_tag = eli.find(class_=fmt.sub_cls) if eli else None
        subtitle = _clean(sub_tag.get_text(), fmt) if sub_tag else ""
    elif fmt is FMT_NEW:
        eli = div.find("div", class_="eli-title", recursive=False)
        subtitle = _clean(eli.get_text(), fmt) if eli else ""
    else:
        sub_tag = div.find(class_=fmt.sub_cls, recursive=False)
        subtitle = _clean(sub_tag.get_text(), fmt) if sub_tag else ""

    full_text = _clean(div.get_text(separator="\n"), fmt)
    body = full_text
    if body.startswith(title):
        body = body[len(title):].lstrip("\n ")
    if subtitle and body.startswith(subtitle):
        body = body[len(subtitle):].lstrip("\n ")
    return title, subtitle, body.strip()


# ---------------------------------------------------------------------------
# Section hierarchy breadcrumb extraction
# ---------------------------------------------------------------------------

def _section_heading(anc, fmt: _Fmt) -> tuple[str, str] | None:
    """If ancestor div is a section container, return (label, name)."""
    children = list(anc.children)
    s1 = next(
        (c for c in children if hasattr(c, "get") and fmt.sec1_cls in c.get("class", [])),
        None,
    )
    if not s1:
        return None
    s1t = _clean(s1.get_text(), fmt)

    if fmt.sec2_in_eli_title:
        eli = next(
            (c for c in children if hasattr(c, "get") and "eli-title" in c.get("class", [])),
            None,
        )
        s2_tag = eli.find(class_=fmt.sec2_cls) if eli else None
        s2t = _clean(s2_tag.get_text(), fmt) if s2_tag else ""
    else:
        s2 = next(
            (c for c in children if hasattr(c, "get") and fmt.sec2_cls in c.get("class", [])),
            None,
        )
        s2t = _clean(s2.get_text(), fmt) if s2 else ""

    return (s1t, s2t)


def _make_crr_breadcrumb(article_div, fmt: _Fmt) -> str:
    headings: list[tuple[str, str]] = []
    for anc in article_div.parents:
        h = _section_heading(anc, fmt)
        if h:
            headings.append(h)
    headings.reverse()
    parts = ["CRR"] + [s1 + (": " + s2 if s2 else "") for s1, s2 in headings]
    return "[" + " | ".join(parts) + "]"


# ---------------------------------------------------------------------------
# As-is parser (replicates ingest.py logic)
# ---------------------------------------------------------------------------

def crr_current_chunks() -> list[str]:
    chunks: list[str] = []
    for div in article_divs:
        title, subtitle, body = _article_text(div, fmt)
        if not body:
            continue
        header = title + (" - " + subtitle if subtitle else "")
        full   = header + "\n" + body
        for part in splitter.split_text(full):
            if part.strip() == header.strip():
                continue
            if not part.startswith(header):
                part = header + "\n" + part
            chunks.append(part)
    return chunks


# ---------------------------------------------------------------------------
# Structure-aware parser
# ---------------------------------------------------------------------------

def crr_structured_chunks() -> list[dict]:
    chunks: list[dict] = []
    for div in article_divs:
        title, subtitle, body = _article_text(div, fmt)
        if not body:
            continue
        article_header = title + (" - " + subtitle if subtitle else "")
        breadcrumb     = _make_crr_breadcrumb(div, fmt)
        full_prefix    = breadcrumb + "\n" + article_header
        full_text      = full_prefix + "\n" + body
        for part in splitter.split_text(full_text):
            if part.strip() in (full_prefix.strip(), article_header.strip()):
                continue
            if not part.startswith(breadcrumb):
                part = full_prefix + "\n" + part
            chunks.append({"content": part, "breadcrumb": breadcrumb, "article_header": article_header})
    return chunks


# ---------------------------------------------------------------------------
# Write review files
# ---------------------------------------------------------------------------
stem = (src_path.stem + "_" + src_path.suffix.lstrip(".")).lower().replace(".", "_")

print("Generating as-is chunks ...")
current = crr_current_chunks()
print(f"  {len(current)} chunks")

out_current = Path(f"chunks_review_crr_current_{stem}.txt")
with open(out_current, "w", encoding="utf-8") as f:
    f.write(f"CRR CHUNKS — AS-IS | source: {src_path.name}\n")
    f.write(f"Total: {len(current)} chunks | chunk_size={CHUNK_SIZE} | overlap={CHUNK_OVERLAP}\n")
    f.write("=" * 80 + "\n\n")
    for i, chunk in enumerate(current, 1):
        f.write(f"--- chunk {i} ({len(chunk)} chars) ---\n")
        f.write(chunk)
        f.write("\n\n")

print("Generating structure-aware chunks ...")
structured = crr_structured_chunks()
print(f"  {len(structured)} chunks")

out_structured = Path(f"chunks_review_crr_structured_{stem}.txt")
with open(out_structured, "w", encoding="utf-8") as f:
    f.write(f"CRR CHUNKS — STRUCTURE-AWARE | source: {src_path.name}\n")
    f.write(f"Total: {len(structured)} chunks | chunk_size={CHUNK_SIZE} | overlap={CHUNK_OVERLAP}\n")
    f.write("=" * 80 + "\n\n")
    for i, chunk in enumerate(structured, 1):
        c = chunk["content"]
        f.write(f"--- chunk {i} ({len(c)} chars) ---\n")
        f.write(c)
        f.write("\n\n")

print(f"\nWrote:")
print(f"  {out_current} ({len(current)} chunks)")
print(f"  {out_structured} ({len(structured)} chunks)")
