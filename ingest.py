import os
import re
import email
from dotenv import load_dotenv
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from pathlib import Path

load_dotenv(override=True)  # .env takes precedence over stale system env vars

DB_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------------------------
# File selection — prefer CRR.mhtml (2026 consolidated) over crr.html
# ---------------------------------------------------------------------------
_SRC = Path("CRR.mhtml") if Path("CRR.mhtml").exists() else Path("crr.html")

# ---------------------------------------------------------------------------
# HTML extraction (handles plain HTML and MHTML)
# ---------------------------------------------------------------------------

def _load_html(path: Path) -> str:
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
#   "old"  — crr.html:   oj-ti-art, oj-sti-art, oj-ti-section-1/2
#   "new"  — CRR.mhtml:  title-article-norm, eli-title, title-division-1/2

class _Fmt:
    def __init__(self, art_cls, sub_cls, sec1_cls, sec2_cls, sub_in_eli_title, sec2_in_eli_title):
        self.art_cls           = art_cls
        self.sub_cls           = sub_cls
        self.sec1_cls          = sec1_cls
        self.sec2_cls          = sec2_cls
        self.sub_in_eli_title  = sub_in_eli_title
        self.sec2_in_eli_title = sec2_in_eli_title

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
    sub_cls           = None,
    sec1_cls          = "title-division-1",
    sec2_cls          = "title-division-2",
    sub_in_eli_title  = False,
    sec2_in_eli_title = False,
)

_MOD_RE = re.compile(r"[▼►][A-Z]\d*")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _clean(text: str, fmt: _Fmt) -> str:
    if fmt is FMT_NEW:
        text = _MOD_RE.sub("", text)
        text = text.replace("\xa0", " ")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Article text extraction
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

def _section_heading(anc, fmt: _Fmt):
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


def _make_breadcrumb(article_div, fmt: _Fmt) -> str:
    headings = []
    for anc in article_div.parents:
        h = _section_heading(anc, fmt)
        if h:
            headings.append(h)
    headings.reverse()
    parts = ["CRR"] + [s1 + (": " + s2 if s2 else "") for s1, s2 in headings]
    return "[" + " | ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
print(f"Step 1: Parsing HTML structure from {_SRC} ...")
html = _load_html(_SRC)
soup = BeautifulSoup(html, "lxml")

fmt = FMT_NEW if soup.find("p", class_="title-article-norm") else FMT_OLD
print(f"  Format: {'new (CRR.mhtml)' if fmt is FMT_NEW else 'old (crr.html)'}")

article_divs = [
    div for div in soup.find_all("div", class_="eli-subdivision")
    if div.find("p", class_=fmt.art_cls, recursive=False)
]
print(f"  Found {len(article_divs)} articles")

# Step 2: Chunk within articles, always prefixing breadcrumb + article header
print("Step 2: Splitting into chunks (structure-aware)...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# chunks is list of (text, breadcrumb)
chunks = []
for div in article_divs:
    title, subtitle, body = _article_text(div, fmt)
    if not body:
        continue
    article_header = title + (" - " + subtitle if subtitle else "")
    breadcrumb     = _make_breadcrumb(div, fmt)
    full_prefix    = breadcrumb + "\n" + article_header
    full_text      = full_prefix + "\n" + body
    for part in splitter.split_text(full_text):
        if part.strip() in (full_prefix.strip(), article_header.strip()):
            continue
        if not part.startswith(breadcrumb):
            part = full_prefix + "\n" + part
        chunks.append((part, breadcrumb))

print(f"  Total chunks: {len(chunks)}")

# Step 3: Embed all chunks first (no DB held open during API calls)
print("Step 3: Embedding all chunks...")
client = OpenAI()

def embed_batch(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [r.embedding for r in resp.data]

BATCH_SIZE = 100
embeddings_all = []
for i in range(0, len(chunks), BATCH_SIZE):
    batch_texts = [c[0] for c in chunks[i:i + BATCH_SIZE]]
    embeddings_all.extend(embed_batch(batch_texts))
    print(f"  Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")

# Step 4: Connect to DB and insert
print("Step 4: Connecting to database and inserting...")

conn = psycopg2.connect(
    DB_URL,
    keepalives=1,
    keepalives_idle=10,
    keepalives_interval=5,
    keepalives_count=5,
)
register_vector(conn)
cur = conn.cursor()
cur.execute("DELETE FROM documents WHERE source = 'CRR'")
conn.commit()
print("  Old CRR chunks cleared.")

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    batch_embs = embeddings_all[i:i + BATCH_SIZE]
    rows = [(text, emb, "CRR", breadcrumb) for (text, breadcrumb), emb in zip(batch, batch_embs)]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO documents (content, embedding, source, breadcrumb) VALUES %s",
        rows,
    )
    conn.commit()
    done = min(i + BATCH_SIZE, len(chunks))
    print(f"  Inserted {done}/{len(chunks)} chunks...")

cur.close()
conn.close()
print("\nDone! CRR is fully indexed and ready to query.")
