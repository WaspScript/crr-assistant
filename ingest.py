import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv(override=True)

DB_URL = os.getenv("DATABASE_URL")
HTML_PATH = "crr.html"

print("Step 1: Parsing HTML structure...")
with open(HTML_PATH, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "lxml")

# Extract article blocks from eli-subdivision divs
# Each article has: oj-ti-art (e.g. "Article 92") + oj-sti-art (subtitle) + body text
article_blocks = []
for div in soup.find_all("div", class_="eli-subdivision"):
    title_tag    = div.find(class_="oj-ti-art")
    subtitle_tag = div.find(class_="oj-sti-art")

    if not title_tag:
        continue  # skip non-article subdivisions

    title    = title_tag.get_text(strip=True)       # e.g. "Article 92"
    subtitle = subtitle_tag.get_text(strip=True) if subtitle_tag else ""

    # Remove the title/subtitle tags so they don't duplicate in body text
    title_tag.decompose()
    if subtitle_tag:
        subtitle_tag.decompose()

    # Remove footnotes and navigation noise
    for tag in div(["script", "style", "meta", "link"]):
        tag.decompose()

    body = div.get_text(separator="\n", strip=True)

    # Build the article text: use single \n so splitter keeps header with body
    header = f"{title}" + (f" - {subtitle}" if subtitle else "")
    full   = f"{header}\n{body}".strip()

    if body.strip():  # skip articles with no body text
        article_blocks.append((header, full))

print(f"  Found {len(article_blocks)} articles")

# Step 2: Chunk within articles, always prefixing the article header
print("Step 2: Splitting into chunks (article-aware)...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = []
for header, article_text in article_blocks:
    parts = splitter.split_text(article_text)
    for part in parts:
        # Skip header-only chunks (no real content)
        if part.strip() == header.strip():
            continue
        # Continuation chunks won't start with the header → prepend it
        if not part.startswith(header):
            part = f"{header}\n{part}"
        chunks.append(part)

print(f"  Total chunks: {len(chunks)}")

# Step 3: Connect to DB
print("Step 3: Connecting to database...")
conn = psycopg2.connect(DB_URL)
register_vector(conn)
cur = conn.cursor()

cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
conn.commit()
print("  Connected and table cleared.")

# Step 4: Embed and insert
print("Step 4: Embedding and inserting chunks (this may take a few minutes)...")
client = OpenAI()

def embed_batch(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [r.embedding for r in resp.data]

BATCH_SIZE = 100
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    embeddings = embed_batch(batch)
    for text, emb in zip(batch, embeddings):
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, emb)
        )
    conn.commit()
    done = min(i + BATCH_SIZE, len(chunks))
    print(f"  Inserted {done}/{len(chunks)} chunks...")

cur.close()
conn.close()
print("\nDone! CRR is fully indexed and ready to query.")
