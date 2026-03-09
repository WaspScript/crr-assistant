import os
import psycopg2
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from rag import ask, deep_think_ask

load_dotenv(override=True)

app = Flask(__name__)


def setup_db():
    """Ensure pgvector extension and documents table exist."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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
    # Non-destructive migration for databases created before metadata columns existed
    cur.execute("""
        ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS source      TEXT DEFAULT 'CRR',
            ADD COLUMN IF NOT EXISTS annex       TEXT,
            ADD COLUMN IF NOT EXISTS breadcrumb  TEXT;
    """)
    cur.close()
    conn.close()


setup_db()

MODELS = [
    {"id": "gpt-4o-mini",  "label": "GPT-4o mini",  "desc": "Fast & cheap"},
    {"id": "gpt-4o",       "label": "GPT-4o",        "desc": "Best quality"},
    {"id": "o3-mini",      "label": "o3-mini",       "desc": "Deep reasoning"},
]

@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/sources")
def sources():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        SELECT source, COUNT(*) AS chunks
        FROM documents
        GROUP BY source
        ORDER BY source
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([{"source": r[0], "chunks": r[1]} for r in rows])


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    question   = data.get("message", "").strip()
    model      = data.get("model", "gpt-4o-mini")
    top_k      = max(1, min(int(data.get("top_k", 5)), 20))
    deep_think    = bool(data.get("deep_think", False))
    iterations    = max(1, min(int(data.get("iterations", 2)), 5))
    query_rewrite = bool(data.get("query_rewrite", False))
    hyde          = bool(data.get("hyde", True))

    if not question:
        return jsonify({"error": "Empty message"}), 400

    # Validate model
    valid_ids = [m["id"] for m in MODELS]
    if model not in valid_ids:
        model = "gpt-4o-mini"

    try:
        if deep_think:
            result = deep_think_ask(question, model=model, top_k=top_k, iterations=iterations,
                                    query_rewrite=query_rewrite, hyde=hyde)
        else:
            result = ask(question, model=model, top_k=top_k,
                         query_rewrite=query_rewrite, hyde=hyde)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
