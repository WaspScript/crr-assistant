import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv(override=True)

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def retrieve(question: str, top_k: int = 5) -> list[str]:
    """Embed the question and fetch the most relevant chunks from the DB."""
    client = get_client()
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    ).data[0].embedding

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(
        """SELECT content FROM documents
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (q_emb, top_k)
    )
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def _generate_answer(question: str, chunks: list[str], model: str, max_tokens: int = 1024) -> str:
    """Generate a final answer from the given chunks."""
    client = get_client()
    context = "\n\n---\n\n".join(chunks)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert on the Capital Requirements Regulation (CRR). "
                    "Answer questions using ONLY the provided excerpts from the regulation. "
                    "If the answer is not in the excerpts, say so clearly. "
                    "Be precise and cite article numbers when they appear in the excerpts."
                )
            },
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


def ask(question: str, model: str = "gpt-4o-mini", top_k: int = 5) -> dict:
    """Standard single-pass RAG."""
    chunks = retrieve(question, top_k)
    answer = _generate_answer(question, chunks, model)
    return {
        "answer": answer,
        "search_log": [{"query": question, "chunks_found": len(chunks)}],
        "total_chunks": len(chunks)
    }


def deep_think_ask(question: str, model: str = "gpt-4o-mini", top_k: int = 5, iterations: int = 2) -> dict:
    """
    Iterative retrieval: after the initial search, the model generates
    follow-up queries to gather additional relevant context. All unique
    chunks are combined for the final answer.
    """
    client = get_client()
    seen   = set()
    chunks = []
    search_log = []

    def add_unique(new_chunks: list[str]) -> int:
        added = 0
        for c in new_chunks:
            if c not in seen:
                seen.add(c)
                chunks.append(c)
                added += 1
        return added

    # ── Step 1: initial retrieval ──────────────────────────────────────
    initial = retrieve(question, top_k)
    found   = add_unique(initial)
    search_log.append({"query": question, "chunks_found": found})

    # ── Step 2: iterative follow-up queries ────────────────────────────
    for i in range(iterations):
        context_so_far = "\n\n---\n\n".join(chunks[:15])  # cap to avoid huge prompts

        followup_resp = client.chat.completions.create(
            model="gpt-4o-mini",   # always use mini for planning — fast & cheap
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal research assistant analyzing EU banking regulation (CRR). "
                        "Based on the question and context retrieved so far, generate 2-3 specific "
                        "search queries to find additional relevant regulation excerpts. "
                        "Return ONLY valid JSON: {\"queries\": [\"query1\", \"query2\", ...]}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question: {question}\n\n"
                        f"Context found so far (excerpts):\n{context_so_far[:2500]}\n\n"
                        f"Iteration {i + 1} of {iterations}: generate follow-up search queries."
                    )
                }
            ]
        )

        # Parse follow-up queries robustly
        follow_up_queries = []
        try:
            raw = followup_resp.choices[0].message.content.strip()
            # Extract JSON even if wrapped in markdown
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            data = json.loads(raw)
            if isinstance(data, dict):
                follow_up_queries = data.get("queries", [])[:3]
            elif isinstance(data, list):
                follow_up_queries = data[:3]
        except Exception:
            follow_up_queries = []

        for q in follow_up_queries:
            new = retrieve(q, top_k)
            found = add_unique(new)
            search_log.append({"query": q, "chunks_found": found})

    # ── Step 3: final answer with all collected context ─────────────────
    answer = _generate_answer(question, chunks, model, max_tokens=2048)

    return {
        "answer": answer,
        "search_log": search_log,
        "total_chunks": len(chunks)
    }
