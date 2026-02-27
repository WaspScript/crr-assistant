import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv(override=True)

_client = None

# ── Option 1: CRR context always injected into every system prompt ────────
CRR_CONTEXT = (
    "The document you are searching is the Capital Requirements Regulation (CRR), "
    "officially Regulation (EU) No 575/2013 of the European Parliament and of the Council. "
    "It establishes uniform prudential requirements for credit institutions and investment firms, "
    "covering: own funds and capital ratios (Tier 1, Tier 2, CET1), credit risk, market risk, "
    "operational risk, leverage ratio, liquidity requirements (LCR, NSFR), large exposures, "
    "and Pillar 3 disclosure. It is complemented by the Capital Requirements Directive (CRD IV). "
    "Common abbreviations: CRR = Capital Requirements Regulation, "
    "CET1 = Common Equity Tier 1, AT1 = Additional Tier 1, T2 = Tier 2, "
    "LCR = Liquidity Coverage Ratio, NSFR = Net Stable Funding Ratio, "
    "RWA = Risk-Weighted Assets, SA = Standardised Approach, IRB = Internal Ratings-Based, "
    "EBA = European Banking Authority, ECB = European Central Bank."
)


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ── Option 2: Query rewriting ─────────────────────────────────────────────
def rewrite_query(question: str) -> str:
    """Rewrite user question into formal EU regulation search terms."""
    client = get_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=150,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer for EU banking regulation (CRR, Regulation EU 575/2013). "
                    "Rewrite the user question using formal regulatory terminology found in the document: "
                    "e.g. 'CRR' → 'Capital Requirements Regulation EU 575/2013', "
                    "'banks' → 'credit institutions', 'capital' → 'own funds', "
                    "'leverage' → 'leverage ratio Article 429', 'liquidity' → 'liquidity coverage ratio', "
                    "'what is' → 'definition subject matter scope', etc. "
                    "Output ONLY the rewritten query in 1-2 sentences, nothing else."
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return resp.choices[0].message.content.strip()


# ── Option 3: HyDE (Hypothetical Document Embedding) ─────────────────────
def hyde_embedding(question: str) -> list:
    """Generate a hypothetical regulation passage and return its embedding for search."""
    client = get_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=250,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an EU banking regulation expert. "
                    "Write a short passage (3-5 sentences) in the formal style of EU regulation text "
                    "(Regulation EU 575/2013 / CRR) that would answer the question. "
                    "Use article references, regulatory terms, and precise definitions. "
                    "Output ONLY the passage, nothing else."
                )
            },
            {"role": "user", "content": question}
        ]
    )
    hypothetical = resp.choices[0].message.content.strip()
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[hypothetical]
    ).data[0].embedding
    return emb


def _embed(text: str) -> list:
    """Embed a plain text string."""
    return get_client().embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    ).data[0].embedding


def retrieve(question: str, top_k: int = 5,
             query_rewrite: bool = False, hyde: bool = True) -> tuple:
    """
    Embed the question and fetch the most relevant chunks from the DB.
    Returns (chunks, effective_query_used).
    Options:
      query_rewrite: rewrite question into formal regulatory language first
      hyde: use hypothetical document embedding instead of direct question embedding
    """
    effective_query = question

    # Option 2: rewrite query into regulation-style language
    if query_rewrite:
        effective_query = rewrite_query(question)

    # Option 3: HyDE — embed a generated hypothetical answer instead of the query
    if hyde:
        embedding = hyde_embedding(effective_query)
    else:
        embedding = _embed(effective_query)

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(
        """SELECT content FROM documents
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (embedding, top_k)
    )
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows, effective_query


def _generate_answer(question: str, chunks: list, model: str, max_tokens: int = 1024) -> str:
    """Generate a final answer from the given chunks, always with CRR context."""
    client = get_client()
    context = "\n\n---\n\n".join(chunks)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{CRR_CONTEXT}\n\n"
                    "Answer questions using ONLY the provided excerpts from the regulation. "
                    "If the answer is not in the excerpts but is covered by the regulation context "
                    "above, answer based on that and note the specific text wasn't retrieved. "
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


def ask(question: str, model: str = "gpt-4o-mini", top_k: int = 5,
        query_rewrite: bool = False, hyde: bool = True) -> dict:
    """Standard single-pass RAG."""
    chunks, effective_query = retrieve(
        question, top_k, query_rewrite=query_rewrite, hyde=hyde
    )
    answer = _generate_answer(question, chunks, model)
    return {
        "answer": answer,
        "search_log": [{"query": effective_query, "chunks_found": len(chunks)}],
        "total_chunks": len(chunks)
    }


def deep_think_ask(question: str, model: str = "gpt-4o-mini", top_k: int = 5,
                   iterations: int = 2, query_rewrite: bool = False, hyde: bool = True) -> dict:
    """
    Iterative retrieval: after the initial search, the model generates
    follow-up queries to gather additional relevant context. All unique
    chunks are combined for the final answer.
    """
    client = get_client()
    seen   = set()
    chunks = []
    search_log = []

    def add_unique(new_chunks: list) -> int:
        added = 0
        for c in new_chunks:
            if c not in seen:
                seen.add(c)
                chunks.append(c)
                added += 1
        return added

    # ── Step 1: initial retrieval ──────────────────────────────────────
    initial, effective_query = retrieve(
        question, top_k, query_rewrite=query_rewrite, hyde=hyde
    )
    found = add_unique(initial)
    search_log.append({"query": effective_query, "chunks_found": found})

    # ── Step 2: iterative follow-up queries ────────────────────────────
    for i in range(iterations):
        context_so_far = "\n\n---\n\n".join(chunks[:15])

        followup_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal research assistant analyzing EU banking regulation (CRR). "
                        "Based on the question and context retrieved so far, generate 2-3 specific "
                        "search queries using formal regulatory terminology to find additional "
                        "relevant regulation excerpts. "
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

        follow_up_queries = []
        try:
            raw = followup_resp.choices[0].message.content.strip()
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
            # Follow-up queries: apply HyDE if enabled, skip query_rewrite (already formal)
            new_chunks, _ = retrieve(q, top_k, query_rewrite=False, hyde=hyde)
            found = add_unique(new_chunks)
            search_log.append({"query": q, "chunks_found": found})

    # ── Step 3: final answer with all collected context ─────────────────
    answer = _generate_answer(question, chunks, model, max_tokens=2048)

    return {
        "answer": answer,
        "search_log": search_log,
        "total_chunks": len(chunks)
    }
