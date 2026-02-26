import os
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv(override=True)

oai = OpenAI()

def retrieve(question: str, top_k: int = 5) -> list[str]:
    """Embed the question and fetch the most relevant chunks from the DB."""
    q_emb = oai.embeddings.create(
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

def ask(question: str) -> str:
    """Retrieve relevant chunks and ask GPT to answer based on them."""
    chunks = retrieve(question)
    context = "\n\n---\n\n".join(chunks)

    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "You are an expert on the Capital Requirements Regulation (CRR). Answer questions using ONLY the provided excerpts. If the answer is not in the excerpts, say so clearly."
            },
            {
                "role": "user",
                "content": f"""<context>
{context}
</context>

Question: {question}"""
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("CRR RAG System ready. Type your question (or 'quit' to exit).\n")
    while True:
        q = input("Question: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        print("\nSearching and generating answer...\n")
        answer = ask(q)
        print(f"Answer:\n{answer}\n")
        print("-" * 60)
