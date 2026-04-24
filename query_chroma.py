# query_chroma.py
# Query logic for exam_qa_pairs Chroma collection.
# Used as a library by app.py (Streamlit) and can also be run standalone as a CLI.

import os
from typing import Optional

from dotenv import load_dotenv
import openai
import chromadb

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in .env")
openai.api_key = OPENAI_API_KEY

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "exam_qa_pairs"
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
EMB_MODEL = "text-embedding-3-small"


# ──────────────────────────────────────────────
# Chroma connection (cached for Streamlit)
# ──────────────────────────────────────────────

_collection_cache: dict = {}

def get_collection(chroma_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    """Return a Chroma collection, using a module-level cache to avoid reconnecting."""
    key = f"{chroma_dir}::{collection_name}"
    if key not in _collection_cache:
        client = chromadb.PersistentClient(path=chroma_dir)
        names = [c.name for c in client.list_collections()]
        if collection_name not in names:
            raise ValueError(
                f"Collection '{collection_name}' not found in {chroma_dir}. "
                "Run ingest_chroma.py first."
            )
        _collection_cache[key] = client.get_collection(collection_name)
    return _collection_cache[key]


# ──────────────────────────────────────────────
# Catalogue helpers
# ──────────────────────────────────────────────

def list_papers(collection=None) -> dict[str, list[str]]:
    """
    Return {paper_id: [sorted question numbers]} for everything in the collection.
    """
    col = collection or get_collection()
    res = col.get(include=["metadatas"])
    found: dict[str, set] = {}
    for md in res["metadatas"]:
        pid = md.get("paper_id", "paper_unknown")
        qnum = md.get("question_number", "")
        found.setdefault(pid, set()).add(qnum)
    return {pid: sorted(qnums) for pid, qnums in sorted(found.items())}


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    resp = openai.embeddings.create(model=EMB_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_by_question(
    paper_id: str,
    question_number: str,
    top_k: int = 1,
    collection=None,
) -> list[tuple[str, dict, float]]:
    """
    Strict lookup: filter by paper_id AND question_number, then rank by similarity.
    Returns list of (document_text, metadata, distance).
    """
    col = collection or get_collection()
    embedding = _embed(f"question {question_number}")
    results = col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={
            "$and": [
                {"paper_id": {"$eq": paper_id}},
                {"question_number": {"$eq": question_number}},
            ]
        },
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return list(zip(docs, metas, dists))


def retrieve_semantic(
    query_text: str,
    paper_id: Optional[str] = None,
    top_k: int = 5,
    collection=None,
) -> list[tuple[str, dict, float]]:
    """
    Semantic search across the whole collection (or one paper if paper_id given).
    Useful when the user describes a topic rather than knowing the question number.
    Returns list of (document_text, metadata, distance).
    """
    col = collection or get_collection()
    embedding = _embed(query_text)
    where = {"paper_id": {"$eq": paper_id}} if paper_id else None
    kwargs = dict(query_embeddings=[embedding], n_results=top_k)
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return list(zip(docs, metas, dists))


# ──────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────

def build_prompt(user_question: str, chunks: list[tuple[str, dict, float]]) -> str:
    context_parts = []
    for i, (doc, md, _dist) in enumerate(chunks, start=1):
        qnum = md.get("question_number", "?")
        paper = md.get("paper_id", "?")
        context_parts.append(
            f"--- CONTEXT {i} [paper: {paper}, question: {qnum}] ---\n{doc}"
        )
    context = "\n\n".join(context_parts)
    return f"""You are an expert exam tutor. Use ONLY the context blocks below to answer the student's question.
Do not use any external knowledge. If the context is insufficient, say so clearly.
Cite which CONTEXT block(s) you used in your answer (e.g. "From CONTEXT 1...").

Student question:
{user_question}

{context}

Answer clearly and concisely, as if explaining to a student revising for an exam."""


# ──────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────

def ask_llm(prompt: str, model: str = LLM_MODEL) -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful exam tutor constrained to provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=768,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# High-level helpers (used by Streamlit app)
# ──────────────────────────────────────────────

def answer_question(
    user_input: str,
    paper_id: str,
    question_number: str,
    model: str = LLM_MODEL,
) -> tuple[str, list[tuple[str, dict, float]]]:
    """
    Retrieve the relevant chunk for paper_id / question_number, then ask the LLM.
    Returns (answer_text, chunks_used).
    """
    chunks = retrieve_by_question(paper_id, question_number, top_k=1)
    if not chunks:
        return "⚠️ No data found for that question. Please check paper and question selection.", []
    prompt = build_prompt(user_input, chunks)
    answer = ask_llm(prompt, model=model)
    return answer, chunks


def answer_topic_search(
    user_input: str,
    paper_id: Optional[str] = None,
    top_k: int = 3,
    model: str = LLM_MODEL,
) -> tuple[str, list[tuple[str, dict, float]]]:
    """
    Semantic search by topic, then ask the LLM across the top results.
    Returns (answer_text, chunks_used).
    """
    chunks = retrieve_semantic(user_input, paper_id=paper_id, top_k=top_k)
    if not chunks:
        return "⚠️ No relevant questions found. Try different search terms.", []
    prompt = build_prompt(user_input, chunks)
    answer = ask_llm(prompt, model=model)
    return answer, chunks


# ──────────────────────────────────────────────
# CLI (kept for debugging / scripted use)
# ──────────────────────────────────────────────

def _cli():
    papers = list_papers()
    if not papers:
        print("No papers found. Run ingest_chroma.py first.")
        return

    print("\n📘 Available papers:")
    for pid, qnums in papers.items():
        print(f"  {pid}  ({len(qnums)} questions)")

    paper_id = input("\nPaper ID (or 'exit'): ").strip()
    if paper_id.lower() in {"exit", "quit"}:
        return
    if paper_id not in papers:
        print("❌ Unknown paper.")
        return

    print(f"\nQuestions: {', '.join(papers[paper_id])}")
    print("\nOptions:")
    print("  • Enter a question number for strict lookup")
    print("  • Type 'search: <topic>' for semantic search")
    print("  • Type 'exit' to quit\n")

    while True:
        user_input = input("Query: ").strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if user_input.lower().startswith("search:"):
            topic = user_input[7:].strip()
            answer, chunks = answer_topic_search(topic, paper_id=paper_id)
        else:
            # Treat as a question number selection
            qnum = user_input
            if qnum not in papers[paper_id]:
                print(f"❌ '{qnum}' is not a valid question number for this paper.")
                continue
            followup = input(f"  What do you want to know about question {qnum}? (Enter = explain model answer): ").strip()
            if not followup:
                followup = f"Explain the model answer for question {qnum}."
            answer, chunks = answer_question(followup, paper_id, qnum)

        print(f"\n🤖 Answer:\n{answer}")
        print(f"\n(Based on {len(chunks)} chunk(s) retrieved)\n")


if __name__ == "__main__":
    _cli()
