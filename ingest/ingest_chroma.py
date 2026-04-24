# ingest_chroma.py
# Ingests parsed exam JSON into ChromaDB.
# Uses deterministic content-based IDs + upsert so re-running is always safe.

import os
import json
import time
import hashlib
import argparse
from pathlib import Path

from dotenv import load_dotenv
import openai
import chromadb

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in .env")
openai.api_key = OPENAI_API_KEY

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "exam_qa_pairs"
EMB_MODEL = "text-embedding-3-small"
EMBED_BATCH = 16        # documents per embedding API call
EMBED_SLEEP = 0.1       # seconds between batches (rate-limit headroom)


# ──────────────────────────────────────────────
# Chroma setup
# ──────────────────────────────────────────────

def get_or_create_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        col = client.get_collection(collection_name)
        print(f"📂 Using existing Chroma collection '{collection_name}'")
    except Exception:
        col = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"🆕 Created Chroma collection '{collection_name}'")
    return col


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_document_text(q: dict) -> str:
    """Build the text representation stored in Chroma for one question."""
    question = (q.get("question") or "").strip()
    answer = (q.get("answer") or "").strip()
    options = q.get("options") or []

    parts = [f"Q: {question}"]
    if options:
        parts.append("Options:\n" + "\n".join(f"  {o}" for o in options))
    parts.append(f"A: {answer}" if answer else "A: (no model answer)")
    return "\n".join(parts)


def make_deterministic_id(paper_id: str, question_number: str, doc_text: str) -> str:
    """
    SHA-256 of 'paper_id::question_number::first_120_chars_of_doc'.
    Stable across runs so upsert can detect existing records correctly.
    """
    key = f"{paper_id}::{question_number}::{doc_text[:120]}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI, with batching and rate-limit sleep."""
    vectors = []
    total = len(texts)
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        batch_num = i // EMBED_BATCH + 1
        total_batches = (total + EMBED_BATCH - 1) // EMBED_BATCH
        print(f"  ✨ Embedding batch {batch_num}/{total_batches} ({len(batch)} docs)...")
        resp = openai.embeddings.create(model=EMB_MODEL, input=batch)
        for r in resp.data:
            vectors.append(r.embedding)
        time.sleep(EMBED_SLEEP)
    return vectors


# ──────────────────────────────────────────────
# Core ingest function
# ──────────────────────────────────────────────

def ingest(input_json: str, chroma_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    input_path = Path(input_json)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run parse_exam_data.py first."
        )

    print(f"\n📖 Loading parsed data from {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    questions = parsed.get("questions", [])
    if not questions:
        print("⚠️  No questions found in JSON. Nothing to ingest.")
        return

    print(f"   Found {len(questions)} questions.")

    # ── Build parallel arrays ──────────────────
    texts, metadatas, ids = [], [], []

    for q in questions:
        paper_id = (q.get("paper_id") or "paper_unknown").strip()
        q_number = str(q.get("number") or "unknown").strip()
        section = str(q.get("section") or "unknown").strip()
        marks_raw = q.get("marks")
        marks = str(marks_raw) if marks_raw is not None else "unknown"

        doc = make_document_text(q)
        doc_id = make_deterministic_id(paper_id, q_number, doc)

        texts.append(doc)
        ids.append(doc_id)
        metadatas.append(
            {
                "paper_id": paper_id,
                "question_number": q_number,
                "section": section,
                "marks": marks,
                "chunk_type": "qa_pair",
                # Store a plain-text version of options for display
                "has_options": str(bool(q.get("options"))),
            }
        )

    # ── Embed ──────────────────────────────────
    print(f"\n🔢 Embedding {len(texts)} documents with {EMB_MODEL} ...")
    embeddings = embed_texts(texts)

    # ── Upsert (safe for re-runs) ──────────────
    collection = get_or_create_collection(chroma_dir, collection_name)
    print(f"\n💾 Upserting {len(texts)} items into '{collection_name}' ...")
    collection.upsert(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    # ── Summary ───────────────────────────────
    total_in_collection = collection.count()
    print(f"\n✅ Done! Upserted {len(texts)} items.")
    print(f"   Collection '{collection_name}' now contains {total_in_collection} total documents.")
    print(f"   Chroma DB dir: {Path(chroma_dir).resolve()}")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest parsed exam JSON into ChromaDB (upsert — safe to re-run)."
    )
    parser.add_argument(
        "--input",
        default="data/parsed_exam.json",
        help="Path to the parsed exam JSON produced by parse_exam_data.py",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
        help=f"ChromaDB directory (default: {CHROMA_DIR})",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Collection name (default: {COLLECTION_NAME})",
    )
    args = parser.parse_args()

    ingest(
        input_json=args.input,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
