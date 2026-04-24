# parse_exam_data.py
# LLM-powered exam parser — works across AQA, OCR, Edexcel and different years
# without any paper-specific regex.

import os
import re
import json
import argparse
import textwrap
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in .env")
openai.api_key = OPENAI_API_KEY

# Maximum characters sent to LLM per chunk.
# gpt-4o has a large context window so 40 000 chars is safe for most papers.
CHUNK_SIZE = 40_000

# ──────────────────────────────────────────────
# PDF extraction
# ──────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Return concatenated text from every page of a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ──────────────────────────────────────────────
# LLM parsing
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert at parsing structured data from exam papers.
    You will receive raw text extracted from a question paper and (optionally) its mark scheme.
    Return ONLY a valid JSON object — no markdown fences, no commentary.

    The JSON object must have a single key "questions" whose value is an array.
    Each element must have exactly these keys:
      - "number"   : question number string, e.g. "01.1", "3a", "Q4"
      - "section"  : section letter/name string, or null
      - "question" : full question text as a single string
      - "options"  : list of MCQ option strings (e.g. ["A …","B …"]) or null
      - "marks"    : integer mark allocation, or null
      - "answer"   : mark-scheme answer text as a single string, or null

    Rules:
    - Preserve all mathematical notation and special characters.
    - Merge any continuation lines that belong to the same question text.
    - If the mark scheme is not provided or does not match a question, set "answer" to null.
    - Do NOT invent data that is not present in the text.
""").strip()


def _call_llm_parse(qp_chunk: str, ms_chunk: Optional[str], model: str = "gpt-4o") -> list[dict]:
    """Send one chunk of text to the LLM and return parsed questions."""
    user_content = f"QUESTION PAPER:\n{qp_chunk}"
    if ms_chunk:
        user_content += f"\n\nMARK SCHEME:\n{ms_chunk}"

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # Belt-and-braces: strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    data = json.loads(raw)
    return data.get("questions", [])


def parse_with_llm(
    qp_text: str,
    ms_text: Optional[str] = None,
    paper_id: str = "paper_unknown",
    model: str = "gpt-4o",
) -> list[dict]:
    """
    Parse a full question paper (and optional mark scheme) using an LLM.
    Handles papers that exceed the chunk size by splitting on page boundaries.
    """
    # Split into chunks if the paper is very long
    qp_chunks = _split_text(qp_text, CHUNK_SIZE)
    ms_chunks = _split_text(ms_text, CHUNK_SIZE) if ms_text else [None] * len(qp_chunks)

    # Pad ms_chunks to match qp_chunks length
    while len(ms_chunks) < len(qp_chunks):
        ms_chunks.append(None)

    all_questions: list[dict] = []
    seen_numbers: set[str] = set()

    for i, (qp_chunk, ms_chunk) in enumerate(zip(qp_chunks, ms_chunks)):
        chunk_label = f"chunk {i+1}/{len(qp_chunks)}"
        print(f"  🤖 Parsing {chunk_label} ...")
        try:
            questions = _call_llm_parse(qp_chunk, ms_chunk, model=model)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  ⚠️  LLM returned invalid JSON for {chunk_label}: {exc}. Skipping.")
            continue

        for q in questions:
            # Attach paper_id to every question
            q["paper_id"] = paper_id
            # Deduplicate across chunks by question number
            num = q.get("number", "")
            if num and num in seen_numbers:
                continue
            if num:
                seen_numbers.add(num)
            all_questions.append(q)

    return all_questions


def _split_text(text: Optional[str], chunk_size: int) -> list[Optional[str]]:
    """Split text into chunks of at most chunk_size characters, preferring page breaks."""
    if not text:
        return [None]
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to split at a newline near the end of the chunk
        split_pos = text.rfind("\n", start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end
        chunks.append(text[start:split_pos])
        start = split_pos
    return chunks


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    # ── Defaults — edit these if you always use the same files ──────────────
    DEFAULT_QP    = "data/June 2018 QP.pdf"
    DEFAULT_MS    = "data/June 2018 MS.pdf"
    DEFAULT_OUT   = "data/parsed_exam.json"
    DEFAULT_MODEL = "gpt-4o"
    # ────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="Parse an exam question paper + mark scheme into structured JSON using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qp",    default=DEFAULT_QP,    help="Path to the question paper PDF")
    parser.add_argument("--ms",    default=DEFAULT_MS,    help="Path to the mark scheme PDF (pass empty string to skip)")
    parser.add_argument("--out",   default=DEFAULT_OUT,   help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use for parsing")
    args = parser.parse_args()

    # Allow --ms "" to explicitly mean "no mark scheme"
    if args.ms == "":
        args.ms = None

    qp_path = Path(args.qp)
    if not qp_path.exists():
        raise FileNotFoundError(f"Question paper not found: {qp_path}")

    # Derive a human-readable paper_id from the filename
    paper_id = qp_path.stem  # e.g. "June 2018 QP"

    print(f"\n📄 Extracting text from question paper: {qp_path}")
    qp_text = extract_text_from_pdf(str(qp_path))

    ms_text = None
    if args.ms:
        ms_path = Path(args.ms)
        if ms_path.exists():
            print(f"📄 Extracting text from mark scheme: {ms_path}")
            ms_text = extract_text_from_pdf(str(ms_path))
        else:
            print(f"⚠️  Mark scheme not found at {ms_path}, continuing without it.")

    print(f"\n🧠 Parsing with LLM ({args.model}) ...")
    questions = parse_with_llm(qp_text, ms_text, paper_id=paper_id, model=args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Parsing complete! Saved {len(questions)} questions → {out_path}")

    # Quick quality report
    answered = sum(1 for q in questions if q.get("answer"))
    marked = sum(1 for q in questions if q.get("marks") is not None)
    print(f"   📌 Questions with answers : {answered}/{len(questions)}")
    print(f"   📌 Questions with marks   : {marked}/{len(questions)}")


if __name__ == "__main__":
    main()
