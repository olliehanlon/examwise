# app.py — Exam Tutor Streamlit UI
# Run with:  streamlit run app.py

import os
import sys
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Secrets: pull from st.secrets (Streamlit Cloud) or .env (local) ──────────
# On Streamlit Cloud, set these in App Settings → Secrets.
# Locally they come from your .env file via load_dotenv() above.
for key in ("OPENAI_API_KEY", "CHROMA_DB_DIR", "OPENAI_LLM_MODEL"):
    if key in st.secrets and key not in os.environ:
        os.environ[key] = st.secrets[key]

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Tutor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from sibling modules ─────────────────────────────────────────────
try:
    from parse_exam_data import extract_text_from_pdf, parse_with_llm
    from ingest_chroma import ingest
    from query_chroma import (
        list_papers,
        answer_question,
        answer_topic_search,
        retrieve_by_question,
    )
except ImportError as exc:
    st.error(
        f"**Import error:** `{exc}`\n\n"
        "Make sure `parse_exam_data.py`, `ingest_chroma.py`, and `query_chroma.py` "
        "are in the **same directory** as `app.py`, and that all dependencies are installed:\n\n"
        "```\npip install openai chromadb PyPDF2 streamlit python-dotenv\n```"
    )
    st.stop()
    sys.exit(1)  # fallback if st.stop() has no runtime to hook into


# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        line-height: 1.25;
        padding: 0.85rem 1rem 0.65rem 1rem;
        min-height: 3rem;
        display: flex;
        align-items: center;
        box-sizing: border-box;
    }
    .stTabs [data-baseweb="tab"] > div {
        line-height: 1.25;
    }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 6px; }
    .section-title {
        margin: 0.5rem 0 0.4rem 0;
        padding-top: 0.15rem;
        font-size: 2rem;
        line-height: 1.25;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def cached_list_papers():
    """Cache the paper catalogue so we don't hit Chroma on every re-render."""
    try:
        return list_papers()
    except ValueError:
        return {}


def reset_paper_cache():
    cached_list_papers.clear()


def catalogue_counts(papers: dict[str, list[str]]) -> tuple[int, int]:
    paper_count = len(papers)
    question_count = sum(len(qnums) for qnums in papers.values())
    return paper_count, question_count


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📘 Exam Tutor")
    st.markdown("---")
    st.caption("Steps: **Parse → Ingest → Query**")
    st.markdown("---")

    chroma_dir = st.text_input(
        "ChromaDB directory",
        value=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
        help="Where ChromaDB stores its data on disk.",
    )
    llm_model = st.selectbox(
        "LLM model (for answers)",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )
    parse_model = st.selectbox(
        "LLM model (for parsing)",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="gpt-4o gives the most accurate parsing.",
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_query, tab_parse, tab_ingest = st.tabs(["💬 Query", "📄 Parse Paper", "💾 Ingest"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query
# ════════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("Ask About Your Exam Papers")
    st.caption("Find a question by number, or search across papers by topic.")

    papers = cached_list_papers()

    if not papers:
        st.info(
            "No papers found in ChromaDB. "
            "Use the **Parse Paper** tab to parse a PDF, "
            "then **Ingest** it to load it into the database."
        )
        if st.button("Refresh catalogue", key="refresh_query_empty"):
            reset_paper_cache()
            st.rerun()
    else:
        # ── Mode toggle ─────────────────────────────────────────────────────
        paper_count, question_count = catalogue_counts(papers)
        st.caption(f"Indexed papers: {paper_count} | Indexed questions: {question_count}")
        mode = st.radio(
            "Search mode",
            ["By question number", "By topic (semantic search)"],
            horizontal=True,
        )

        if mode == "By question number":
            col1, col2 = st.columns(2)
            with col1:
                paper_id = st.selectbox(
                    "Paper",
                    list(papers.keys()),
                    key="query_paper",
                )
            with col2:
                question_number = st.selectbox(
                    "Question",
                    papers.get(paper_id, []),
                    key="query_question",
                )

            # Show stored Q&A
            with st.expander("📋 Stored question & model answer", expanded=False):
                chunks = retrieve_by_question(paper_id, question_number, top_k=1)
                if chunks:
                    st.text(chunks[0][0])
                else:
                    st.warning("No data found for this question.")

            user_input = st.text_area(
                "Your question (leave blank to explain the model answer)",
                placeholder=f"e.g. Why is the answer to {question_number} correct?",
                height=80,
                key="query_input_strict",
            )
            if st.button("Ask 🤖", key="ask_strict"):
                if not user_input.strip():
                    user_input = f"Explain the model answer for question {question_number}."
                with st.spinner("Thinking..."):
                    answer, used_chunks = answer_question(
                        user_input, paper_id, question_number, model=llm_model
                    )
                st.markdown("### Answer")
                st.markdown(answer)
                with st.expander("Source chunk used", expanded=False):
                    for doc, md, dist in used_chunks:
                        st.caption(
                            f"paper: {md.get('paper_id')}  |  "
                            f"question: {md.get('question_number')}  |  "
                            f"marks: {md.get('marks')}  |  "
                            f"distance: {dist:.4f}"
                        )
                        st.text(doc)

        else:  # Semantic / topic search
            paper_options = ["All papers"] + list(papers.keys())
            selected_paper = st.selectbox("Filter by paper (optional)", paper_options)
            paper_filter = None if selected_paper == "All papers" else selected_paper

            top_k = st.slider("Number of questions to retrieve", 1, 10, 3)

            user_input = st.text_area(
                "Describe a topic or ask a question",
                placeholder="e.g. What questions are about river erosion processes?",
                height=80,
                key="query_input_semantic",
            )
            if st.button("Search 🔍", key="ask_semantic"):
                if not user_input.strip():
                    st.warning("Please enter a search query.")
                else:
                    with st.spinner("Searching..."):
                        answer, used_chunks = answer_topic_search(
                            user_input,
                            paper_id=paper_filter,
                            top_k=top_k,
                            model=llm_model,
                        )
                    st.markdown("### Answer")
                    st.markdown(answer)
                    with st.expander(f"Top {len(used_chunks)} source chunks", expanded=False):
                        for doc, md, dist in used_chunks:
                            st.caption(
                                f"paper: {md.get('paper_id')}  |  "
                                f"question: {md.get('question_number')}  |  "
                                f"marks: {md.get('marks')}  |  "
                                f"distance: {dist:.4f}"
                            )
                            st.text(doc)
                            st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Parse Paper
# ════════════════════════════════════════════════════════════════════════════════
with tab_parse:
    st.markdown('<div class="section-title">Parse a New Exam Paper</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a question paper PDF and (optionally) its mark scheme. "
        "The LLM will extract every question and match it to the model answer."
    )
    st.caption("The output JSON can be reused later, so you only need to parse a paper once.")

    col_qp, col_ms = st.columns(2)
    with col_qp:
        qp_file = st.file_uploader("Question paper PDF *", type=["pdf"], key="qp_upload")
    with col_ms:
        ms_file = st.file_uploader("Mark scheme PDF (optional)", type=["pdf"], key="ms_upload")

    out_path = st.text_input(
        "Save parsed JSON to",
        value="data/parsed_exam.json",
        key="parse_out_path",
    )

    if st.button("Parse 🧠", key="parse_btn"):
        if not qp_file:
            st.warning("Please upload a question paper PDF.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write uploaded files to tmp
                qp_tmp = Path(tmpdir) / qp_file.name
                qp_tmp.write_bytes(qp_file.read())

                ms_tmp = None
                if ms_file:
                    ms_tmp = Path(tmpdir) / ms_file.name
                    ms_tmp.write_bytes(ms_file.read())

                paper_id = Path(qp_file.name).stem

                with st.spinner(f"Extracting text and parsing with {parse_model}..."):
                    qp_text = extract_text_from_pdf(str(qp_tmp))
                    ms_text = extract_text_from_pdf(str(ms_tmp)) if ms_tmp else None
                    questions = parse_with_llm(
                        qp_text, ms_text, paper_id=paper_id, model=parse_model
                    )

                # Save
                out = Path(out_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)

            st.success(f"✅ Parsed {len(questions)} questions → saved to `{out_path}`")

            answered = sum(1 for q in questions if q.get("answer"))
            marked = sum(1 for q in questions if q.get("marks") is not None)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total questions", len(questions))
            m2.metric("With model answers", answered)
            m3.metric("With mark values", marked)

            with st.expander("Preview parsed questions", expanded=False):
                for q in questions[:5]:
                    st.json(q)
                if len(questions) > 5:
                    st.caption(f"… and {len(questions) - 5} more.")

            st.info(
                "Next: go to the **Ingest** tab and point it at "
                f"`{out_path}` to load these into ChromaDB."
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ingest
# ════════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Ingest Parsed JSON into ChromaDB")
    st.markdown(
        "Point to a parsed JSON file (produced by the **Parse Paper** tab or "
        "`parse_exam_data.py`) to embed and upsert it into ChromaDB. "
        "Re-running is safe — duplicates are detected and overwritten."
    )

    st.caption("This is the step that makes questions searchable in the Query tab.")

    ingest_input = st.text_input(
        "Parsed JSON file path",
        value="data/parsed_exam.json",
        key="ingest_input",
    )
    ingest_collection = st.text_input(
        "Collection name",
        value="exam_qa_pairs",
        key="ingest_collection",
    )

    if st.button("Ingest 💾", key="ingest_btn"):
        json_path = Path(ingest_input)
        if not json_path.exists():
            st.error(f"File not found: `{ingest_input}`")
        else:
            log_box = st.empty()
            logs: list[str] = []

            def log(msg: str):
                logs.append(msg)
                log_box.code("\n".join(logs))

            log(f"Loading {ingest_input} ...")
            try:
                with st.spinner("Embedding and upserting..."):
                    ingest(
                        input_json=ingest_input,
                        chroma_dir=chroma_dir,
                        collection_name=ingest_collection,
                    )
                log("✅ Ingestion complete!")
                reset_paper_cache()
                st.success(
                    "Ingestion complete! Switch to the **Query** tab to start asking questions."
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    # ── Database info ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Database contents")
    if st.button("Refresh 🔄", key="refresh_db"):
        reset_paper_cache()

    papers = cached_list_papers()
    if not papers:
        st.info("ChromaDB is empty or not yet initialised.")
    else:
        paper_count, question_count = catalogue_counts(papers)
        st.caption(f"{paper_count} papers | {question_count} questions indexed")
        for pid, qnums in papers.items():
            with st.expander(f"📄 {pid}  ({len(qnums)} questions)"):
                st.write(", ".join(qnums))
