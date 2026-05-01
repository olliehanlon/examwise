# Codebase Reference

This project is an AI exam tutor built around a simple parse -> ingest -> query workflow.
It uses Streamlit for the UI, OpenAI for parsing/embeddings/answers, and ChromaDB as a
local persistent vector store.

## Project Shape

```text
.
|-- app.py                  # Streamlit UI and main user workflow
|-- parse_exam_data.py      # PDF text extraction and LLM-based exam parsing
|-- ingest_chroma.py        # Embeds parsed questions and upserts them into ChromaDB
|-- query_chroma.py         # Retrieval and answer generation helpers, plus CLI
|-- requirements.txt        # Python dependencies
|-- data/                   # Source PDFs and parsed exam outputs
|-- chroma_db/              # Local Chroma persistence directory; ignored by git
|-- .streamlit/             # Streamlit config/secrets; secrets ignored by git
|-- .env                    # Local environment secrets; ignored by git
`-- .gitattributes/.gitignore
```

The `ingest/` directory currently exists but appears empty.

## Main Workflow

1. Parse PDFs
   - UI path: `app.py` -> "Parse Paper" tab.
   - CLI path: `python parse_exam_data.py --qp <question-paper.pdf> --ms <mark-scheme.pdf> --out data/parsed_exam.json`.
   - Output shape: JSON object with a `questions` array.
   - Each question is expected to include: `number`, `section`, `question`, `options`, `marks`, `answer`, and later `paper_id`.

2. Ingest parsed JSON
   - UI path: `app.py` -> "Ingest" tab.
   - CLI path: `python ingest_chroma.py --input data/parsed_exam.json`.
   - `ingest_chroma.py` converts each parsed question into Chroma document text, embeds it with `text-embedding-3-small`, and upserts into the `exam_qa_pairs` collection.
   - IDs are deterministic hashes based on `paper_id`, `question_number`, and the start of the document text, so re-running ingestion is intended to be safe.

3. Query the database
   - UI path: `app.py` -> "Query" tab.
   - CLI path: `python query_chroma.py`.
   - Query modes:
     - Strict lookup by `paper_id` and `question_number`.
     - Semantic topic search across all papers or one selected paper.
   - `query_chroma.py` retrieves Chroma chunks, builds a constrained tutor prompt, and calls an OpenAI chat model.

## Important Files

### `app.py`

Streamlit application entry point. It:

- Loads `.env`, then copies selected `st.secrets` into `os.environ` when running on Streamlit Cloud.
- Imports parsing, ingestion, and query helpers from sibling modules.
- Provides three tabs:
  - `Query`
  - `Parse Paper`
  - `Ingest`
- Caches the paper catalogue with `@st.cache_data(ttl=60)`.
- Writes uploaded PDFs to a temporary directory before parsing.

### `parse_exam_data.py`

PDF-to-JSON parser. It:

- Extracts text from PDFs using `PyPDF2.PdfReader`.
- Sends question paper text, and optional mark scheme text, to an OpenAI chat model.
- Requests JSON output using `response_format={"type": "json_object"}`.
- Splits long text into chunks up to `CHUNK_SIZE = 40_000`.
- Deduplicates parsed questions by question number.
- Has defaults pointing to the June 2018 PDFs in `data/`.

### `ingest_chroma.py`

Parsed-JSON-to-Chroma ingestion. It:

- Reads `data/parsed_exam.json` by default.
- Creates or reuses a persistent Chroma collection named `exam_qa_pairs`.
- Builds document text in `Q: ... A: ...` format.
- Stores metadata including `paper_id`, `question_number`, `section`, `marks`, `chunk_type`, and `has_options`.
- Embeds with OpenAI model `text-embedding-3-small`.
- Uses `collection.upsert(...)` so repeated ingestion can update existing records.

### `query_chroma.py`

Retrieval and answer generation. It:

- Connects to the persistent Chroma collection and caches collection objects in `_collection_cache`.
- Lists available papers and question numbers.
- Supports strict retrieval by paper/question metadata.
- Supports semantic retrieval using an embedded query.
- Builds a tutor prompt that instructs the model to use only retrieved context.
- Uses `OPENAI_LLM_MODEL` from the environment, defaulting to `gpt-4o-mini`.

## Environment And Config

Required environment values:

```text
OPENAI_API_KEY
```

Optional environment values:

```text
CHROMA_DB_DIR=./chroma_db
OPENAI_LLM_MODEL=gpt-4o-mini
```

Local secrets live in `.env`.
Streamlit Cloud secrets live in `.streamlit/secrets.toml`.
Both are intentionally ignored by git and should not be committed.

## Dependencies

From `requirements.txt`:

```text
openai>=1.0.0
chromadb>=0.4.0
PyPDF2>=3.0.0
streamlit>=1.32.0
python-dotenv>=1.0.0
```

## Common Commands

Run the Streamlit app:

```powershell
streamlit run app.py
```

Parse the default PDFs:

```powershell
python parse_exam_data.py
```

Parse explicit PDFs:

```powershell
python parse_exam_data.py --qp "data/June 2018 QP.pdf" --ms "data/June 2018 MS.pdf" --out "data/parsed_exam.json"
```

Ingest parsed JSON:

```powershell
python ingest_chroma.py --input "data/parsed_exam.json"
```

Query from the CLI:

```powershell
python query_chroma.py
```

## Data And Git Hygiene

- `chroma_db/` is ignored because it is local, large, and machine-specific.
- `.env` and `.streamlit/secrets.toml` are ignored because they contain secrets.
- `data/parsed_exam.json` is ignored by default, though the comment says it can be committed intentionally if pre-parsed papers should live in the repo.
- PDF source files in `data/` are currently visible to git unless separately ignored.
- Hidden temporary Office lock files are present in `data/` (`~$...pdf`) and should generally be removed or ignored.

## Review Notes

- Several source files contain mojibake in comments/UI strings, suggesting an encoding issue from pasted Unicode or a previous save operation. This affects readability and may affect visible UI labels.
- The app sidebar lets users enter a ChromaDB directory, but `query_chroma.py` defaults to its module-level `CHROMA_DIR` for listing/retrieval. Query mode may not fully respect the sidebar value unless query helpers are extended to accept it.
- `parse_exam_data.py` chunks the question paper and mark scheme independently by character count, then zips chunks together. For long papers, this may pair unrelated question-paper and mark-scheme chunks.
- `query_chroma.py` raises at import time if `OPENAI_API_KEY` is missing. This is fine for production but can make local inspection and tests harder.
- There are no automated tests yet. Good first tests would cover parsed JSON shape, deterministic Chroma IDs, document text construction, and prompt construction.

## Mental Model For Future Changes

Treat the project as four layers:

1. UI orchestration in `app.py`.
2. Parsing and schema creation in `parse_exam_data.py`.
3. Embedding and persistence in `ingest_chroma.py`.
4. Retrieval and answer generation in `query_chroma.py`.

When adding features, keep those boundaries intact unless there is a clear reason to change them.
