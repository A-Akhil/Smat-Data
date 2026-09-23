# 10-Q RAG Q&A

A Retrieval-Augmented Generation system that answers questions about the text,
tables, and figures in a single SEC Form 10-Q filing (Apple Inc., quarter
ended June 25, 2022), built as a single Django project (UI + backend
together).

The accompanying design write-up (submitted separately as a PDF, per the
assessment's requirements) covers the full approach, design decisions,
assumptions, and known limitations.

## Architecture

```
PDF ─▶ layout-aware parsing (ingestion/parser.py)
     ─▶ chunking: one chunk per page of narrative text,
                  one chunk per table, kept whole (ingestion/chunker.py)
     ─▶ embeddings via Hugging Face Inference API (ingestion/embeddings.py)
     ─▶ ChromaDB Cloud vector store (ingestion/vectorstore.py)

Question ─▶ hybrid retrieval: dense (ChromaDB) + BM25 keyword (rag/retrieval.py)
          ─▶ context assembly, tables passed as Markdown (rag/generation.py)
          ─▶ Groq LLM chat completion, with API-key rotation (rag/llm_client.py)
          ─▶ answer + cited source chunks rendered in the web UI (rag/views.py)
```

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys (see below)
```

Required in `.env`:

- `GROQ_API_KEY_1` (and optionally `_2`, `_3`, ... for rotation) — [console.groq.com](https://console.groq.com)
- `HUGGINGFACE_API_KEY` — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- `CHROMADB_API_KEY`, `CHROMADB_TENANT`, `CHROMADB_DATABASE` — [trychroma.com](https://www.trychroma.com/) Cloud

The source PDF isn't committed to the repo (see `.gitignore`). Fetch it with:

```bash
curl -sL -o 2022_Q3_AAPL.pdf \
  "https://github.com/docugami/KG-RAG-datasets/raw/main/sec-10-q/data/v1/docs/2022%20Q3%20AAPL.pdf"
```

## Run

```bash
python manage.py migrate
python manage.py seed_api_keys   # loads GROQ_API_KEY_* from .env into the DB
python manage.py ingest_pdf      # parses, chunks, embeds, and indexes the PDF
python manage.py runserver
```

Then open http://localhost:8000 and ask a question.

## Project layout

- `ingestion/` — PDF parsing, chunking, embedding, and vector-store indexing.
  Run via `python manage.py ingest_pdf [path/to.pdf]`.
- `rag/` — retrieval, LLM client (with key rotation), answer generation, and
  the Q&A web UI.
- `core/` — Django project settings and URL routing.

## Docker

```bash
docker build -t tenq-rag .
docker run --env-file .env -p 8000:8000 tenq-rag
```
