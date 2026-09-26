# Hybrid RAG

Grounded question answering over your own documents. Every answer cites the
page it came from, and the app refuses rather than guesses when the documents
don't contain the answer.

![CI](https://github.com/palanijrcs/hybrid-rag-app/actions/workflows/ci.yml/badge.svg)

## How it works

```
Question
  -> Input guardrail        (prompt injection, secrets, off-topic)
  -> Hybrid retrieval       FAISS (semantic) + BM25 (keyword) + Neo4j (knowledge graph)
  -> RRF fusion -> cross-encoder re-ranker (drops weak matches)
  -> Context builder        numbered <source> blocks
  -> Grounded LLM           JSON answer with [n] citations and a coverage check
  -> Output guardrail       number/period checks, citation fixes, LLM verification
Answer + sources
```

| Layer | Tech |
|---|---|
| API | FastAPI |
| UI | Streamlit |
| Vector search | FAISS + sentence-transformers |
| Keyword search | BM25 (rank-bm25) |
| Knowledge graph | Neo4j AuraDB, LLM-extracted entities and relations |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI `gpt-4o-mini` |
| Evaluation | Golden dataset + optional DeepEval metrics |
| Deployment | Docker Compose |

## Quick start (Docker)

```bash
cp .env.example .env        # then fill in OPENAI_API_KEY and NEO4J_* values
docker compose up -d --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

Upload a PDF, DOCX, TXT or MD file in the sidebar, wait for "graph ready", then ask a question.

## Local development

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt -r backend\requirements-dev.txt -r frontend\requirements.txt
cd backend;  uvicorn app.main:app --reload          # terminal 1
cd frontend; streamlit run app.py                   # terminal 2
```

## Tests

```powershell
.\scripts\run_tests.ps1 -Quick   # fast, no model downloads
.\scripts\run_tests.ps1          # all offline tests + coverage
.\scripts\run_tests.ps1 -Live    # also a real OpenAI/Neo4j smoke test (a few cents)
```

## Evaluation

```powershell
cd backend
python ..\scripts\evaluate.py --no-deepeval --label baseline   # behaviour, sources, key facts
python ..\scripts\evaluate.py --compare                        # compare saved runs
```

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents/upload` | Upload and index a document |
| GET | `/documents` | List documents |
| POST | `/query` | Ask a grounded question |
| GET | `/health` | Status of retrievers, Neo4j and the LLM |
| GET | `/evaluation` | Saved evaluation runs |

## Configuration

All settings are environment variables; see `.env.example`. Useful ones:
`CHUNK_SIZE`, `RERANK_TOP_K`, `MIN_RELEVANCE_SCORE`, `ENABLE_KG_RETRIEVAL`,
`ENABLE_OUTPUT_GUARDRAIL`, `ENABLE_LLM_VERIFICATION`.

## Security

- Secrets live only in `.env`, which is git-ignored and never copied into Docker images.
- Question text and keys are never logged.
- Uploaded documents and indexes (`data/`) are git-ignored.
