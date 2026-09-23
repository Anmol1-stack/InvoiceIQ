# Repository/codebase understanding experiment

## Question

Can the current InvoiceIQ LLM + RAG system answer questions requiring relationships across multiple source files?

## What the current RAG index contains

The ingestion scripts read only `knowledge_base/*.txt`, create 25 word-window chunks, and store their embeddings in `data/embeddings.json`. Python files, Docker Compose, and the Streamlit UI are not indexed. Therefore, a question such as “Which files process a user question?” has no code evidence available to the RAG retriever.

## Manual codebase tracing: a representative multi-file question

**Question:** What happens after a user submits a question in InvoiceIQ?

| Step | File/component | Evidence |
| --- | --- | --- |
| 1 | `frontend/app.py` | The Streamlit UI sends `question` (and optionally `model`) to `http://localhost:8000/ask`. |
| 2 | `scripts/app_service.py` | The application service forwards the same JSON request to `http://orchestrator:8003/ask`. |
| 3 | `scripts/orchestrator.py` | The orchestrator calls retrieval at `http://retrieval:8001/retrieve`, builds context from returned text, then calls the LLM service. |
| 4 | `scripts/retrieval_service.py` | It embeds the question with `nomic-embed-text`, computes cosine similarity against `data/embeddings.json`, and returns up to four results at or above 0.60. |
| 5 | `scripts/llm_service.py` | It combines question and context in the fixed prompt and calls the local Ollama generate API. |
| 6 | `scripts/orchestrator.py` → `scripts/app_service.py` → UI | The answer, model, and retrieved documents are returned to the user. |

`docker-compose.yml` supplies the service names and ports that connect these files. This is a genuine repository-level answer because it needs relationships among the frontend, three backend services, stored embeddings, Ollama, and Docker networking.

## Current-system result

The current policy RAG system cannot reliably produce the answer above because no source-code chunks are in its vector store. It would either retrieve unrelated policy documents or return no context. This is expected and is an important Exercise 6 finding, not a failure of the policy assistant’s main use case.

## Additional repository questions that the current index cannot answer

| Repository question | Files that must be related |
| --- | --- |
| Which component determines whether a document is relevant? | `scripts/retrieval_service.py`, `data/embeddings.json` |
| Where is the answer prompt defined and where is it used? | `scripts/llm_service.py`, `scripts/orchestrator.py` |
| What would be affected by changing the retrieval threshold? | `scripts/retrieval_service.py`, `scripts/orchestrator.py`, `frontend/app.py`, evaluation result files |
| Which files allow a user to select a different model? | `frontend/app.py`, `scripts/orchestrator.py`, `scripts/llm_service.py`, `docker-compose.yml` |

## Next-step design for Week 5 / Sourcegraph-style work

Keep the policy index for InvoiceIQ answers, and build a **separate code index** for repository questions. Chunk Python/YAML by function or class, attach metadata (`path`, symbol, imports, callers, line range), and retrieve code chunks before asking the LLM. A static call/dependency graph would make “what is affected?” and “which function calls?” queries more reliable than plain text similarity alone.
