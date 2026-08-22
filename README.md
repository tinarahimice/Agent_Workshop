# llm-agent-workshop

A small, production-minded teaching project that follows **Documents → OCR → Ingestion → Chunking → Embeddings → Reranking → RAG → Function Calling → Agent** using only fictional NovaTech data. Python 3.12, current LlamaIndex workflow agents, a Jina AI embedding/reranking pipeline, an OpenAI-compatible or local Ollama LLM, Streamlit, and async `redis-py` are used—without an arbitrary-code tool.

## Architecture

```text
                 Scanned Documents
                        │
                       OCR
                        │
                        ↓
Text Documents ───→ Ingestion
                        │
                    Chunking
                        │
                   Embeddings
                        │
                  Vector Index
                        │
                Jina AI Reranker
                        │
                        ↓
                      RAG
                        │
                     RAG Tool
                        │
                        ↓
User → Streamlit/CLI → Agent
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
           RAG Tool          Function Tools
```

```text
                 Redis
              /         \
          Cache         Queue
            │             │
           RAG         OCR / Ingest
                          │
                        Worker
```

* **OCR** uses replaceable `OCRService` behavior backed by local Tesseract. It converts supported synthetic scans into plain text in `storage/ocr`; one bad scan is logged and does not stop batch OCR.
* **Ingestion** loads both `data/*.txt` and `storage/ocr/*.txt`. A `SentenceSplitter` makes overlapping chunks; `jina-embeddings-v3` maps each chunk to numbers encoding semantic similarity; LlamaIndex persists a vector index in `storage/index`.
* **Reranking** sends the eight initially retrieved candidates to `jina-reranker-v2-base-multilingual`, which scores query/document relevance and keeps the best three. This makes the distinction between fast vector retrieval and more precise reranking visible in the workshop.
* **RAG** embeds a question with Jina AI, reranks the nearest chunks, and asks the OpenAI-compatible LLM for a grounded answer with source filenames. It loads rather than rebuilds the persisted index.
* **Function calling** lets the LLM select named functions while Python—not the model—does discount and tax arithmetic.
* **Agent** is LlamaIndex's workflow `FunctionAgent`. It can search first and calculate second. Its system prompt is loaded only from `prompts/system_prompt.txt`.
* **UI** is a small Streamlit chat application. Its sidebar switches between standalone RAG and Agent modes and shows the selected models; RAG messages display cache HIT/MISS and source files.
* **Redis cache** is cache-aside: normalized questions are SHA-256 hashed, results expire after 600 seconds, and HIT/MISS/SET events are visible. The document/config fingerprint embedded in every key changes after indexing, making old answers unreachable without a costly key scan.
* **Redis queue** atomically moves jobs from pending to processing and records each state in a job hash. Failed jobs are retried and ultimately retained as failed.
* **Worker** performs slow OCR/ingestion away from the caller and shuts down cleanly on SIGINT/SIGTERM. An OCR job extracts the image and then updates the index.

`CHUNK_SIZE=512` keeps chunks understandable while retaining product records; `CHUNK_OVERLAP=50` carries limited context across boundaries. `RETRIEVAL_TOP_K=8` provides candidates and `RERANK_TOP_N=3` controls the final context. This intentionally offers one clear retrieval pipeline, not a framework of strategies.

## Project tree

```text
data/{products.txt,faq.txt,policies.txt,scanned/}  # PNG scans are generated locally
prompts/system_prompt.txt
scripts/{generate_mock_data.py,generate_scanned_docs.py}
src/{config,llm,logging_config,redis_client,cache,queue,worker,ocr,ingest,rag,tools,agent,main}.py
storage/{ocr,index}/
tests/{test_cache,test_queue,test_tools,test_rag}.py
.env.example  streamlit_app.py  Dockerfile  docker-compose.yml  requirements.txt
```

## Setup and commands

```bash
cp .env.example .env                 # add OpenAI-compatible and Jina AI keys
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main generate-data      # creates the two local PNG scans (not committed)
python -m src.main ocr                # direct/batch OCR (continues on a bad image)
docker compose up -d redis
python -m src.main ingest             # build/persist; `reindex` clears first
python -m src.main rag "What is NovaTech's return policy?"
python -m src.main agent "Find the price of NovaBook Air and apply a 20% discount."
python -m src.main health             # Redis, key presence, index; no paid call
streamlit run streamlit_app.py         # UI at http://localhost:8501
```

Generated PNG files are intentionally ignored by Git because the review system does not accept binary diffs; `generate-data` deterministically recreates `product_catalog.png` and `warranty_policy.png` before the OCR demo.

`ingest` writes a fresh persisted index from current documents. `reindex` first removes the old persisted index. Neither query command silently rebuilds it.

Queue commands:

```bash
docker compose up -d redis worker
python -m src.main enqueue-ocr data/scanned/product_catalog.png
python -m src.main enqueue-ingest --reindex
python -m src.main job-status JOB_ID
# foreground alternative: python -m src.main worker
```

Docker-only equivalents (the `--profile cli` app is intentionally one-shot):

```bash
docker compose build
docker compose up -d redis worker
docker compose --profile cli run --rm app generate-data
docker compose --profile cli run --rm app ingest
docker compose --profile cli run --rm app rag "What is the return policy?"
docker compose up -d redis ui          # UI at http://localhost:8501
```

### Local Ollama fallback (Gemma 3, 4B)

The requested “Gemma 4” fallback is configured as Ollama's **4-billion-parameter Gemma model**, `gemma3:4b`. It replaces only answer generation and agent tool selection; Jina AI still performs embedding and reranking, so `JINA_API_KEY` remains required.

For a host-installed Ollama:

```bash
ollama pull gemma3:4b
# in .env: LLM_PROVIDER=ollama
python -m src.main rag "What is NovaTech's return policy?"
streamlit run streamlit_app.py
```

For a fully containerized Ollama and UI:

```bash
# Set LLM_PROVIDER=ollama in .env first.
docker compose --profile ollama up -d redis ollama
docker compose --profile ollama exec ollama ollama pull gemma3:4b
docker compose --profile ollama up -d ui worker
open http://localhost:8501
```

Switch `LLM_PROVIDER` back to `openai` to use `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL`. No application code changes are needed.

### Environment

| Variable | Purpose / default |
|---|---|
| `OPENAI_API_KEY` | Required for grounded answer generation and agent operations |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` |
| `LLM_PROVIDER` | `openai`; set to `ollama` for the local fallback |
| `OLLAMA_BASE_URL` | `http://localhost:11434`; Compose overrides it to `http://ollama:11434` |
| `OLLAMA_MODEL` | `gemma3:4b` (Gemma 3, 4B parameters) |
| `OLLAMA_REQUEST_TIMEOUT` | `120` seconds, useful for local CPU inference |
| `JINA_API_KEY` | Required for Jina AI embedding and reranking operations |
| `EMBEDDING_MODEL` | `jina-embeddings-v3` |
| `RERANKER_MODEL` | `jina-reranker-v2-base-multilingual` |
| `REDIS_URL` | `redis://localhost:6379/0` locally; Compose overrides host to `redis` |
| `CACHE_ENABLED`, `CACHE_TTL_SECONDS` | `true`, `600` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `512`, `50` |
| `RETRIEVAL_TOP_K`, `RERANK_TOP_N` | Retrieve `8` vector candidates, retain `3` reranked chunks |
| `JOB_MAX_RETRIES` | `3` retries before failed state |
| `LOG_LEVEL` | `INFO`; use `DEBUG` for CLI tracebacks |

No key is logged, baked into the image, or committed: Compose injects keys from `.env` at container runtime. Ollama mode does not require `OPENAI_API_KEY`. OCR paths must exist below `data/scanned`, calculator percentages are bounded, and there is no shell/Python execution tool. If Redis is unavailable, start it and retry; if the index/key is missing, follow the corrective CLI message. Tesseract is installed by the Docker image; local users must install its executable separately. Jina embedding and reranking use Jina AI's hosted API, so the app/worker containers require outbound HTTPS even when answer generation uses local Ollama.

## Workshop demo sequence

1. **Normal RAG** — Run `python -m src.main rag "What is NovaTech's return policy?"`. Explain **Question → Jina embedding → Vector retrieval → Jina reranking → Relevant chunks → LLM → Answer** and find the deliberately specific 21-calendar-day rule.
2. **Redis cache** — Repeat that exact command. The first logs `CACHE MISS`/`CACHE SET`; the second logs `CACHE HIT`. Avoiding duplicate generation reduces latency, work, and API cost.
3. **Function calling** — Run `python -m src.main agent "Calculate a 20% discount on $1200."`. The agent selects deterministic Python; the answer is `$960`.
4. **RAG + function calling** — Ask `python -m src.main agent "Find the price of NovaBook Air and calculate its price after a 20% discount."`: **RAG → $1200 → calculator → $960**. Observable tool names, arguments, results, and final answer are logged—not hidden reasoning.
5. **OCR + queue + RAG + agent** — Start the worker, enqueue `data/scanned/product_catalog.png`, inspect its job status, then ask for NovaMonitor Ultra's price (`$850`) and ask for its price after 15% discount (`$722.50`). This fact is absent from normal text data, proving **scan → queue/worker → OCR → ingestion → RAG tool → agent → function tool → answer**. The second scan adds a unique 90-day bright-pixel warranty condition.

## Tests and limitations

Run `pytest -q` and `python -m compileall -q src scripts tests`; unit tests use fake Redis and no paid model call. End-to-end embedding/reranking requires a reachable Jina AI API, answer generation/agent execution requires a reachable OpenAI-compatible API, and OCR requires Tesseract. This educational queue keeps processing payloads recoverable but intentionally does not implement distributed leases or automatic recovery of a worker killed mid-job; an operator can inspect `workshop:queue:processing`.
