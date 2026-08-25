# llm-agent-workshop

A small, production-minded teaching project that follows **Documents → OCR → Ingestion → Chunking → Embeddings → Reranking → RAG → Function Calling → Agent** using only fictional BitTeck data. Python 3.12, current LlamaIndex workflow agents, a Jina AI embedding/reranking pipeline, an OpenAI-compatible or local Ollama LLM, Streamlit, and async `redis-py` are used—without an arbitrary-code tool.

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
                  ┌─────┴─────┐
                  ↓           ↓
            Dense vectors  BM25 sparse
                  └─────┬─────┘
                        ↓
                Qdrant hybrid search
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
* **Ingestion** loads both `data/*.txt` and `storage/ocr/*.txt`. A `SentenceSplitter` makes overlapping chunks; `jina-embeddings-v3` creates dense semantic vectors while `Qdrant/bm25` creates sparse lexical vectors. Both are stored in the `bitteck_knowledge` Qdrant collection.
* **Hybrid retrieval** runs dense and BM25 searches together in Qdrant and fuses their candidates. `HYBRID_ALPHA=0.5` gives the semantic and lexical paths equal weight; `1.0` favors only dense similarity and `0.0` only sparse similarity.
* **Reranking** sends the eight hybrid candidates to `jina-reranker-v2-base-multilingual`, which scores query/document relevance and keeps the best three. This makes the distinction between fast vector retrieval and more precise reranking visible in the workshop.
* **RAG** embeds a question with Jina AI, reranks the nearest chunks, and asks the OpenAI-compatible LLM for a grounded answer with source filenames. It loads rather than rebuilds the persisted index.
* **Function calling** lets the LLM select named functions while Python—not the model—does discount and tax arithmetic.
* **Agent** is LlamaIndex's workflow `FunctionAgent`. It can search first and calculate second. Its system prompt is loaded only from `prompts/system_prompt.txt`.
* **UI** is a small Streamlit chat application. Its sidebar switches between standalone RAG and Agent modes and shows the selected models; RAG messages display cache HIT/MISS and source files.
* **Redis cache** is cache-aside: normalized questions are SHA-256 hashed, results expire after 600 seconds, and HIT/MISS/SET events are visible. The document/config fingerprint embedded in every key changes after indexing, making old answers unreachable without a costly key scan.
* **Redis queue** atomically moves jobs from pending to processing and records each state in a job hash. Failed jobs are retried and ultimately retained as failed.
* **Worker** performs slow OCR/ingestion away from the caller and shuts down cleanly on SIGINT/SIGTERM. An OCR job extracts the image and then updates the index.

`CHUNK_SIZE=512` keeps chunks understandable while retaining product records; `CHUNK_OVERLAP=50` carries limited context across boundaries. `RETRIEVAL_TOP_K=8` provides candidates and `RERANK_TOP_N=3` controls the final context.

### BM25, in this workshop's pipeline

BM25 is a **lexical ranking** algorithm: instead of asking whether two passages have a similar meaning, it rewards passages containing the query's actual terms. It improves basic term counting in three important ways:

1. **Term frequency (TF):** a query term appearing in a chunk is useful, but repeatedly adding the same word gives diminishing returns.
2. **Inverse document frequency (IDF):** a rare term such as a product code is more informative than a common term such as “the”.
3. **Length normalization:** a long chunk does not win merely because it has more opportunities to contain a term.

A common form is `score(D,Q) = Σ IDF(q) × TF(q,D) × (k₁ + 1) / (TF(q,D) + k₁ × (1 - b + b × |D| / avgdl))`. Here `D` is a chunk, `Q` is the question, `|D|` is its length, and `avgdl` is the average chunk length. `k₁` controls term-frequency saturation and `b` controls length normalization.

The trade-off is easy to demonstrate: dense retrieval can connect “money back” with “return policy”, while BM25 is particularly strong for an exact token such as `NovaBook`, `$1200`, or a policy identifier. The hybrid retriever combines both candidate lists, then the Jina reranker makes the final relevance decision. The teaching flow is therefore **Question → dense embedding + BM25 sparse query → Qdrant fusion → Jina reranking → context → LLM answer**.

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
docker compose up -d redis qdrant
python -m src.main ingest             # replace the Qdrant collection snapshot
python -m src.main rag "What is BitTeck's return policy?"
python -m src.main agent "Find the price of NovaBook Air and apply a 20% discount."
python -m src.main health             # Redis, key presence, index; no paid call
streamlit run streamlit_app.py         # UI at http://localhost:8501
```

Generated PNG files are intentionally ignored by Git because the review system does not accept binary diffs; `generate-data` deterministically recreates `product_catalog.png` and `warranty_policy.png` before the OCR demo.

Tesseract is an operating-system executable, not only a Python package. For
the quickest setup that does **not** install anything on the host, run:

```bash
docker compose --profile cli run --build --rm app generate-data
docker compose --profile cli run --build --rm app ocr
```

The `--build` flag matters if the local image predates the Dockerfile's
Tesseract installation. To run OCR directly with `python -m src.main ocr`,
install the native engine first (`sudo apt-get install tesseract-ocr` on
Debian/Ubuntu, `brew install tesseract` on macOS, or `choco install tesseract`
on Windows). Installing `pytesseract` alone is not sufficient. If the executable
is installed outside `PATH`, set
`TESSERACT_CMD=/full/path/to/tesseract`. The project Docker image already
installs it. When the executable is absent, batch OCR fails once with these
instructions instead of logging one identical failure per image and exiting
successfully.

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

The `ingest` step is required before the first `rag` or `agent` query and before
using the UI. Wait for it to print `Index version: ...`; if ingestion fails,
fix the reported configuration or connectivity problem and rerun it. Starting
Qdrant or the UI alone does not create an index. The generated version marker
is persisted in the bind-mounted `storage/index` directory, while the vectors
are persisted by Qdrant's named volume.

If ingestion times out, use its stage-specific error to isolate the dependency.
Local Qdrant requests bypass `HTTP_PROXY`/`HTTPS_PROXY` automatically, so a
developer proxy cannot intercept `localhost`, `127.0.0.1`, or `::1` traffic.
A Qdrant error means the service or port `6333` is unreachable; check it with
`docker compose up -d qdrant` and `curl http://localhost:6333/healthz`. An index
creation/Jina error means the host needs outbound HTTPS access to `api.jina.ai`
and a valid `JINA_API_KEY`; on the first run, FastEmbed also downloads the BM25
model. Set `LOG_LEVEL=DEBUG` to include the underlying traceback. Qdrant HTTP
requests use `QDRANT_TIMEOUT` seconds (default `15`).

### اجرای کامل با Docker و Ollama

The default local model is Ollama's lightweight `gemma3:270m`. Ollama replaces
only answer generation and agent tool selection; Jina AI still performs
embedding and reranking, so a valid `JINA_API_KEY` and outbound HTTPS access
remain required. `OPENAI_API_KEY` may stay empty in this mode.

For a host-installed Ollama:

```bash
ollama pull gemma3:270m
# in .env: LLM_PROVIDER=ollama
python -m src.main rag "What is BitTeck's return policy?"
streamlit run streamlit_app.py
```

For a fully containerized first run, execute these commands from the repository
root in this order:

```bash
# 1. Create the runtime configuration. Do this only once.
cp .env.example .env

# 2. Edit .env and set these values (JINA_API_KEY must be a real key):
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=gemma3:270m
# JINA_API_KEY=your-jina-api-key

# 3. Build the Python application image and start its dependencies.
docker compose build
docker compose --profile ollama up -d redis qdrant ollama

# 4. Download the model into Ollama's persistent named volume.
docker compose --profile ollama exec ollama ollama pull gemma3:270m

# 5. Confirm that Ollama can see the downloaded model.
docker compose --profile ollama exec ollama ollama list

# 6. Generate sample files, run OCR, and build the required search index.
docker compose --profile cli --profile ollama run --rm app generate-data
docker compose --profile cli --profile ollama run --rm app ocr
docker compose --profile cli --profile ollama run --rm app ingest

# 7. Check Redis, Qdrant, provider configuration, and index presence.
docker compose --profile cli --profile ollama run --rm app health

# 8. Run either CLI query (the quotes keep each question one argument).
docker compose --profile cli --profile ollama run --rm app rag "What is BitTeck's return policy?"
docker compose --profile cli --profile ollama run --rm app agent "Find the price of NovaBook Air and apply a 20% discount."

# 9. Start the browser UI and background worker.
docker compose --profile ollama up -d ui worker
# Open http://localhost:8501 in a browser.
```

Use `docker compose --profile ollama ps` to inspect container state and
`docker compose --profile ollama logs -f ui ollama` to follow UI/Ollama logs
(`Ctrl+C` stops following logs without stopping containers). After source or
dependency changes, rebuild and recreate the application services with
`docker compose build && docker compose --profile ollama up -d --force-recreate ui worker`.
Normal shutdown keeps Redis, Qdrant, and Ollama data in named volumes:

```bash
docker compose --profile ollama down
```

To perform a destructive reset, including the downloaded Ollama model, vector
index, Redis data, and Qdrant data, add `--volumes` and then repeat the pull and
ingestion steps:

```bash
docker compose --profile ollama down --volumes
```

Inside Compose, the application must reach Ollama at
`http://ollama:11434`, not `localhost`; the Compose file overrides
`OLLAMA_BASE_URL` for `app`, `ui`, and `worker`. The host-facing
`http://localhost:11434` address is only for commands run directly on the host.
If a query reports a missing index, rerun step 6's `ingest` command. If it
reports a Jina error, verify `JINA_API_KEY` and outbound HTTPS connectivity.

Switch `LLM_PROVIDER` back to `openai` to use `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL`. No application code changes are needed.

### GapGPT (OpenAI-compatible)

GapGPT uses the same OpenAI-compatible client path; it is not a third LLM
implementation. Configure its API key in `OPENAI_API_KEY` and its endpoint in
`OPENAI_BASE_URL`:

```dotenv
OPENAI_API_KEY=your-gapgpt-key
OPENAI_BASE_URL=https://api.gapgpt.app/v1
LLM_MODEL=gpt-4
LLM_PROVIDER=gapgpt
```

The settings loader normalizes `gapgpt` to the internal `openai` backend. An
empty `OPENAI_API_KEY` is sufficient to start the UI, but an LLM request will
fail with an actionable missing-key error. `JINA_API_KEY` is independently
required for ingestion, hybrid retrieval, and reranking.

### Environment

| Variable | Purpose / default |
|---|---|
| `OPENAI_API_KEY` | Required for grounded answer generation and agent operations |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` |
| `LLM_PROVIDER` | `openai`; `gapgpt`, `openrouter`, and `openai-compatible` are accepted aliases, or use `ollama` for the local fallback |
| `OLLAMA_BASE_URL` | `http://localhost:11434`; Compose overrides it to `http://ollama:11434` |
| `OLLAMA_MODEL` | `gemma3:270m` (Gemma 3, 4B parameters) |
| `OLLAMA_REQUEST_TIMEOUT` | `120` seconds, useful for local CPU inference |
| `JINA_API_KEY` | Required for Jina AI embedding and reranking operations |
| `EMBEDDING_MODEL` | `jina-embeddings-v3` |
| `RERANKER_MODEL` | `jina-reranker-v2-base-multilingual` |
| `REDIS_URL` | `redis://localhost:6379/0` locally; Compose overrides host to `redis` |
| `QDRANT_URL`, `QDRANT_COLLECTION` | `http://localhost:6333`, `bitteck_knowledge`; Compose overrides the host to `qdrant` |
| `QDRANT_TIMEOUT` | `15` seconds for Qdrant HTTP operations |
| `SPARSE_MODEL`, `HYBRID_ALPHA` | `Qdrant/bm25`, `0.5`; lexical model and dense/sparse fusion balance |
| `CACHE_ENABLED`, `CACHE_TTL_SECONDS` | `true`, `600` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `512`, `50` |
| `RETRIEVAL_TOP_K`, `RERANK_TOP_N` | Retrieve `8` vector candidates, retain `3` reranked chunks |
| `JOB_MAX_RETRIES` | `3` retries before failed state |
| `LOG_LEVEL` | `INFO`; use `DEBUG` for CLI tracebacks |
| `TESSERACT_CMD` | `tesseract`; executable name or full path used by local OCR |

No key is logged, baked into the image, or committed: Compose injects keys from `.env` at container runtime. Ollama mode does not require `OPENAI_API_KEY`. OCR paths must exist below `data/scanned`, calculator percentages are bounded, and there is no shell/Python execution tool. If Redis is unavailable, start it and retry; if the index/key is missing, follow the corrective CLI message. Tesseract is installed by the Docker image; local users must install its executable separately. Jina embedding and reranking use Jina AI's hosted API, so the app/worker containers require outbound HTTPS even when answer generation uses local Ollama.

## Workshop demo sequence

1. **Hybrid RAG** — Run `python -m src.main rag "What is BitTeck's return policy?"`. Explain **Question → Jina dense embedding + BM25 terms → Qdrant hybrid retrieval → Jina reranking → Relevant chunks → LLM → Answer** and find the deliberately specific 21-calendar-day rule.
2. **Redis cache** — Repeat that exact command. The first logs `CACHE MISS`/`CACHE SET`; the second logs `CACHE HIT`. Avoiding duplicate generation reduces latency, work, and API cost.
3. **Function calling** — Run `python -m src.main agent "Calculate a 20% discount on $1200."`. The agent selects deterministic Python; the answer is `$960`.
4. **RAG + function calling** — Ask `python -m src.main agent "Find the price of NovaBook Air and calculate its price after a 20% discount."`: **RAG → $1200 → calculator → $960**. Observable tool names, arguments, results, and final answer are logged—not hidden reasoning.
5. **OCR + queue + RAG + agent** — Start the worker, enqueue `data/scanned/product_catalog.png`, inspect its job status, then ask for NovaMonitor Ultra's price (`$850`) and ask for its price after 15% discount (`$722.50`). This fact is absent from normal text data, proving **scan → queue/worker → OCR → ingestion → RAG tool → agent → function tool → answer**. The second scan adds a unique 90-day bright-pixel warranty condition.

## Tests and limitations

Run `pytest -q` and `python -m compileall -q src scripts`; unit tests use fake Redis and no paid model call. The first BM25 use downloads its FastEmbed model. End-to-end ingestion/search requires Qdrant and a reachable Jina AI API, answer generation/agent execution requires a reachable OpenAI-compatible API, and OCR requires Tesseract. This educational queue keeps processing payloads recoverable but intentionally does not implement distributed leases or automatic recovery of a worker killed mid-job; an operator can inspect `workshop:queue:processing`.
