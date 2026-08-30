# Docker

Everything runs with `docker compose`. This document covers the container layout, the commands you
need, and the failure modes worth knowing about.

---

## Commands

### First run

```bash
# 1. Build and start everything
docker compose up -d --build

# 2. Wait for the models. First start downloads ~4 GB, once.
docker compose logs -f model-init          # wait for "models ready", then Ctrl-C

# 3. Create the workspace (idempotent — safe to re-run)
docker compose exec gateway python -m legalaid.db.seed

# 4. Load the corpus. Legislation takes ~6 min, paced politely for a government server.
docker compose exec gateway python scripts/ingest_legislation.py --all
docker compose exec gateway python scripts/ingest_cases.py --limit 60

# 5. Open it
#    UI        http://localhost:3000
#    API docs  http://localhost:8000/docs
```

### Day to day

```bash
docker compose up -d                 # start
docker compose down                  # stop, keep data
docker compose ps                    # status + health
docker compose logs -f gateway       # follow one service
docker compose restart gateway       # restart after a config change
docker compose up -d --build gateway # rebuild after a code change
```

### Verification

```bash
curl http://localhost:8000/api/system/health
docker compose exec gateway python -m pytest tests -q
docker compose exec gateway python scripts/smoke_ui_endpoints.py
docker compose exec gateway python scripts/check_corpus_quality.py
docker compose exec gateway python scripts/ingest_legislation.py --stats
```

### Reset

```bash
docker compose down                  # keeps volumes
docker compose down -v               # DELETES the database, indexes and models — full re-ingest
docker volume ls | grep legalaid
```

---

## Services

| Service | Port | Purpose |
|---|---|---|
| `frontend` | 3000 → 80 | nginx serving the React bundle, reverse-proxying `/api` |
| `gateway` | 8000 | The API the UI talks to; runs the orchestrator |
| `llm` | 8004 | The only component that talks to Ollama for generation |
| `embedding` | 8006 | Document and query embeddings |
| `retrieval` | 8003 | Query embedding → search → filter → rank → context |
| `ollama` | 11434 | Model runtime |
| `model-init` | — | Pulls the two models on first start, then exits |

`gateway`, `llm`, `embedding` and `retrieval` **share one image**. That is deliberate: they share a
codebase and the same Pydantic contracts, so separate images would duplicate ~40 MB of identical
dependencies and let the contracts drift apart. The service is selected by the container command.

### Why `model-init` is separate

Pulling models inside the `ollama` service would re-download 4 GB on every restart. As its own
one-shot container it runs when needed and exits, and `docker compose ps` shows it as `exited (0)`
— which is correct, not a failure.

---

## Transport modes

The compose file defaults to `SERVICE_TRANSPORT=inprocess`: the gateway dispatches directly to the
same handler functions, which keeps the demo path fast.

To force **real HTTP between services** — which is what Exercise 4 asks to demonstrate:

```bash
SERVICE_TRANSPORT=http docker compose up -d
```

Both paths use identical Pydantic contracts and the same handler implementations, so behaviour
cannot diverge and there is no mock service anywhere. The UI displays which transport served each
hop on the Pipeline Trace page.

---

## Volumes

| Volume | Holds | Why it must persist |
|---|---|---|
| `legalaid-data` | SQLite database, uploads, FAISS indexes | Re-ingesting the corpus takes minutes of paced requests to indiacode.nic.in |
| `ollama-models` | Downloaded model weights | ~4 GB re-download otherwise |

`prompts/` and `scripts/` are bind-mounted read-only, so you can edit a prompt and restart without
rebuilding the image.

---

## GPU

**GPU passthrough is commented out in `docker-compose.yml`.** It requires the NVIDIA Container
Toolkit, which is not available by default in Docker Desktop on Windows.

The practical consequence is real: on the host GPU, generation runs at **~40 tok/s** on
`llama3.2:3b` (measured, `eval/latency_breakdown.py`); in a CPU-only container expect a small
fraction of that, so a full multi-agent run may take several times the 60–90 s it takes on the
host.

Three options:

1. **Run Ollama on the host, containers for everything else.** Fastest, and what I'd suggest for a
   live demo. Comment out the `ollama` and `model-init` services and point the backend at the host:
   ```yaml
   OLLAMA_BASE_URL: http://host.docker.internal:11434
   ```
2. **Enable GPU passthrough** on Linux, or Windows with WSL2 + the NVIDIA toolkit: uncomment the
   `deploy.resources.reservations.devices` block under `ollama`.
3. **Use a smaller model.** `LLM_MODEL=qwen2.5:3b-instruct docker compose up -d` is roughly twice
   as fast and still produces valid schema-constrained output.

---

## Troubleshooting

**`failed to connect to the docker API ... dockerDesktopLinuxEngine`**
Docker Desktop is not running. Start it and wait for the whale icon to settle — the CLI being
installed does not mean the daemon is up.

**Health endpoint reports `ollama.reachable: false`**
```bash
docker compose logs ollama
docker compose exec gateway curl -s http://ollama:11434/api/tags
```
Inside the network the host is `ollama`, not `localhost`. If you changed `OLLAMA_BASE_URL`, check
it matches.

**`llm_model_available: false`**
The pull has not finished, or the name does not match:
```bash
docker compose logs model-init
docker compose exec ollama ollama list
```

**Empty knowledge base — 0 provisions**
The corpus is not baked into the image, by design (it is fetched from an official source at setup,
not vendored). Run step 4 above.

**India Code returns 403 during ingestion**
Rate limiting, not a block. The fetcher backs off automatically. If it persists, wait a few minutes
and re-run — ingestion is resumable and skips what is already stored.

**A research request times out in the browser**
CPU-only generation is slow. nginx is configured with a 1800 s read timeout; if you are hitting the
gateway directly through some other proxy, raise its timeout too.

**Frontend loads but the API 502s**
The gateway is still starting, or unhealthy:
```bash
docker compose ps
docker compose logs gateway
```

**Port already in use**
Something is on 3000/8000/11434 — often a local dev server or a host Ollama. Stop it, or remap:
`"3001:80"`.
