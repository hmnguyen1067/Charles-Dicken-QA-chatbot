# Docker Infrastructure Guide

This document describes the Docker services used by the Charles Dickens QA chatbot system and how to run them via Docker Compose.

## Services Overview

The system uses Docker Compose to orchestrate these services:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **Qdrant** | `qdrant/qdrant:latest` | 6333 (API), 6334 (gRPC) | Vector database for embeddings and hybrid search |
| **Redis** | `redis:latest` | 6380 (mapped from 6379) | Document store, index store, and ingestion cache |
| **Opik** | Various (multi-container) | 5173 | Local evaluation tracking, metrics, and dataset management |
| **API (FastAPI)** | built from `infra/Dockerfile.api` | 8001 | Backend REST API for the chatbot |
| **Streamlit** | built from `infra/Dockerfile.streamlit` | 8501 | Frontend UI; behind profile `app` |

## Quick Start

### Start All Services (includes Streamlit UI)
```bash
make docker-up
```
Equivalent to:
```bash
bash scripts/pull_opik.sh
docker compose -f infra/docker-compose.yaml --profile opik --profile app up -d --build
```

This starts Qdrant, Redis, Opik, the API service, and Streamlit. It also runs an init step that uploads a Qdrant snapshot before the API starts.

### Stop All Services
```bash
make docker-down
```
Equivalent to:
```bash
docker compose -f infra/docker-compose.yaml --profile opik --profile app down
```

### Check Service Status
```bash
docker ps
```

### Quick Health Checks
- API: `curl http://localhost:8001/health`
- API docs: open `http://localhost:8001/docs`
- Streamlit: open `http://localhost:8501`

## Service Details

### Qdrant (Vector Store)
- **Container Name**: `qdrant`
- **Ports**:
  - 6333: HTTP API
  - 6334: gRPC API
  - 6335: Internal metrics
- **Volume**: `qdrant_data` (persisted storage)
- **Config**: Custom `production.yaml` with log level INFO
- **Restart Policy**: Always

**Accessing Qdrant**:
- Dashboard: http://localhost:6333/dashboard
- API endpoint: http://localhost:6333

Snapshot preload:
- On startup, a one-shot init container uploads `qdrant_snapshots/charles_dickens-2025.snapshot` into the `charles_dickens` collection using Qdrant's snapshot upload API.
- Logs: `docker logs api-prestart-curl`

### Redis (Cache & Document Store)
- **Container Name**: `redis-general`
- **Port**: 6380 (external storage), 6379 (Opik internal)
- **Volume**: `redis_data` (persisted storage)
- **Restart Policy**: Always

**Accessing Redis**:
```bash
redis-cli -p 6380
```

### API (FastAPI)
- **Container Name**: `api`
- **Port**: 8001
- **Healthcheck**: `GET /health`
- **Env**: `.env` is passed via `env_file` (e.g., `OPENAI_API_KEY`)

**Accessing API**:
- Base: http://localhost:8001
- Docs: http://localhost:8001/docs

### Streamlit (Frontend)
- **Container Name**: `streamlit`
- **Profile**: `app` (Makefile includes this by default)
- **Port**: 8501

**Accessing Streamlit**:
- UI: http://localhost:8501

### Opik (Evaluation Platform)
- **Profile**: `opik` (included from opik submodule)
- **Source**: `opik/deployment/docker-compose/docker-compose.yaml`
- **Ports**:
  - 5173: Web UI
- **Components**: Frontend, backend, Python backend, guardrails, ClickHouse, MySQL

**Accessing Opik**:
- Dashboard: http://localhost:5173
- API: http://localhost:5173/api

## Data Persistence

All services use Docker volumes for data persistence:

```yaml
volumes:
  qdrant_data:    # Vector embeddings and indices
  redis_data:     # Cached documents and metadata
  # Opik volumes defined in submodule
```

Data persists across container restarts unless volumes are explicitly removed.

## Troubleshooting

### Services Not Starting
```bash
# Check logs for specific service
docker logs qdrant
docker logs redis-general

# Check all services
docker compose -f infra/docker-compose.yaml --profile opik --profile app logs
```

### Port Conflicts
If ports 6333, 6380, or 5173 are already in use, modify the port mappings in:
- `infra/docker-compose.yaml` (Qdrant, Redis)
- Update corresponding values in `constants.py`
If ports 8001 or 8501 are in use, adjust the API/Streamlit port mappings in `infra/docker-compose.yaml`.

### Reset Data
```bash
# Stop services
make docker-down

# Remove volumes
docker volume rm charles-dicken-qa-chatbot_qdrant_data
docker volume rm charles-dicken-qa-chatbot_redis_data

# Restart
make docker-up
```

### Opik Setup
Opik is included as a Git submodule. If missing:
```bash
make pull-opik
# or manually:
git clone https://github.com/comet-ml/opik.git opik
```

## Architecture Notes

- **Compose File Structure**: Main `docker-compose.yaml` includes Redis and Qdrant configs, adds API and optional Streamlit services, then includes Opik from submodule
- **Profiles**: Opik services run under the `opik` profile (all services started by default with `--profile opik`)
- **Profiles (app)**: Streamlit runs under the `app` profile; include `--profile app` to start it
- **Networking**: Services communicate via Docker's default bridge network
- **Health Checks**: Applications (app.py, workflow.py) connect to services at configured ports

## Integration with RAGFlow

The RAGFlow workflow (`workflow.py`) connects to these services via configuration:

```python
RAGFlow(
    collection_name="charles_dickens",  # Qdrant collection
    qdrant_host="localhost",
    qdrant_port=6333,
    redis_host="localhost",
    redis_port=6380,
    opik_host="http://localhost:5173/api"
)
```

Ensure services are running before:
- Running ingestion (`source_path` parameter)
- Loading context (`initialize_ctx=True`)
- Querying the system (`query` parameter)
- Starting the Streamlit app (Compose includes `--profile app` by default; for local dev use `streamlit run src/chat/app.py --server.port 8501`)
