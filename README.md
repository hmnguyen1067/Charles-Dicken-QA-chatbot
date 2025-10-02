# Charles Dickens QA RAG-powered Chat Interface

[![Demo video](https://img.youtube.com/vi/5b3XEsJqm00/hqdefault.jpg)](https://www.youtube.com/embed/5b3XEsJqm00)

## Description
This project is a Retrieval-Augmented Generation (RAG) chatbot focused on Charles Dickens’ works. It combines LlamaIndex Workflows for orchestration, a FastAPI backend, and a Streamlit frontend. Context and embeddings persist in Qdrant and Redis for fast, repeatable queries. Opik provides evaluation and monitoring so you can iterate on retrievers, prompts, and models with data.

Technologies: Python, LlamaIndex, Qdrant, Opik, FastAPI, Streamlit, Docker, OpenAI.

Project structure overview and key links:

```
.
├─ data/
├─ docs/
├─ infra/
├─ notebooks/
├─ opik/
├─ qdrant_snapshots/
├─ scripts/
├─ src/
│  ├─ api/
│  ├─ chat/
│  └─ charles_dicken_qa_chatbot/
├─ Makefile
├─ requirements.txt
├─ pyproject.toml
├─ pixi.lock
├─ README.md
├─ .env
```

- `src/charles_dicken_qa_chatbot/`: Core library (ingestion, retrieval, workflow). See `src/charles_dicken_qa_chatbot/workflow.py`.
- `src/api/`: FastAPI backend and schemas. See `src/api/api.py`.
- `src/chat/`: Streamlit chat UI. See `src/chat/app.py`.
- `infra/`: Docker Compose and Dockerfiles. See `infra/docker-compose.yaml`, `infra/Dockerfile.api`, `infra/Dockerfile.streamlit`.
- `docs/`: Operations guides. See `docs/DOCKER.md`, `docs/STREAMLIT_GUIDE.md`.
- `opik/`: Opik deployment (after pulling `make pull-opik`). Compose includes this stack.
- `qdrant_snapshots/`: Prebuilt Qdrant snapshot used to warm the vector store on startup.
- `notebooks/`: Experimental notebooks (`ragflow.ipynb` shows how to run retrieval and response evaluation).
- `scripts/`: Utility scripts (e.g., Opik fetch).

## Architecture Overview
For a visual walkthrough of the system components and data flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Features
- Orchestration with LlamaIndex Workflow and flexible indexing
- LLM evaluation and monitoring with Opik
- FastAPI backend and Streamlit chat UI
- High-performance vector search with Qdrant + Redis document/index stores
- Hybrid retrieval (embedding + BM25) with reranking and automatic best-retriever selection
- Docker Compose for a smooth local stack and snapshot preloading

## Installation

Prerequisites:
- Python 3.11+
- Docker Desktop with Docker Compose
 - Make
 - Pixi (for environment management; used by some Makefile targets)
- An OpenAI API key in `.env` at the project root:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```

Clone and enter the repo:
```bash
git clone <this-repo>
cd Charles-Dicken-QA-chatbot
```

### Docker
1) Start all services (Qdrant, Redis, Opik, FastAPI, Streamlit):
```bash
make docker-up
# equivalent
docker compose -f infra/docker-compose.yaml --profile opik --profile app up -d --build
```


Notes:
- See additional operational details in [docs/DOCKER.md](docs/DOCKER.md) and the UI guide in [docs/STREAMLIT_GUIDE.md](docs/STREAMLIT_GUIDE.md).

## Usage
- API health: `curl http://localhost:8001/health`
- API docs: open `http://localhost:8001/docs`
- Streamlit UI: `http://localhost:8501`
- Opik dashboard: `http://localhost:5173`
- Qdrant dashboard: `http://localhost:6333/dashboard#/collections`

Using the Streamlit UI:
1) Ensure the backend is healthy on port 8001 and the system is initialized (auto on startup; otherwise click “Initialize System” in the sidebar).
2) Open `http://localhost:8501` and ask a question (e.g., “What are the main themes in Great Expectations?”). The UI shows answers and source documents.

API examples:
```bash
# Health
curl http://localhost:8001/health

# Config
curl http://localhost:8001/config

# Initialize (if needed)
curl -X POST http://localhost:8001/initialize

# Query
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Who is Pip in Great Expectations?"}'
```

Python example:
```python
import requests

BASE = "http://localhost:8001"

# Optional: initialize if not already
print(requests.post(f"{BASE}/initialize").json())

resp = requests.post(f"{BASE}/query", json={"question": "What is the plot of A Tale of Two Cities?"})
data = resp.json()
print("Answer:\n", data["answer"])  # text
for i, s in enumerate(data.get("sources", []), 1):
    print(f"\nSource {i} ({s['metadata'].get('title', 'Unknown')}) — score={s.get('score')}")
    print(s["text"])  # snippet
```

Where things live:
- Backend endpoints and logic: [src/api/api.py](src/api/api.py)
- Streamlit UI app: [src/chat/app.py](src/chat/app.py)
- LlamaIndex Workflow and RAG logic: [src/charles_dicken_qa_chatbot/workflow.py](src/charles_dicken_qa_chatbot/workflow.py)
- Docker + Compose: [infra/docker-compose.yaml](infra/docker-compose.yaml), [infra/Dockerfile.api](infra/Dockerfile.api), [infra/Dockerfile.streamlit](infra/Dockerfile.streamlit)
- Additional guides: [docs/DOCKER.md](docs/DOCKER.md), [docs/STREAMLIT_GUIDE.md](docs/STREAMLIT_GUIDE.md)

Tips:
- First-time setup is fastest with Docker Compose, which uploads the provided Qdrant snapshot before the API starts.
- If you prefer to ingest documents yourself, see the notebook [notebooks/ingestion_no_workflow.ipynb](notebooks/ingestion_no_workflow.ipynb), the ingestion utilities in [src/charles_dicken_qa_chatbot/ingestion.py](src/charles_dicken_qa_chatbot/ingestion.py) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
- There is also query rewriting with HyDE (Hypothetical Document Embeddings) in [notebooks/generation.ipynb](notebooks/generation.ipynb) but due to the overhead, it's not worth using it in workflow as seen in [src/charles_dicken_qa_chatbot/workflow.py](src/charles_dicken_qa_chatbot/workflow.py) line 364
- If you want to install dev environment, install Pixi and checkout `make env-setup-dev`

## Rooms for improvement
* [ ] Add Guardrails
* [ ] Cloud deployment
