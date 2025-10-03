# Charles Dickens QA RAG-powered Chat Interface

[![Demo video](https://img.youtube.com/vi/5b3XEsJqm00/hqdefault.jpg)](https://www.youtube.com/embed/5b3XEsJqm00)

## Description
This repository provides a measured, end‑to‑end Retrieval-Augmented Generation (RAG) system dedicated to Charles Dickens' works, available publicly from Project Gutenberg. It retrieves from a curated corpus, cites sources for every answer, and includes evaluation and monitoring so you can iterate on retrievers, prompts, and models with data—while serving a reliable API and chat UI.

- Builds a Dickens knowledge base with chunked texts and rich metadata.
- Uses hybrid retrieval (embeddings + BM25) with reranking; selects the best retriever via evaluation.
- Generates answers that cite the exact passages used; the UI surfaces sources and scores.
- Persists embeddings and context in Qdrant and Redis; ships a warm-start snapshot for fast, repeatable startup.
- Exposes a FastAPI backend and a Streamlit chat UI for quick interaction and demos.
- Tracks retrieval and generation quality with Opik to compare approaches over time.
- Ships a Docker Compose stack for one-command local deployment and reproducibility.

What it answers well:
- "Where does this quote appear, and in what context?"
- "Compare Miss Havisham and Madame Defarge’s motivations."
- "What themes of redemption show up in Great Expectations?"

Technologies: Python, LlamaIndex Workflows, Qdrant, Redis, Opik, FastAPI, Streamlit, Docker, OpenAI.

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
 - Pixi (Optional for dev environment management)
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
Start all services (Qdrant, Redis, Opik, FastAPI, Streamlit):
```bash
make docker-up
# equivalent
docker compose -f infra/docker-compose.yaml --profile opik --profile app up -d --build
```

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
- Documentations: [docs/](docs/)

Notes:
- First-time setup is fastest with Docker Compose, which uploads the provided Qdrant snapshot before the API starts.
- If you prefer to ingest documents yourself, see the notebook [notebooks/ingestion_no_workflow.ipynb](notebooks/ingestion_no_workflow.ipynb), the ingestion utilities in [src/charles_dicken_qa_chatbot/ingestion.py](src/charles_dicken_qa_chatbot/ingestion.py) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
- There is also query rewriting with HyDE (Hypothetical Document Embeddings) in [notebooks/generation.ipynb](notebooks/generation.ipynb) but due to the overhead, it's not worth using it in workflow as seen in [src/charles_dicken_qa_chatbot/workflow.py](src/charles_dicken_qa_chatbot/workflow.py) line 364
- If you want to install dev environment, install Pixi and checkout `make env-setup-dev`
- See additional operational details in [docs/DOCKER.md](docs/DOCKER.md) and the UI guide in [docs/STREAMLIT_GUIDE.md](docs/STREAMLIT_GUIDE.md).

## Evaluation Criteria

This section is for DataTalksClub's LLM Zoomcamp. Please refer to [ARCHITECTURE.md](docs/ARCHITECTURE.md) for further implementation explanations of the criteria below

* Problem description
    * 0 points: The problem is not described
    * 1 point: The problem is described but briefly or unclearly
    * [x] 2 points: The problem is well-described and it's clear what problem the project solves
* Retrieval flow
    * 0 points: No knowledge base or LLM is used
    * 1 point: No knowledge base is used, and the LLM is queried directly
    * [x] 2 points: Both a knowledge base and an LLM are used in the flow
* Retrieval evaluation
    * 0 points: No evaluation of retrieval is provided
    * 1 point: Only one retrieval approach is evaluated
    * [x] 2 points: Multiple retrieval approaches are evaluated, and the best one is used
* LLM evaluation
    * 0 points: No evaluation of final LLM output is provided
    * 1 point: Only one approach (e.g., one prompt) is evaluated
    * [x] 2 points: Multiple approaches are evaluated, and the best one is used
* Interface
   * 0 points: No way to interact with the application at all
   * 1 point: Command line interface, a script, or a Jupyter notebook
   * [x] 2 points: UI (e.g., Streamlit), web application (e.g., Django), or an API (e.g., built with FastAPI)
* Ingestion pipeline
   * 0 points: No ingestion
   * 1 point: Semi-automated ingestion of the dataset into the knowledge base, e.g., with a Jupyter notebook
   * [x] 2 points: Automated ingestion with a Python script or a special tool (e.g., Mage, dlt, Airflow, Prefect)
* Monitoring
   * 0 points: No monitoring
   * [x] 1 point: User feedback is collected OR there's a monitoring dashboard `(Can be improved with collecting user feedback in Streamlit -> Opik)`
   * 2 points: User feedback is collected and there's a dashboard with at least 5 charts
* Containerization
    * 0 points: No containerization
    * 1 point: Dockerfile is provided for the main application OR there's a docker-compose for the dependencies only
    * [x] 2 points: Everything is in docker-compose
* Reproducibility
    * 0 points: No instructions on how to run the code, the data is missing, or it's unclear how to access it
    * 1 point: Some instructions are provided but are incomplete, OR instructions are clear and complete, the code works, but the data is missing
    * [x] 2 points: Instructions are clear, the dataset is accessible, it's easy to run the code, and it works. The versions for all dependencies are specified.
* Best practices
    * [x] Hybrid search: combining both text and vector search (at least evaluating it) (1 point)
    * [x] Document re-ranking (1 point)
    * [ ] User query rewriting (1 point)`(I did include it in the notebook experiment but not in workflow)`
* Bonus points (not covered in the course)
    * [ ] Deployment to the cloud (2 points)
    * [ ] Up to 3 extra bonus points if you want to award for something extra (write in feedback for what)
