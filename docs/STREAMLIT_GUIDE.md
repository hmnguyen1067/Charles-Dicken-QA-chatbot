# Streamlit Chatbot Guide

This guide explains how to use the Streamlit chatbot interface (frontend) with FastAPI backend for querying Charles Dickens novels using the RAG (Retrieval-Augmented Generation) system.

## Prerequisites

1. **Start Required Services**: Start Qdrant, Redis, Opik, and the API via Docker Compose:
   ```bash
   make docker-up
   ```
   This brings up the API container and uploads the Qdrant snapshot as part of initialization.

2. **Ingest Documents (optional)**: By default, the system preloads a Qdrant snapshot and the backend auto-initializes on startup. Ingestion is only needed if you want to customize or refresh the dataset. You can ingest using:
   - The notebooks in `notebooks/` directory
   - The main workflow programmatically

## Architecture Overview

The system now uses a **FastAPI backend** with a **Streamlit frontend**:

```
Streamlit Frontend (port 8501)
        ↓ HTTP Requests
FastAPI Backend (port 8001)
        ↓
RAGFlow Workflow → Redis/Qdrant
```

### Benefits of this Architecture
- **Separation of Concerns**: Frontend (UI) and backend (RAG logic) are decoupled
- **Shared State**: Single RAGFlow instance serves all users
- **Easy Scaling**: Backend can be deployed independently
- **Context Management**: Context is loaded once on backend startup

## Starting the System

### Step 1: Start Services (Backend + Frontend)

#### Option 1: Docker Compose (Recommended)
```bash
docker compose -f infra/docker-compose.yaml --profile opik --profile app up -d --build
```

#### Option 2: Local Dev (Alternative)
```bash
uvicorn src.api.api:app --reload --port 8001
```

The backend will be available at: http://localhost:8001
- API docs: http://localhost:8001/docs
- Health check: http://localhost:8001/health

### Step 2: Start Streamlit Frontend (Local Dev)
If you run the backend locally and want a local UI (not using Compose), start Streamlit with:
```bash
streamlit run src/chat/app.py --server.port 8501
```

The frontend will be available at: http://localhost:8501

## Using the Chatbot

### Initial Setup

1. **Ensure Backend is Running**: The FastAPI backend must be running on port 8001

2. **Open the Application**: Navigate to http://localhost:8501 in your web browser

3. **Check System Status**: The sidebar will show:
   - ✅ **System Ready**: Backend is running and initialized
   - ⚠️ **System Not Initialized**: Backend is running but needs initialization
   - ❌ **Backend Offline**: FastAPI server is not running

4. **Initialize the System** (if needed):
   - The backend attempts to auto-initialize on startup by loading context from Redis
   - If auto-initialization fails, click "🚀 Initialize System" in the sidebar
   - Wait for the success message: "✅ System Ready"

### Asking Questions

Once initialized, you can start asking questions about Charles Dickens novels:

1. Type your question in the chat input at the bottom
2. Press Enter or click the send button
3. The Streamlit frontend sends an HTTP request to the FastAPI backend
4. The backend:
   - Retrieves relevant context from Qdrant vector store
   - Generates a response using the LLM
   - Returns the answer along with source documents
5. The frontend displays the answer and sources

### Example Questions

Try these example questions:

- "What are the main themes in Great Expectations?"
- "Who is Pip in Charles Dickens' novels?"
- "Describe the setting of Oliver Twist"
- "What is the plot of A Tale of Two Cities?"
- "Compare the characters of Scrooge and Fagin"
- "What social issues did Dickens address in his novels?"

### Understanding the Response

Each response includes:

1. **Answer**: The generated response from the LLM
2. **Source Documents**: Expandable sections showing:
   - Book title and source (book or Wikipedia)
   - Relevance score (higher = more relevant)
   - Context snippet used to generate the answer
   - Gutenberg ID

## Features

### Chat History
- All messages are preserved during your session
- Scroll up to review previous questions and answers
- Click "🗑️ Clear Chat History" to start fresh

### Source Transparency
- Every answer shows the source documents used
- Helps verify the accuracy of responses
- Provides context and additional information

### Configuration Flexibility
- Adjust LLM model on the fly
- Switch between different collections
- Modify service endpoints without code changes

## Troubleshooting

### "Backend Offline" Error
**Cause**: FastAPI backend is not running.

**Solution**: Start the backend with Docker Compose or locally:
```bash
# Compose (recommended; includes Streamlit)
docker compose -f infra/docker-compose.yaml --profile opik --profile app up -d --build

# Local (dev)
uvicorn src.api.api:app --reload --port 8001
```

### "System Not Initialized" Error
**Cause**: The RAG system hasn't been initialized yet.

**Solution**:
1. The backend tries to auto-initialize on startup
2. If that fails, click the "🚀 Initialize System" button in the Streamlit sidebar
3. Check that context exists in Redis (from previous ingestion)

### "No context found in Redis" Error
**Cause**: No documents have been ingested into the system, or context wasn't persisted.

**Solution**:
1. Stop both backend and frontend
2. Run the ingestion process using notebooks or workflow
3. Restart the backend (it will load context on startup)
4. Restart the frontend

### Connection Errors to Backend
**Cause**: Backend URL is incorrect or backend crashed.

**Solution**:
1. Verify backend is running: `curl http://localhost:8001/health`
2. Check backend logs for errors
3. Set `API_BASE_URL` environment variable if using non-default port
4. Restart the backend

### Connection Errors to Services
**Cause**: Qdrant or Redis services are not running.

**Solution**:
```bash
# Check if services are running
docker ps

# Start all services (includes API and Streamlit)
make docker-up
```

### API Key Errors
**Cause**: `OPENAI_API_KEY` not set in environment.

**Solution**:
1. Create/update `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
2. Restart the **FastAPI backend** (frontend doesn't need it)

### Slow Responses
**Cause**: Complex queries or large document collections.

**Solutions**:
- Use more specific questions
- Consider using a faster LLM model
- Reduce `similarity_top_k` parameter (retrieves fewer documents)

## Architecture

The system consists of two main components:

### 1. FastAPI Backend (`api.py`)
- RESTful API with endpoints: `/health`, `/config`, `/initialize`, `/query`
- Manages single shared RAGFlow workflow instance
- Loads context from Redis on startup (auto-initialization)
- Handles all RAG operations (retrieval, generation)
- Port: 8001
 - Runs in Docker Compose by default (`api` service)
 - Creates the query engine from the selected retriever (see `src/charles_dicken_qa_chatbot/workflow.py`:361)

### 2. Streamlit Frontend (`app.py`)
- User interface with chat components
- Makes HTTP requests to FastAPI backend
- Session state for chat history only
- Health checks and status display
- Port: 8501
- Optional containerized service under `--profile app` in Compose, or run locally for dev

### Workflow

```
User Input (Streamlit)
        ↓ POST /query
FastAPI Backend
        ↓
RAGFlow Workflow
        ↓
Context Loading (Redis/Qdrant)
        ↓
Retriever Selection (Best Performing)
        ↓
Query Engine (Retrieve + Generate)
        ↓
Response + Sources
        ↓ HTTP Response
Display in Streamlit
```

## Configuration Options

### Environment Variables (`.env`)
```bash
OPENAI_API_KEY=your_key_here          # Required for backend
API_BASE_URL=http://localhost:8001    # Optional: FastAPI backend URL (default: http://localhost:8001)
```

### Constants (`constants.py`)
These are used by the **FastAPI backend** only:
```python
LLM_MODEL = "gpt-5-nano"          # OpenAI model to use
COLLECTION_NAME = "charles_dickens" # Qdrant collection
QDRANT_HOST = "localhost"          # Qdrant host
QDRANT_PORT = 6333                 # Qdrant port
REDIS_HOST = "localhost"           # Redis host
REDIS_PORT = 6380                  # Redis port
OPIK_BASE_URL = "http://localhost:5173/api"
OPIK_PROJ_NAME = "charles-dicken-qa"
```

## Performance Tips

1. **Keep Services Running**: Don't stop Docker services or FastAPI backend between sessions
2. **Shared Backend**: All users share the same RAGFlow instance, so initialization happens once
3. **Auto-Initialization**: Backend loads context on startup, no manual initialization needed (usually)
4. **Clear History**: If the frontend becomes slow, clear chat history (doesn't affect backend)
5. **Use Specific Questions**: More targeted questions get better responses
6. **Monitor Opik**: Check Opik dashboard for query performance metrics at http://localhost:5173
7. **Backend Logs**: Monitor FastAPI logs for debugging and performance insights

## API Endpoints

The FastAPI backend exposes the following endpoints:

### `GET /health`
Check backend health and initialization status.

**Response**:
```json
{
  "status": "healthy",
  "initialized": true
}
```

### `GET /config`
Get current system configuration.

**Response**:
```json
{
  "llm_model": "gpt-4o-mini",
  "collection_name": "charles_dickens",
  "opik_project": "charles-dicken-qa",
  "initialized": true
}
```

### `POST /initialize`
Manually initialize the RAG system from persisted context.

**Response**:
```json
{
  "success": true,
  "message": "System initialized successfully"
}
```

### `POST /query`
Query the RAG system with a question.

**Request**:
```json
{
  "question": "What are the main themes in Great Expectations?"
}
```

**Response**:
```json
{
  "answer": "The main themes in Great Expectations include...",
  "sources": [
    {
      "text": "Excerpt from the book...",
      "score": 0.89,
      "metadata": {
        "title": "Great Expectations",
        "source": "gutenberg"
      }
    }
  ]
}
```

## Next Steps

- Explore the Opik dashboard at http://localhost:5173 for detailed metrics
- Review FastAPI docs at http://localhost:8001/docs for interactive API testing
- Review source code in `api.py` (backend) and `app.py` (frontend) to customize
- Experiment with different LLM models by modifying `constants.py` and restarting the backend
- Add more books to the collection using the ingestion workflow
