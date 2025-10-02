"""FastAPI backend for Charles Dickens QA Chatbot."""

import redis
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core.workflow import Context, JsonSerializer

from charles_dicken_qa_chatbot.workflow import RAGFlow

from .schemas import (
    QueryRequest,
    SourceDocument,
    QueryResponse,
    ConfigResponse,
    HealthResponse,
    InitializeResponse,
)


# Global state
app_state = {
    "workflow": None,
    "ctx": None,
    "initialized": False,
}

OPIK_PROJ_NAME = os.getenv("OPIK_PROJ_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
OPIK_URL_OVERRIDE = os.getenv("OPIK_URL_OVERRIDE")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load context and initialize workflow on startup."""
    try:
        # Initialize workflow
        workflow = RAGFlow(
            opik_host=OPIK_URL_OVERRIDE,
            opik_project_name=OPIK_PROJ_NAME,
            llm_model_name=LLM_MODEL,
            collection_name=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            timeout=300,
        )

        # Load context from Redis
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
        )

        ctx_data = redis_client.get("ctx")
        if not ctx_data:
            print(
                "Warning: No context found in Redis. System will use default initialization."
            )
            ctx = Context(workflow)
            _ = await workflow.run(from_default=True, ctx=ctx)
            await workflow.run(initialize_ctx=True, ctx=ctx)

            app_state["workflow"] = workflow
            app_state["ctx"] = ctx
            app_state["initialized"] = True
            print(
                "✅ RAG System initialized from default snapshot successfully on startup"
            )
        else:
            loaded_ctx_dict = json.loads(ctx_data)
            ctx = Context.from_dict(
                workflow, loaded_ctx_dict, serializer=JsonSerializer()
            )

            # Initialize workflow with loaded context
            await workflow.run(initialize_ctx=True, ctx=ctx)

            app_state["workflow"] = workflow
            app_state["ctx"] = ctx
            app_state["initialized"] = True
            print("✅ RAG System initialized from Context successfully on startup")

    except Exception as e:
        print(f"⚠️  Failed to initialize on startup: {e}")
        print("System will require manual initialization via /initialize endpoint")

    yield

    # Shutting down clean up
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    redis_client.set("ctx", json.dumps(ctx_dict))


app = FastAPI(
    title="Charles Dickens QA Chatbot API",
    description="FastAPI backend for RAG-based question answering about Charles Dickens novels",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and system initialization status."""
    return HealthResponse(status="healthy", initialized=app_state["initialized"])


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current system configuration."""
    if not app_state["workflow"]:
        raise HTTPException(
            status_code=503,
            detail="Workflow not initialized. System may be starting up.",
        )

    return ConfigResponse(
        llm_model=app_state["workflow"].llm_model_name,
        collection_name=app_state["workflow"].collection_name,
        opik_project=app_state["workflow"].opik_project_name,
        initialized=app_state["initialized"],
    )


@app.post("/initialize", response_model=InitializeResponse)
async def initialize_system():
    """Manually initialize the RAG system from persisted context."""
    if app_state["initialized"]:
        return InitializeResponse(success=True, message="System already initialized")

    try:
        workflow = app_state.get("workflow")
        if not workflow:
            workflow = RAGFlow(
                opik_host=OPIK_URL_OVERRIDE,
                opik_project_name=OPIK_PROJ_NAME,
                llm_model_name=LLM_MODEL,
                collection_name=COLLECTION_NAME,
                qdrant_host=QDRANT_HOST,
                qdrant_port=QDRANT_PORT,
                redis_host=REDIS_HOST,
                redis_port=REDIS_PORT,
                timeout=300,
            )
            app_state["workflow"] = workflow

        # Load context from Redis
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
        )

        ctx_data = redis_client.get("ctx")
        if not ctx_data:
            raise HTTPException(
                status_code=404,
                detail="No context found in Redis. Please run document ingestion first.",
            )

        loaded_ctx_dict = json.loads(ctx_data)
        ctx = Context.from_dict(workflow, loaded_ctx_dict, serializer=JsonSerializer())

        # Initialize workflow
        await workflow.run(initialize_ctx=True, ctx=ctx)

        app_state["ctx"] = ctx
        app_state["initialized"] = True

        return InitializeResponse(
            success=True, message="System initialized successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize system: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """Query the RAG system with a question about Charles Dickens novels."""
    if not app_state["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please call /initialize endpoint first or wait for startup initialization.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        workflow = app_state["workflow"]
        ctx = app_state["ctx"]

        response = await workflow.run(
            query=request.question, thread_id=request.thread_id, ctx=ctx
        )

        # Extract answer and sources
        answer_text = str(response.response)
        sources = []

        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                source = SourceDocument(
                    text=node.node.text[:300] + "..."
                    if len(node.node.text) > 300
                    else node.node.text,
                    score=node.score if hasattr(node, "score") else None,
                    metadata=node.node.metadata,
                )
                sources.append(source)

        return QueryResponse(answer=answer_text, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
