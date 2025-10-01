from pydantic import BaseModel
from typing import Optional


# Request/Response Models
class QueryRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    text: str
    score: Optional[float]
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class ConfigResponse(BaseModel):
    llm_model: str
    collection_name: str
    opik_project: str
    initialized: bool


class HealthResponse(BaseModel):
    status: str
    initialized: bool


class InitializeResponse(BaseModel):
    success: bool
    message: str
