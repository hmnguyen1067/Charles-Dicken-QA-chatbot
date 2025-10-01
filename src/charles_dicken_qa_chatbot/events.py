from llama_index.core import Document
from llama_index.core.workflow import Event
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


class SourceExtractionEvent(Event):
    """Result of running extraction from csv files"""

    docs: list[Document]


class GutenbergIDExtractionEvent(Event):
    """Result of running extraction from default on startup"""

    docs: list[Document]


class RetrievalDatasetEvent(Event):
    """Result of creating synthetic QnA dataset from nodes"""

    qa_dataset: EmbeddingQAFinetuneDataset


class OpikDatasetEvent(Event):
    """Result of creating synthetic Opik dataset from nodes"""

    done: bool


class RetrivalEvalEvent(Event):
    set_internal_retriever: bool


class ContextInitializationEvent(Event):
    set_ctx: bool
