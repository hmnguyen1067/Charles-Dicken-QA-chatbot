import gutenbergpy.textget
import pandas as pd
from llama_index.core import Document, StorageContext
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.wikipedia import WikipediaReader

from .config import (
    get_openai_embed_model,
    get_redis_cache_storage,
    get_vector_store,
)


def get_books_from_path(source_path: str):
    df = pd.read_csv(source_path)
    return df


def extract_doc_from_gutenberg_wikipedia(df: pd.DataFrame):
    reader = WikipediaReader()

    docs = []

    for _, row in df.iterrows():
        book_id = row["Gutenberg ID"]
        book_title = row["Title"]
        book_text = (
            gutenbergpy.textget.get_text_by_id(book_id)
            .decode("utf-8")
            .replace("\r\n", "\n")
        )
        wiki_doc = reader.load_data(pages=[book_title])
        docs.extend(
            [
                Document(
                    text=book_text,
                    metadata={
                        "title": book_title,
                        "gutenberg_id": book_id,
                        "source": "book",
                    },
                ),
                Document(
                    text=wiki_doc[0].text,
                    metadata={
                        "title": book_title,
                        "gutenberg_id": book_id,
                        "source": "wikipedia",
                    },
                ),
            ]
        )
    return docs


def create_cache_context_storage(
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    redis_host: str,
    redis_port: int,
):
    vector_store = get_vector_store(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    redis_docstore, redis_indexstore, redis_cache = get_redis_cache_storage(
        collection_name=collection_name,
        redis_host=redis_host,
        redis_port=redis_port,
    )

    storage_context = StorageContext.from_defaults(
        docstore=redis_docstore, vector_store=vector_store, index_store=redis_indexstore
    )
    return storage_context, redis_cache


def create_ingestion_pipeline(storage_context, cache):
    embed_model = get_openai_embed_model()

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        include_metadata=True,
    )

    # summary_extractor = SummaryExtractor(summaries=["prev", "self"])
    keyword_extractor = KeywordExtractor(keywords=10)

    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            keyword_extractor,
            embed_model,
        ],
        vector_store=storage_context.vector_store,
        docstore=storage_context.docstore,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    return pipeline


async def generate_synthetic_eval_dataset(
    nodes,
    num_questions_per_chunk: int = 1,
    show_progress=True,
):
    dataset_generator = RagDatasetGenerator(
        nodes,
        show_progress=show_progress,
        num_questions_per_chunk=num_questions_per_chunk,
    )

    rag_dataset = await dataset_generator.agenerate_dataset_from_nodes()

    return rag_dataset
