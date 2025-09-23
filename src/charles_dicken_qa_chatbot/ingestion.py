import gutenbergpy.textget
import pandas as pd
from llama_index.core import Document
from llama_index.core.extractors import KeywordExtractor, SummaryExtractor
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.readers.wikipedia import WikipediaReader

from .config import (get_fastembed_model, get_openai_model,
                     get_redis_cache_storage, get_vector_store)


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
                    text=book_text, metadata={"title": book_title, "source": "book"}
                ),
                Document(
                    text=wiki_doc[0].text,
                    metadata={"title": book_title, "source": "wikipedia"},
                ),
            ]
        )
    return docs


def create_ingestion_pipeline(
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
    redis_docstore, redis_cache = get_redis_cache_storage(
        collection_name=collection_name,
        redis_host=redis_host,
        redis_port=redis_port,
    )

    text_splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separator=" ",
    )

    summary_extractor = SummaryExtractor(summaries=["prev", "self"])
    keyword_extractor = KeywordExtractor(keywords=10)
    embed_model = get_fastembed_model("BAAI/bge-base-en-v1.5")

    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            keyword_extractor,
            summary_extractor,
            embed_model,
        ],
        vector_store=vector_store,
        docstore=redis_docstore,
        cache=redis_cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    return pipeline


def generate_synthetic_eval_dataset(
    nodes,
    llm_model: str = "gpt-4.1-mini",
    num_questions_per_chunk: int = 1,
    show_progress=True,
):
    eval_llm = get_openai_model(llm_model, temperature=0.1)

    dataset_generator = RagDatasetGenerator(
        nodes,
        llm=eval_llm,
        show_progress=show_progress,
        num_questions_per_chunk=num_questions_per_chunk,
    )

    rag_dataset = dataset_generator.generate_dataset_from_nodes()
    rag_df = rag_dataset.to_pandas().drop(["reference_answer_by", "query_by"], axis=1)
    rag_df.columns = ["question", "context", "answer"]

    return rag_df
