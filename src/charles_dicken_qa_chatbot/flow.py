import os
import random
from datetime import datetime

from phoenix.client import Client
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from .config import define_settings
from .constants import PHOENIX_API_URL, PREFECT_API_URL
from .ingestion import (create_ingestion_pipeline,
                        extract_doc_from_gutenberg_wikipedia,
                        generate_synthetic_eval_dataset, get_books_from_path)


@task
def load_settings():
    define_settings()


@task
def extract_docs_from_path(source_path: str):
    df = get_books_from_path(source_path)
    docs = extract_doc_from_gutenberg_wikipedia(df)
    return docs


@task
def ingest_documents(
    docs,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    redis_host: str,
    redis_port: int,
):
    pipeline = create_ingestion_pipeline(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    nodes = pipeline.run(
        documents=docs,
        in_place=True,
        show_progress=True,
    )

    return nodes


@task(cache_policy=NO_CACHE)
def create_phoenix_eval_dataset(
    nodes,
    px_client: Client,
    num_samples: int = 30,
    eval_name: str = "charles-dicken-qa-eval",
):
    formatted_datetime = datetime.now().strftime("%m-%d/%H:%M")

    sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))
    rag_df = generate_synthetic_eval_dataset(
        nodes=sampled_nodes,
        llm_model="gpt-4.1-mini",
        num_questions_per_chunk=1,
        show_progress=True,
    )

    _ = px_client.datasets.create_dataset(
        dataframe=rag_df,
        name=f"{eval_name}-{formatted_datetime}",
        input_keys=["question", "context"],
        output_keys=["answer"],
    )


@task(cache_policy=NO_CACHE)
def get_latest_px_dataset(px_client: Client):
    latest_px_dataset_name = px_client.datasets.list()[0]["name"]
    latest_dataset = px_client.datasets.get_dataset(dataset=latest_px_dataset_name)
    return latest_dataset


@task
def run_evaluations():
    pass


@flow(log_prints=True)
def rag_flow(
    source_path: str = "data/gutenberg",
    collection_name: str = "charles_dickens",
    prefect_url: str = PREFECT_API_URL,
    phoenix_url: str = PHOENIX_API_URL,
    num_eval_samples: int = 30,
    phoenix_eval_name: str = "charles-dicken-qa-eval",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    redis_host: str = "localhost",
    redis_port: int = 6379,
):
    # Initialization
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["PREFECT_API_URL"] = prefect_url
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_url
    px_client = Client()
    load_settings.submit().result()

    # Ingestion pipeline
    docs = extract_docs_from_path.submit(source_path).result()
    nodes = ingest_documents.submit(
        docs,
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        redis_host=redis_host,
        redis_port=redis_port,
    ).result()
    create_phoenix_eval_dataset.submit(
        nodes=nodes,
        px_client=px_client,
        num_samples=num_eval_samples,
        eval_name=phoenix_eval_name,
    ).result()
    _ = get_latest_px_dataset.submit(px_client=px_client).result()

    # Generation pipeline
