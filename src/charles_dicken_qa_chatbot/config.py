import qdrant_client
from llama_index.core.ingestion import IngestionCache
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.vector_stores.qdrant import QdrantVectorStore


def get_fastembed_model(model_name: str):
    return FastEmbedEmbedding(model_name=model_name)


def get_openai_embed_model():
    return OpenAIEmbedding()


def get_openai_model(model_name: str, temperature: float = 0.0):
    return OpenAI(model=model_name, temperature=temperature)


def get_vector_store(
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    fastembed_sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
):
    client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port)
    aclient = qdrant_client.AsyncQdrantClient(host=qdrant_host, port=qdrant_port)

    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        enable_hybrid=True,
        fastembed_sparse_model=fastembed_sparse_model,
        collection_name=collection_name,
    )
    return vector_store


def get_redis_cache_storage(collection_name: str, redis_host: str, redis_port: str):
    redis_docstore = RedisDocumentStore.from_host_and_port(
        host=redis_host, port=redis_port, namespace=collection_name
    )

    redis_indexstore = RedisIndexStore.from_host_and_port(
        host=redis_host, port=redis_port, namespace=collection_name
    )

    redis_cache = IngestionCache(
        cache=RedisCache.from_host_and_port(host=redis_host, port=redis_port),
        collection=collection_name,
    )
    return redis_docstore, redis_indexstore, redis_cache
