from llama_index.retrievers.bm25 import BM25Retriever

from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, QueryBundle, get_response_synthesizer
from llama_index.core.evaluation import (
    generate_question_context_pairs,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

import pandas as pd
import Stemmer

from typing import List


def create_embedding_retriever(
    nodes_, storage_context, embed_model, similarity_top_k=2
):
    """Function to create an embedding retriever for a list of nodes"""
    vector_index = VectorStoreIndex(
        nodes=nodes_,
        use_async=True,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


def create_bm25_retriever(nodes_, similarity_top_k=2):
    """Function to create a bm25 retriever for a list of nodes"""
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes_,
        # docstore=docstore,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    return bm25_retriever


def create_eval_dataset(nodes_, llm, num_questions_per_chunk=2):
    """Function to create a evaluation dataset for a list of nodes"""
    qa_dataset = generate_question_context_pairs(
        nodes_, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    return qa_dataset


def display_results(name, eval_results):
    """Display results from evaluate."""

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in metrics},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


def create_query_engine_from_retriever(
    retriever, response_mode: ResponseMode = ResponseMode.COMPACT
):
    response_synthesizer = get_response_synthesizer(response_mode)

    return RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer
    )


class EmbeddingBM25RerankerRetriever(BaseRetriever):
    """Custom retriever that uses both embedding and bm25 retrievers and reranker"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: SentenceTransformerRerank,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        vector_nodes.extend(bm25_nodes)

        retrieved_nodes = self.reranker.postprocess_nodes(vector_nodes, query_bundle)

        return retrieved_nodes
