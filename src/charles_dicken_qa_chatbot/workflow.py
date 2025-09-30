# from .events import DocsEvent
from datetime import datetime

import os
import pandas as pd
import opik

from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    global_handler,
    set_global_handler,
    get_response_synthesizer,
)
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


from .config import get_openai_embed_model, get_openai_model
from .ingestion import (
    create_cache_context_storage,
    create_ingestion_pipeline,
    extract_doc_from_gutenberg_wikipedia,
    generate_synthetic_eval_dataset,
    get_books_from_path,
)
from .events import (
    SourceExtractionEvent,
    RetrievalDatasetEvent,
    OpikDatasetEvent,
    RetrivalEvalEvent,
    ContextInitializationEvent,
)
from .utils import (
    create_eval_dataset,
    create_embedding_retriever,
    retrieval_results,
    create_bm25_retriever,
    EmbeddingBM25RerankerRetriever,
    display_results,
)

from .evaluation import run_evaluation


class RAGFlow(Workflow):
    def __init__(
        self,
        opik_host: str,
        opik_project_name: str,
        llm_model_name: str = "gpt-5-nano",
        collection_name: str = "charles_dickens",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        redis_host: str = "localhost",
        redis_port: int = 6380,
        *args,
        **kwargs,
    ):
        load_dotenv()

        self.today = datetime.now().strftime("%Y-%m-%d")
        self.opik_project_name = f"{opik_project_name}-{self.today}"
        os.environ["OPIK_PROJECT_NAME"] = self.opik_project_name
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        opik.configure(
            url=opik_host,
            use_local=True,
            force=True,
        )
        set_global_handler("opik")
        self.opik_callback_handler = global_handler

        self.opik_client = opik.Opik(project_name=self.opik_project_name)

        self.llm_model_name = llm_model_name
        self.collection_name = collection_name

        self.llm = get_openai_model(llm_model_name)
        self.embed_model = get_openai_embed_model()
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.storage_context, self.redis_cache = create_cache_context_storage(
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            redis_host=redis_host,
            redis_port=redis_port,
        )

        self.ingestion_pipeline = create_ingestion_pipeline(
            storage_context=self.storage_context, cache=self.redis_cache
        )

        super().__init__(*args, **kwargs)

    ### Ingestion
    @opik.track
    @step
    async def source_extraction(
        self, ctx: Context, ev: StartEvent
    ) -> SourceExtractionEvent:
        """Entry point to ingest a document, triggered by a StartEvent with `source_path`."""
        source_path = ev.get("source_path")
        if not source_path:
            return None

        df = get_books_from_path(source_path=source_path)
        docs = extract_doc_from_gutenberg_wikipedia(df)

        await ctx.store.set("docs", docs)
        print(f"Number of documents extracted: {len(docs)}")

        return SourceExtractionEvent(docs=docs)

    @opik.track
    @step
    async def ingest_from_source(
        self, ctx: Context, ev: SourceExtractionEvent
    ) -> StopEvent:
        nodes = await self.ingestion_pipeline.arun(
            documents=ev.docs,
            in_place=True,
            show_progress=True,
        )

        print(f"Number of chunks is: {len(nodes)}")

        await ctx.store.set("nodes", nodes)

        return StopEvent(result=nodes)

    ### Retrieval Eval
    @opik.track
    @step
    async def generate_qa_dataset(
        self, ctx: Context, ev: StartEvent
    ) -> RetrievalDatasetEvent:
        qa_nodes = ev.get("qa_nodes")
        if not qa_nodes:
            return None

        num_questions_per_chunk = ev.get("num_questions_per_chunk", 1)

        # Use default llm
        qa_dataset = create_eval_dataset(
            qa_nodes[:5], llm=self.llm, num_questions_per_chunk=num_questions_per_chunk
        )

        if save_path := ev.get("save_path", "qa_dataset.json"):
            qa_dataset.save_json(save_path)

        metric = ev.get("best_metric", "hit_rate")
        await ctx.store.set("best_metric", metric)

        self.similarity_top_k = ev.get("similarity_top_k", 2)
        await ctx.store.set("similarity_top_k", self.similarity_top_k)

        return RetrievalDatasetEvent(qa_dataset=qa_dataset)

    @opik.track
    @step
    async def load_qa_dataset(
        self, ctx: Context, ev: StartEvent
    ) -> RetrievalDatasetEvent:
        qa_json_load_path = ev.get("qa_json_load_path")
        if not qa_json_load_path:
            return None

        qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_json_load_path)

        metric = ev.get("best_metric", "hit_rate")
        await ctx.store.set("best_metric", metric)

        self.similarity_top_k = ev.get("similarity_top_k", 2)
        await ctx.store.set("similarity_top_k", self.similarity_top_k)

        return RetrievalDatasetEvent(qa_dataset=qa_dataset)

    @opik.track
    @step
    async def run_retrieval_evaluation(
        self, ctx: Context, ev: RetrievalDatasetEvent
    ) -> RetrivalEvalEvent:
        top_k = self.similarity_top_k

        # Define reranker
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L4-v2", top_n=top_k
        )
        nodes = await ctx.store.get("nodes")

        # Vector store embedding retriever
        embedding_retriever = create_embedding_retriever(
            nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            similarity_top_k=top_k,
        )
        embedding_retriever_results = await retrieval_results(
            embedding_retriever, ev.qa_dataset
        )

        # BM25 retriever
        bm25_retriever = create_bm25_retriever(
            nodes,
            similarity_top_k=self.similarity_top_k,
        )
        bm25_retriever_results = await retrieval_results(bm25_retriever, ev.qa_dataset)

        # Combination of embedding and bm25 with reranker
        embedding_bm25_rerank_retriever = EmbeddingBM25RerankerRetriever(
            embedding_retriever, bm25_retriever, reranker=reranker
        )
        embedding_bm25_rerank_retriever_results = await retrieval_results(
            embedding_bm25_rerank_retriever, ev.qa_dataset
        )

        results_table = pd.concat(
            [
                display_results("Embedding Retriever", embedding_retriever_results),
                display_results("BM25 Retriever", bm25_retriever_results),
                display_results(
                    "Embedding + BM25 Retriever + Reranker",
                    embedding_bm25_rerank_retriever_results,
                ),
            ],
            ignore_index=True,
            axis=0,
        )

        # Setting the default retriever to the one with highest hit rate
        best_metric = await ctx.store.get("best_metric")
        idx_loc = results_table[best_metric].idxmax()

        if idx_loc == 0:
            retriever = embedding_retriever
        elif idx_loc == 1:
            retriever = bm25_retriever
        else:
            retriever = embedding_bm25_rerank_retriever

        await ctx.store.set("best_retriever_idx", idx_loc)
        await ctx.store.set("retriever_results_table", results_table.to_json())
        self.retriever = retriever

        return RetrivalEvalEvent(set_internal_retriever=True)

    @opik.track
    @step
    async def initialize_from_context(
        self, ctx: Context, ev: StartEvent
    ) -> ContextInitializationEvent:
        initialize_ctx = ev.get("initialize_ctx", None)
        if not initialize_ctx:
            return None

        nodes = await ctx.store.get("nodes", default=[])
        if not nodes:
            print("There are no chunks in store, please ingest some first!")
            return None

        best_retriever_idx = await ctx.store.get("best_retriever_idx", default=2)
        top_k = await ctx.store.get("similarity_top_k", default=2)

        if best_retriever_idx == 0:
            self.retriever = create_embedding_retriever(
                nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                similarity_top_k=top_k,
            )
        elif best_retriever_idx == 1:
            self.retriever = create_bm25_retriever(
                nodes,
                similarity_top_k=top_k,
            )
        else:
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L4-v2", top_n=top_k
            )
            emb_retriever = create_embedding_retriever(
                nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                similarity_top_k=top_k,
            )
            bm25_retriever = create_bm25_retriever(
                nodes,
                similarity_top_k=top_k,
            )
            self.retriever = EmbeddingBM25RerankerRetriever(
                emb_retriever, bm25_retriever, reranker=reranker
            )

        return ContextInitializationEvent(set_ctx=True)

    @opik.track
    @step
    async def create_query_engine_from_retriever_with_hyde(
        self, ctx: Context, ev: RetrivalEvalEvent | ContextInitializationEvent
    ) -> StopEvent:
        response_synthesizer = get_response_synthesizer()
        retriever_query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
        )

        self.query_engine = retriever_query_engine

        # hyde = HyDEQueryTransform(self.llm, include_original=True)
        # hyde_query_engine = TransformQueryEngine(
        #     retriever_query_engine, query_transform=hyde
        # )

        # self.query_engine = hyde_query_engine
        return StopEvent()

    ### Response Eval
    @opik.track
    @step
    async def create_opik_eval_dataset(
        self, ctx: Context, ev: StartEvent
    ) -> OpikDatasetEvent | None:
        if not ev.get("opik"):
            return None

        opik_nodes = ev.get("opik_nodes")
        opik_dataset_name = ev.get("opik_dataset_name")
        if not opik_nodes and not opik_dataset_name:
            print("Required either nodes or Opik evaluation name as argument")
            return None

        num_questions_per_chunk = ev.get("num_questions_per_chunk", 1)

        if not opik_dataset_name:
            opik_dataset_name = (
                f"{self.llm_model_name}-{self.collection_name}-eval-{self.today}"
            )

        self.opik_dataset = self.opik_client.get_or_create_dataset(
            name=opik_dataset_name
        )
        items = self.opik_dataset.get_items()

        await ctx.store.set("opik_dataset_name", opik_dataset_name)

        # Pre-loaded dataset then move on to evaluation
        if items:
            return OpikDatasetEvent(done=True)
        else:
            # Insert items into dataset
            rag_dataset = await generate_synthetic_eval_dataset(
                nodes=opik_nodes[:5], num_questions_per_chunk=num_questions_per_chunk
            )
            self.opik_dataset.insert_from_pandas(rag_dataset.to_pandas())

            return OpikDatasetEvent(done=True)

    @opik.track
    @step
    async def run_response_evaluation(
        self, ctx: Context, ev: OpikDatasetEvent
    ) -> StopEvent:
        eval_result = run_evaluation(
            dataset=self.opik_dataset,
            query_engine=self.query_engine,
            llm_model=self.llm_model_name,
        )

        return StopEvent(result=eval_result)

    ### Retrieval + Synthesize
    @opik.track
    @step
    async def query_response(self, ctx: Context, ev: StartEvent) -> StopEvent:
        query = ev.get("query")

        if not query:
            return None

        if not hasattr(self, "query_engine"):
            print(
                "Query engine is empty, load some documents to Context before querying!"
            )
            return None

        await ctx.store.set("query", query)
        response = self.query_engine.query(query)
        return StopEvent(result=response)
