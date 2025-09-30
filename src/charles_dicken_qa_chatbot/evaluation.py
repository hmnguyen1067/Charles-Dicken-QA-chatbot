import opik
from opik.evaluation.metrics import (
    Hallucination,
    Usefulness,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
)
from opik.evaluation import evaluate
from llama_index.core.evaluation import (
    RetrieverEvaluator,
)


@opik.track
def query_vector(query, query_engine):
    response = query_engine.query(query)
    return response


def evaluation_task(x, query_engine):
    return {
        "output": query_vector(x["query"], query_engine),
        "context": x["reference_contexts"],
        "expected_output": x["reference_answer"],
    }


def make_task(query_engine):
    def _task(x):
        return evaluation_task(x, query_engine)

    return _task


@opik.track
def run_evaluation(
    dataset: opik.Dataset,
    query_engine,
    llm_model: str,
    metrics: list = [],
):
    task = make_task(query_engine)

    if len(metrics) == 0:
        hallucination_metric = Hallucination(model=llm_model)
        usefulness_metric = Usefulness(model=llm_model)
        answer_relevance_metric = AnswerRelevance(model=llm_model)
        context_precision_metric = ContextPrecision(model=llm_model)
        context_recall_metric = ContextRecall(model=llm_model)

        metrics = [
            hallucination_metric,
            usefulness_metric,
            answer_relevance_metric,
            context_precision_metric,
            context_recall_metric,
        ]

    evaluation = evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=metrics,
        scoring_key_mapping={"input": "query"},
        experiment_config={"rag": "base"},
    )

    scores = evaluation.aggregate_evaluation_scores()
    for metric_name, statistics in scores.aggregated_scores.items():
        print(f"{metric_name}: {statistics}")
    return scores


@opik.track
async def retrieval_results(retriever, eval_dataset):
    """Function to get retrieval results for a retriever and evaluation dataset"""

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )

    eval_results = await retriever_evaluator.aevaluate_dataset(eval_dataset)

    return eval_results
