# Development Patterns

## Adding Documents
Use ingestion workflow steps:
```python
# From CSV with Gutenberg IDs
await workflow.run(source_path="data/dickens_books.csv", ctx=ctx)

# Single Gutenberg ID
await workflow.run(gutenberg_id=98, ctx=ctx)
```

## Running Retrieval Evaluation
```python
# Generate QA dataset from nodes
await workflow.run(
    qa_nodes=nodes,
    num_questions_per_chunk=2,
    sample_percentage=0.15,
    similarity_top_k=3,
    best_metric="hit_rate",
    ctx=ctx
)

# Load existing dataset
await workflow.run(
    qa_json_load_path="notebooks/qa_dataset.json",
    similarity_top_k=3,
    ctx=ctx
)
```

## Response Evaluation
```python
# Create/load Opik dataset and run evaluation
await workflow.run(
    opik=True,
    opik_nodes=nodes,
    opik_dataset_name="eval-dataset",
    ctx=ctx
)
```

## Querying
```python
response = await workflow.run(
    query="What happens in Great Expectations?",
    thread_id="user-123",  # For conversation tracking
    ctx=ctx
)
```

## Notebooks
- `notebooks/ingestion_no_workflow.ipynb`: Manual ingestion without workflow
- `notebooks/generation.ipynb`: HyDE (Hypothetical Document Embeddings) experiments — not used in production workflow due to overhead (see create_query_engine_from_retriever_with_hyde in `src/charles_dicken_qa_chatbot/workflow.py`:361)
- `notebooks/ragflow.ipynb`: Retrieval and response evaluation examples
