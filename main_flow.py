import argparse

from charles_dicken_qa_chatbot.constants import (PHOENIX_API_URL,
                                                 PHOENIX_EVAL_DATA_NAME,
                                                 PREFECT_API_URL, QDRANT_HOST,
                                                 QDRANT_PORT, REDIS_HOST,
                                                 REDIS_PORT)
from charles_dicken_qa_chatbot.flow import rag_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prefect churn prediction flow")
    parser.add_argument(
        "--source-path",
        type=str,
        default="data/test.csv",
        help="Path to book records",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="charles_dickens",
        help="Name of collection for storage",
    )
    parser.add_argument(
        "--prefect-url",
        type=str,
        default=PREFECT_API_URL,
        help="Prefect API URL",
    )
    parser.add_argument(
        "--phoenix-url",
        type=str,
        default=PHOENIX_API_URL,
        help="Phoenix API URL",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=30,
        help="Number of samples for evaluation dataset",
    )
    parser.add_argument(
        "--phoenix-eval-name",
        type=str,
        default=PHOENIX_EVAL_DATA_NAME,
        help="Name for phoenix evaluation dataset",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=QDRANT_HOST,
        help="Qdrant host address",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=QDRANT_PORT,
        help="Qdrant host port",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default=REDIS_HOST,
        help="Redis host address",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=REDIS_PORT,
        help="Redis host port",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag_flow(
        source_path=args.source_path,
        collection_name=args.collection_name,
        prefect_url=args.prefect_url,
        phoenix_url=args.phoenix_url,
        num_eval_samples=args.num_eval_samples,
        phoenix_eval_name=args.phoenix_eval_name,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )


if __name__ == "__main__":
    main()
