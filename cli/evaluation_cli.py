#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from hybrid_search import HybridSearch


def load_movies() -> list[dict]:
    file_path_movies = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    with open(file_path_movies, "r") as f:
        data_json = json.load(f)
    return data_json["movies"]


def load_golden_dataset() -> dict:
    file_path_golden = Path(
        "~/Krish/RAG/rag-search-engine/data/golden_dataset.json"
    ).expanduser()
    with open(file_path_golden, "r") as f:
        return json.load(f)


def precision_at_k(retrieved_titles: list[str], relevant_titles: list[str], limit: int) -> float:
    relevant_set = set(relevant_titles)
    matches = sum(1 for title in retrieved_titles[:limit] if title in relevant_set)
    return matches / limit if limit > 0 else 0.0


def recall_at_k(retrieved_titles: list[str], relevant_titles: list[str], limit: int) -> float:
    relevant_set = set(relevant_titles)
    matches = sum(1 for title in retrieved_titles[:limit] if title in relevant_set)
    return matches / len(relevant_set) if relevant_set else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_golden_dataset()
    search = HybridSearch(load_movies())

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        results = search.rrf_search(query, 60, limit)[:limit]
        retrieved_titles = [result["title"] for result in results]
        precision = precision_at_k(retrieved_titles, relevant_titles, limit)
        recall = recall_at_k(retrieved_titles, relevant_titles, limit)
        f1 = f1_score(precision, recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}\n")


if __name__ == "__main__":
    main()
