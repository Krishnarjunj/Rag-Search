#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from hybrid_search import HybridSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    score_range = max_score - min_score
    return [(score - min_score) / score_range for score in scores]


def load_movies():
    file_path_movies = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    with open(file_path_movies, "r") as f:
        data_json = json.load(f)
    return data_json["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize scores with min-max scaling"
    )
    normalize_parser.add_argument("scores", nargs="*", type=float)

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Run weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Run hybrid search with Reciprocal Rank Fusion"
    )
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            search = HybridSearch(load_movies())
            results = search.weighted_search(args.query, args.alpha, args.limit)[: args.limit]

            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['keyword_score']:.3f}, Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {result['document'][:100]}...")
        case "rrf-search":
            search = HybridSearch(load_movies())
            results = search.rrf_search(args.query, args.k, args.limit)[: args.limit]

            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['document'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
