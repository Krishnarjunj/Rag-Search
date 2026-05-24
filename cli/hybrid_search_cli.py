#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

from hybrid_search import HybridSearch


GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-3-27b-it")


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


def build_enhancement_prompt(query: str, method: str) -> str:
    if method == "spell":
        return f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    if method == "rewrite":
        return f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    if method == "expand":
        return f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    raise ValueError(f"Unsupported enhancement method: {method}")


def enhance_query(query: str, method: str | None) -> str:
    if method is None:
        return query

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return query

    from google import genai

    prompt = build_enhancement_prompt(query, method)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMMA_MODEL, contents=prompt)
        enhanced_query = (response.text or "").strip()
    except Exception:
        return query

    if method == "expand":
        if not enhanced_query:
            return query
        return f"{query} {enhanced_query}".strip()

    return enhanced_query or query


def build_individual_rerank_prompt(query: str, doc: dict) -> str:
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


def build_batch_rerank_prompt(query: str, results: list[dict]) -> str:
    doc_lines = []
    for doc_id, doc in enumerate(results, start=1):
        doc_lines.append(
            f'{doc_id}: {doc.get("title", "")} - {doc.get("document", "")}'
        )
    doc_list_str = "\n".join(doc_lines)

    return f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""


def build_cross_encoder_pairs(query: str, results: list[dict]) -> list[list[str]]:
    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    return pairs


def build_evaluation_prompt(query: str, results: list[dict]) -> str:
    formatted_results = [f"{index}. {result.get('title', '')}" for index, result in enumerate(results, start=1)]
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def evaluate_results(query: str, results: list[dict]) -> list[int]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return [0] * len(results)

    from google import genai

    prompt = build_evaluation_prompt(query, results)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMMA_MODEL, contents=prompt)
        scores = json.loads((response.text or "").strip())
    except Exception:
        return [0] * len(results)

    if not isinstance(scores, list):
        return [0] * len(results)

    normalized_scores = []
    for score in scores[: len(results)]:
        try:
            normalized_scores.append(int(score))
        except (TypeError, ValueError):
            normalized_scores.append(0)

    if len(normalized_scores) < len(results):
        normalized_scores.extend([0] * (len(results) - len(normalized_scores)))

    return normalized_scores


def rerank_results(query: str, results: list[dict], method: str | None) -> list[dict]:
    if method is None:
        return results

    if method not in {"individual", "batch", "cross_encoder"}:
        raise ValueError(f"Unsupported rerank method: {method}")

    if method == "cross_encoder":
        from sentence_transformers import CrossEncoder

        try:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        except Exception:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")

        pairs = build_cross_encoder_pairs(query, results)
        scores = cross_encoder.predict(pairs)

        reranked_results = []
        for result, score in zip(results, scores, strict=False):
            reranked_result = dict(result)
            reranked_result["cross_encoder_score"] = float(score)
            reranked_results.append(reranked_result)

        reranked_results.sort(key=lambda result: result["cross_encoder_score"], reverse=True)
        return reranked_results

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return results

    from google import genai

    client = genai.Client(api_key=api_key)
    if method == "batch":
        prompt = build_batch_rerank_prompt(query, results)
        try:
            response = client.models.generate_content(model=GEMMA_MODEL, contents=prompt)
            ranked_ids = json.loads((response.text or "").strip())
        except Exception:
            return results

        rank_by_id = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
        reranked_results = []
        for doc_id, result in enumerate(results, start=1):
            reranked_result = dict(result)
            reranked_result["rerank_rank"] = rank_by_id.get(doc_id, len(results) + doc_id)
            reranked_results.append(reranked_result)

        reranked_results.sort(key=lambda result: result["rerank_rank"])
        return reranked_results

    reranked_results = []

    for index, result in enumerate(results):
        prompt = build_individual_rerank_prompt(query, result)

        try:
            response = client.models.generate_content(model=GEMMA_MODEL, contents=prompt)
            rerank_score = float((response.text or "").strip())
        except Exception:
            rerank_score = 0.0

        reranked_result = dict(result)
        reranked_result["rerank_score"] = rerank_score
        reranked_results.append(reranked_result)

        if index < len(results) - 1:
            time.sleep(3)

    reranked_results.sort(key=lambda result: result["rerank_score"], reverse=True)
    return reranked_results


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
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the search results with an LLM",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM-based reranking method",
    )

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
            query = enhance_query(args.query, args.enhance)
            if args.enhance is not None:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            search = HybridSearch(load_movies())
            search_limit = args.limit * 5 if args.rerank_method is not None else args.limit
            results = search.rrf_search(query, args.k, search_limit)[:search_limit]

            if args.rerank_method is not None:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                results = rerank_results(query, results, args.rerank_method)

            results = results[: args.limit]
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")

            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']}")
                if args.rerank_method == "individual":
                    print(f"  Re-rank Score: {result.get('rerank_score', 0.0):.3f}/10")
                if args.rerank_method == "batch":
                    print(f"  Re-rank Rank: {result.get('rerank_rank', index)}")
                if args.rerank_method == "cross_encoder":
                    print(
                        f"  Cross Encoder Score: {result.get('cross_encoder_score', 0.0):.3f}"
                    )
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['document'][:100]}...")

            if args.evaluate:
                print()
                scores = evaluate_results(query, results)
                for index, (result, score) in enumerate(zip(results, scores, strict=False), start=1):
                    print(f"{index}. {result['title']}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
