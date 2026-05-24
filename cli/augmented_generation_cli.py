#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hybrid_search import HybridSearch


GEMMA_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_CANDIDATES = (
    GEMMA_MODEL,
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
)
MAX_ATTEMPTS_PER_MODEL = 3


def load_movies() -> list[dict]:
    file_path_movies = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    with open(file_path_movies, "r") as f:
        data_json = json.load(f)
    return data_json["movies"]


def build_docs_string(results: list[dict]) -> str:
    lines = []
    for result in results:
        lines.append(f"{result.get('title', '')}: {result.get('document', '')}")
    return "\n".join(lines)


def build_rag_prompt(query: str, docs: str) -> str:
    return f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""


def build_fallback_answer(query: str, results: list[dict]) -> str:
    if not results:
        return f"I could not find any strong matches for '{query}'."

    titles = [result.get("title", "") for result in results if result.get("title")]
    best_match = titles[0] if titles else "the top retrieved title"
    if len(titles) == 1:
        related_text = titles[0]
    elif len(titles) == 2:
        related_text = f"{titles[0]} and {titles[1]}"
    else:
        related_text = f"{', '.join(titles[:-1])}, and {titles[-1]}"

    return (
        f"Based on the retrieved titles, {best_match} is the strongest match for '{query}'. "
        f"Other relevant options include {related_text}."
    )


def generate_response(client: Any, prompt: str):
    last_error = None

    for model in MODEL_CANDIDATES:
        if not model:
            continue

        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                return client.models.generate_content(model=model, contents=prompt)
            except Exception as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS_PER_MODEL:
                    import time

                    time.sleep(attempt)

    raise RuntimeError(f"Gemini request failed after trying multiple models: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            api_key = os.environ.get("GEMINI_API_KEY")

            search = HybridSearch(load_movies())
            results = search.rrf_search(query, 60, 5)[:5]
            docs = build_docs_string(results)
            prompt = build_rag_prompt(query, docs)

            if api_key:
                from google import genai

                client = genai.Client(api_key=api_key)
                response_text = generate_response(client, prompt).text or ""
            else:
                response_text = build_fallback_answer(query, results)

            print("Search Results:")
            for result in results:
                print(f"- {result['title']}")

            print("\nRAG Response:")
            print(response_text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
