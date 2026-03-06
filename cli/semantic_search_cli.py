#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
from pathlib import Path
import json
import re

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command Verify
    verify_parser = subparsers.add_parser("verify")

    # Command embed
    embed_parser = subparsers.add_parser("embed_text")
    embed_parser.add_argument("text", type=str)

    # Command Verify embeddings
    verify_embed = subparsers.add_parser("verify_embeddings")

    # Command embedquery
    embedquery_parser = subparsers.add_parser("embedquery")
    embedquery_parser.add_argument("query", type=str)

    # Command search
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5, nargs='?')

    # Command chunk
    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("query", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200, nargs='?')
    chunk_parser.add_argument("--overlap", type=int)

    # Command semantic chunk
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("query", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, nargs='?')
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, nargs='?')

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
            return

        case "embed_text":
            text = args.text
            embed_text(text)
            return

        case "verify_embeddings":
            verify_embeddings()
            return

        case "embedquery":
            query = args.query
            embed_query_text(query)

        case "search":
            query = args.query
            limit = args.limit
            Obj = SemanticSearch()
            file_path_movies = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
            with open(file_path_movies, 'r') as f:
                data_json = json.load(f)

            Obj.load_or_create_embeddings(data_json["movies"])

            results = Obj.search(query, limit)

            for i in results:
                print(i)

        case "chunk":
            text = args.query
            chunk_size = args.chunk_size
            overlap = args.overlap

            words = text.split()

            chunks = []
            step = chunk_size - overlap if overlap > 0 else chunk_size

            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)

            print(f"Chunking {len(text)} characters")

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            text = args.query
            chunk_size = args.max_chunk_size
            overlap = args.overlap

            sentences = re.split("(?<=[.!?])\s+", text)

            chunks = []
            step = chunk_size - overlap if overlap > 0 else chunk_size

            for i in range(0, len(sentences), step):
                chunk = " ".join(sentences[i:i + chunk_size])
                chunks.append(chunk)

            print(f"Semantically chunking {len(text)} characters")

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
