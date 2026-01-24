#!/usr/bin/env python3

import argparse
import json 
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    file_path = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()

    movies = {}
    
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except:
        print("couldnt open file")
        return

    for ele in data["movies"]:
        movies[ele["id"]] = ele["title"]

    result = []

    target = args.query

    for i,j in movies.items():
        if target in j:
            result.append(j)
    
    for i in range(len(result)):
        print(i+1,". ", result[i])


if __name__ == "__main__":
    main()