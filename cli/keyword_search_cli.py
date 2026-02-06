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

    # FILE PATH 
    file_path_json = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()


    # LOAD FILE 
    with open(file_path_json, 'r') as f:
        data_json = json.load(f)

    #QUERY 
    seach_query = args.query



    match args.command:
        case "search":
            print(f"Searching for: {seach_query}")
            pass
        case _:
            parser.print_help()

    # Adding movies to a list from data
    result_list = []
    for m in data_json["movies"]:
        if seach_query in m["title"].split():
            result_list.append(m["title"])

    #Printing Results
    if len(result_list) > 5:
        for i in range(5):
            print(f"{i+1}. {result_list[i]}")
    else:
        for i in range(len(result_list)):
            print(f"{i+1}. {result_list[i]}")



if __name__ == "__main__":
    main()
