#!/usr/bin/env python3

import argparse
import json 
import string 
from pathlib import Path

def remove_punc(query):
    punc = string.punctuation
    table = str.maketrans("", "", punc)
    clean_str = query.translate(table)
    return clean_str 

def tokenization(query):
    result = query.split()
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    args.query = args.query.lower()
    args.query = remove_punc(args.query)
    user_query = tokenization(args.query)

    match args.command:
        case "search":
            print("Searching for:", args.query)
            pass
        case _:
            parser.print_help()

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
    found = False 

    for i,j in movies.items():
        clean_title = remove_punc(j.lower())
        tokenized_title = tokenization(clean_title)
        for k in user_query:
            if k in tokenized_title:
                result.append(j)
                found = True
                break
        if found:
            found = False
            continue

                
        #if target in clean_title:
        #   result.append(j)
    
    if len(result) > 5:
        for i in range(len(result)):
            print(str(i+1)+".", result[i])
    else:
        for i in range(len(result)):
            print(str(i+1) + ".", result[i])


if __name__ == "__main__":
    main()
