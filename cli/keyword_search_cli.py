#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import string
from nltk.stem import PorterStemmer

def filter_stopwords_stemming(input_list, stop_words):
    stemmer = PorterStemmer()

    if isinstance(input_list, str):
        input_list = input_list.split()

        # removing stop words
        for word in input_list:
            if word in stop_words:
                input_list.remove(word)
        # stemming
        for i in range(len(input_list)):
            input_list[i] = stemmer.stem(input_list[i])

        res_str = ""
        for word in input_list:
            res_str += word
            res_str += " "
        res_str = res_str.rstrip()

        return res_str

    else:
        # removing stop words
        for word in input_list:
            if word in stop_words:
                input_list.remove(word)
        # stemming
        for i in range(len(input_list)):
            input_list[i] = stemmer.stem(input_list[i])
        return input_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # FILE PATH
    file_path_json = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    file_path_stop = Path("~/Krish/RAG/rag-search-engine/data/stopwords.txt").expanduser()

    # Removing punctuations
    Punctuations = string.punctuation
    table = str.maketrans("", "", Punctuations)


    # LOAD FILE - Json and stop words
    with open(file_path_json, 'r') as f:
        data_json = json.load(f)

    with open(file_path_stop, 'r') as f:
        data_stop = f.read()
    stop_words = data_stop.splitlines()

    # QUERY
    search_query = args.query.lower() #Lowercasing
    search_query.translate(table) #Removing punctuations
    tokenized_query = search_query.split() #tokenizing query
    clean_query = filter_stopwords_stemming(tokenized_query, stop_words) #removing stop words

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            pass
        case _:
            parser.print_help()

    # Adding movies to a list from data
    movie_list = []
    for m in data_json["movies"]:
        movie_list.append(m["title"])


    # Token to token comparison and adding to result
    result_list = []
    found = False
    for query_token in tokenized_query:
        for title in movie_list:
            clean_title = filter_stopwords_stemming(title.lower().translate(table), stop_words)
            for title_token in clean_title.split():
                if query_token in title_token:
                    result_list.append(title)
                    break

    # Printing Results
    if len(result_list) > 5:
        for i in range(5):
            print(f"{i+1}. {result_list[i]}")
    else:
        for i in range(len(result_list)):
            print(f"{i+1}. {result_list[i]}")



if __name__ == "__main__":
    main()
