#!/usr/bin/env python3

import argparse
import json
import math
import string
from pathlib import Path
from nltk.stem import PorterStemmer
from Inverted_Index import InvertedIndex

# --- Constants ---
BM25_K1 = 1.5
BM25_b = 0.75

# --- Helper Functions ---

def filter_stopwords_stemming(input_list, stop_words):
    """Handles text preprocessing: stopword removal and Porter stemming."""
    stemmer = PorterStemmer()

    if isinstance(input_list, str):
        # Case: Input is a string (processing raw text)
        input_list = input_list.split()
        input_list = [word for word in input_list if word not in stop_words]

        # Apply stemming
        for i in range(len(input_list)):
            input_list[i] = stemmer.stem(input_list[i])

        return " ".join(input_list).rstrip()

    else:
        # Case: Input is already a list (processing tokens)
        # Fix: Use list comprehension to avoid skipping elements during removal
        input_list = [word for word in input_list if word not in stop_words]
        
        # Apply stemming
        for i in range(len(input_list)):
            input_list[i] = stemmer.stem(input_list[i])
            
        return input_list

def bm25_idf_command(term, Obj):
    """Wrapper to call BM25 IDF calculation."""
    bm25idf = Obj.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1, Obj):
    """Wrapper to call BM25 TF calculation."""
    res = Obj.get_bm25_tf(doc_id, term, k1)
    return res

# --- Main CLI Logic ---

def main() -> None:
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # Command: build
    subparsers.add_parser("build")

    # Command: tf
    tf_parser = subparsers.add_parser("tf")
    tf_parser.add_argument("docid", type=int)
    tf_parser.add_argument("term", type=str)

    # Command: idf
    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term", type=str)

    # Command: tfidf
    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("docid", type=int)
    tfidf_parser.add_argument("term", type=str)

    # Command: bm25idf
    bmdidf_parser = subparsers.add_parser("bm25idf")
    bmdidf_parser.add_argument("term", type=str)

    # Command: bm25tf
    bm25tf_parser = subparsers.add_parser("bm25tf")
    bm25tf_parser.add_argument("doc_id", type=int)
    bm25tf_parser.add_argument("term", type=str)
    bm25tf_parser.add_argument("k1", type=float, default=BM25_K1, nargs='?')
    bm25tf_parser.add_argument("b", type=float, default=BM25_b, nargs='?')

    # Command: bm25search
    bm25search_parser = subparsers.add_parser("bm25search")
    bm25search_parser.add_argument("query", type=str)
    bm25search_parser.add_argument("limit", type=int, default=5, nargs='?')

    # Setup core objects and paths
    Obj = InvertedIndex()
    file_path_json = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    file_path_stop = Path("~/Krish/RAG/rag-search-engine/data/stopwords.txt").expanduser()

    # Punctuation removal table
    table = str.maketrans("", "", string.punctuation)

    # Parse and Load resources
    args = parser.parse_args()

    with open(file_path_json, 'r') as f:
        data_json = json.load(f)

    with open(file_path_stop, 'r') as f:
        data_stop = f.read()
    stop_words = data_stop.splitlines()

    # --- Command Routing ---
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            Obj.load()
            pass

        case "build":
            Obj.build()
            Obj.save()
            return

        case "tf":
            Obj.load()
            freq = Obj.get_tf(args.docid, args.term)
            print(freq)
            return

        case "idf":
            term = args.term.translate(table)
            term = str(filter_stopwords_stemming(term, stop_words))
            
            Obj.load()
            term_match_doc_count = len(set(Obj.index[term]))
            total_doc_count = len(Obj.docmap)
            
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
            return

        case "tfidf":
            Obj.load()
            # Falls through to the bottom TF-IDF calculation block

        case "bm25idf":
            Obj.load()
            bm25_idf_command(args.term, Obj)
            return

        case "bm25tf":
            Obj.load()
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, Obj)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            return

        case "bm25search":
            Obj.load()
            Obj.bm25_search(args.query, args.limit)
            return

        case _:
            parser.print_help()

    # --- Legacy/Common TF-IDF Logic ---
    # This block executes for 'tfidf' or any command that doesn't return early
    search_query = args.term.translate(table)
    tokenized_query = search_query.split()
    clean_query = filter_stopwords_stemming(tokenized_query, stop_words)

    tf = Obj.get_tf(args.docid, args.term)
    term_match_doc_count = len(set(Obj.index[str(args.term)]))
    total_doc_count = len(Obj.docmap)

    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    tfidf = idf * tf

    print(f"TF-IDF score of '{args.term}' in document '{args.docid}': {tfidf:.2f}")

if __name__ == "__main__":
    main()
