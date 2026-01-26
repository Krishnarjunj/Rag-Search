#!/usr/bin/env python3

import argparse
import json 
import string 
from pathlib import Path
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}


    def __add_document(self, doc_id, text):
        token = text.lower().split()
        for i in token:
            index[i] = doc_id

    def get_documents(self, term):
        


def remove_punc(query):
    punc = string.punctuation
    table = str.maketrans("", "", punc)
    clean_str = query.translate(table)
    return clean_str 

def tokenization(query):
    result = query.split()
    return result

def remove_stopwords(query, stopwords):
    for i in query:
        if i in stopwords:
            query.remove(i)
    return query

def stemming(query):
    # instance of PorterStemmer
    stemmer = PorterStemmer()
    for i in range(len(query)):
        query[i] = stemmer.stem(query[i])
    return query


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

    file_path_json = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    file_path_stop = Path("~/Krish/RAG/rag-search-engine/data/stopwords.txt").expanduser()

    with open(file_path_stop, "r") as file:
        content = file.read()

    stopwords = content.splitlines()

    user_query = remove_stopwords(user_query, stopwords)
    user_query = stemming(user_query)

    movies = {}
    
    try:
        with open(file_path_json, "r") as file:
            data = json.load(file)
    except:
        print("couldnt open file")
        return

    for ele in data["movies"]:
        movies[ele["id"]] = ele["title"]

    result = []
    found = False 

    for i, j in movies.items():
        clean_title = remove_punc(j.lower())
        tokenized_title = tokenization(clean_title)
        cleaned_title = remove_stopwords(tokenized_title, stopwords)
        preprocessed_title = stemming(cleaned_title)

        found = False 

        for word in preprocessed_title:
            for k in user_query:
                if k in word:
                    result.append(j)
                    found = True
                    break
            if found:
                break


                
        #if target in clean_title:
        #   result.append(j)
    
    if len(result) > 5:
        for i in range(5):
            print(str(i+1)+".", result[i])
    else:
        for i in range(len(result)):
            print(str(i+1) + ".", result[i])


if __name__ == "__main__":
    main()
