#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model
from lib.semantic_search import embed_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser("verify")

    embed_parser = subparsers.add_parser("embed_text")
    embed_parser = embed_parser.add_argument("term", type=str)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify()

        case "embed_text":
            term = args.term
            embed_text(term)


        case _:
            parser.print_help()

def verify():
    verify_model()


'''
def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors of different lengths")

    else:
        res = []
        for i in len(vec1):
            res.append(vec1[i] + vec2[i])

        return res

    return

def subtract_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors of different lengths")

    else:
        res = []
        for i in len(vec1):
            res.append(vec2[i] - vec1[i])

        return res

    return

'''


if __name__ == "__main__":
    main()
