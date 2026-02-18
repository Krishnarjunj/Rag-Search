#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify()

        case _:
            parser.print_help()

def verify():
    verify_model()

if __name__ == "__main__":
    main()
