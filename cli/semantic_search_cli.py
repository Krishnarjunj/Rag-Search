#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
