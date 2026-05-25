"""
main.py
-------
Entry point for the Local AI Document Understanding System.

Usage
-----
  # Process documents and write output.json
  python main.py --input_folder ./docs --output_json output.json

  # Process + semantic search
  python main.py --input_folder ./docs --search "payments due in January"

  # Adjust number of search results
  python main.py --input_folder ./docs --search "electricity usage" --top_k 3
"""

import argparse
import json
import os
import sys

from document_processor import process_documents
from classifier import classify_document
from extractor import extract_fields
from search import SemanticSearch


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_folder: str, output_json: str) -> tuple[dict, dict]:
    """
    Process all documents in *input_folder*, classify and extract fields.

    Returns
    -------
    results : dict
        Mapping of filename → {class, ...extracted fields}
    docs : dict
        Mapping of filename → raw cleaned text (used for search indexing)
    """
    print(f"\n[1/3] Reading documents from '{input_folder}' …")
    docs = process_documents(input_folder)

    if not docs:
        print("  No supported documents found (.pdf, .txt, .docx).")
        return {}, {}

    print(f"\n[2/3] Classifying and extracting fields from {len(docs)} document(s) …")
    results: dict = {}

    for fname, text in docs.items():
        doc_class = classify_document(text)
        fields = extract_fields(doc_class, text)
        results[fname] = {"class": doc_class, **fields}
        print(f"  {fname:40s} -> {doc_class}")

    print(f"\n[3/3] Writing results to '{output_json}' …")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Done. {len(results)} document(s) processed.")

    return results, docs


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def run_search(docs: dict, query: str, top_k: int = 5) -> None:
    """Build a FAISS index over *docs* and print the top-k results for *query*."""
    if not docs:
        print("No documents available for search.")
        return

    print(f"\n[Search] Building semantic index …")
    fnames = list(docs.keys())
    texts = list(docs.values())
    engine = SemanticSearch(texts, fnames)

    print(f"[Search] Query: \"{query}\"\n")
    matches = engine.search(query, top_k=top_k)

    print(f"{'Rank':<6} {'Score':>7}  Filename")
    print("-" * 55)
    for rank, (fname, score) in enumerate(matches, start=1):
        print(f"  {rank:<4} {score:>7.4f}  {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local AI Document Understanding System — classify, extract, and search documents offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        metavar="PATH",
        help="Folder containing PDF, TXT, or DOCX files to process.",
    )
    parser.add_argument(
        "--output_json",
        default="output.json",
        metavar="FILE",
        help="Path for the output JSON file (default: output.json).",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        default=None,
        help="Optional semantic search query to run after processing.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        metavar="N",
        help="Number of search results to return (default: 5).",
    )
    parser.add_argument(
        "--search_only",
        action="store_true",
        help="Skip classification/extraction and only run semantic search.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.search_only and not args.search:
        parser.error("--search_only requires --search QUERY")

    if args.search_only:
        # Just index and search, no JSON output
        print(f"\n[1/1] Reading documents from '{args.input_folder}' …")
        docs = process_documents(args.input_folder)
        run_search(docs, args.search, top_k=args.top_k)
    else:
        results, docs = run_pipeline(args.input_folder, args.output_json)
        if args.search and docs:
            run_search(docs, args.search, top_k=args.top_k)

    print()


if __name__ == "__main__":
    main()
