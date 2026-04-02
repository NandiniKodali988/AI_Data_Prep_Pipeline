"""CLI entry point for the AI Data Prep Pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import Pipeline


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Data Prep Pipeline — convert documents to Markdown and index into ChromaDB"
    )
    parser.add_argument(
        "--input", "-i", required=False, type=Path, help="Input directory containing documents"
    )
    parser.add_argument(
        "--output", "-o", default="./output", type=Path, help="Output directory for Markdown files"
    )
    parser.add_argument(
        "--chroma-db", default="./chroma_db", help="Path to ChromaDB persistent storage"
    )
    parser.add_argument("--search", "-s", help="Run a search query against the indexed documents")
    parser.add_argument(
        "--query",
        "-q",
        help="Ask a question — retrieves relevant chunks and generates a grounded answer",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results for --search / --query"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.query:
        from src.agents.indexing_agent import IndexingAgent
        from src.agents.rag_agent import RAGAgent

        indexing_agent = IndexingAgent(chroma_db_path=args.chroma_db)
        rag_agent = RAGAgent()

        chunks = indexing_agent.search(args.query, top_k=args.top_k)
        if not chunks:
            print("No relevant documents found in the index.")
            return

        result = rag_agent.answer(args.query, chunks)

        print(f"\nQuestion: {args.query}\n")
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources ({result['chunks_used']} chunks): {', '.join(result['sources'])}")
        return

    if args.search:
        # Search-only mode — skip processing
        from src.agents.indexing_agent import IndexingAgent

        agent = IndexingAgent(chroma_db_path=args.chroma_db)
        results = agent.search(args.search, top_k=args.top_k)
        if not results:
            print("No results found.")
            return
        for i, hit in enumerate(results, 1):
            src = hit["metadata"].get("source_file", "unknown")
            section = hit["metadata"].get("section_heading", "")
            print(f"\n--- Result {i} (distance: {hit['distance']:.4f}) ---")
            print(f"Source: {src}" + (f" | Section: {section}" if section else ""))
            print(hit["text"][:500] + ("..." if len(hit["text"]) > 500 else ""))
        return

    if not args.input:
        print("Error: --input is required when not using --search")
        sys.exit(1)

    if not args.input.is_dir():
        print(f"Error: --input must be a directory, got: {args.input}")
        sys.exit(1)

    pipeline = Pipeline(output_dir=args.output, chroma_db_path=args.chroma_db)
    summary = pipeline.run(input_dir=args.input)

    print("\n=== Pipeline Complete ===")
    print(f"Processed : {summary['processed']} files")
    print(f"Skipped   : {summary['skipped']} files")
    print(f"Failed    : {summary['failed']} files")
    print(f"Chunks    : {summary['total_chunks']} indexed into ChromaDB")
    print(f"Output    : {args.output}")


if __name__ == "__main__":
    main()
