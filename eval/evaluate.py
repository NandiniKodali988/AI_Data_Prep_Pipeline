import argparse
import sys
from pathlib import Path

import yaml

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.indexing_agent import IndexingAgent


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["questions"]


def hit_rank(results: list[dict], expected_filename: str) -> int | None:
    """Return 1-based rank of the first result from expected_filename, or None."""
    for i, hit in enumerate(results, start=1):
        if hit["metadata"].get("filename", "") == expected_filename:
            return i
    return None


def run_eval(questions: list[dict], indexing_agent: IndexingAgent, top_k: int) -> list[dict]:
    rows = []
    for q in questions:
        query = q["query"]
        expected = q["expected_source"]
        results = indexing_agent.search(query, top_k=top_k)
        rank = hit_rank(results, expected)
        rows.append({"query": query, "expected": expected, "rank": rank, "results": results})
    return rows


def print_report(rows: list[dict], top_k: int):
    for row in rows:
        rank = row["rank"]
        print(f'\nQuery: "{row["query"]}"')
        for i, hit in enumerate(row["results"], start=1):
            fname = hit["metadata"].get("filename", "?")
            dist = hit["distance"]
            marker = " ✓" if fname == row["expected"] else ""
            print(f"  [{i}] {fname} ({dist:.4f}){marker}")
        if rank is None:
            print(f"  ✗ expected '{row['expected']}' not found in top {top_k}")

    # aggregate metrics
    n = len(rows)
    ranks = [r["rank"] for r in rows]

    def recall_at(k):
        """Fraction of queries where the correct doc appears anywhere in top-k."""
        hits = sum(1 for r in ranks if r is not None and r <= k)
        return hits, n, hits / n * 100

    def precision_at(k):
        """Average fraction of top-k results that come from the correct doc."""
        scores = []
        for row in rows:
            correct = sum(
                1 for hit in row["results"][:k]
                if hit["metadata"].get("filename", "") == row["expected"]
            )
            scores.append(correct / k)
        return sum(scores) / n

    mrr = sum((1 / r) for r in ranks if r is not None) / n

    print(f"\n{'='*45}")
    print(f"Evaluation Summary  ({n} queries, top_k={top_k})")
    print(f"{'='*45}")
    for k in [1, 3, 5]:
        if k <= top_k:
            hits, total, pct = recall_at(k)
            p = precision_at(k)
            print(f"Recall@{k}     : {hits}/{total}  ({pct:.1f}%)")
            print(f"Precision@{k}  : {p:.3f}")
    print(f"MRR          : {mrr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate search quality of the pipeline")
    parser.add_argument("--eval-set", default="eval/eval_set.yaml", type=Path)
    parser.add_argument("--chroma-db", default="./chroma_db")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(args.eval_set)
    agent = IndexingAgent(chroma_db_path=args.chroma_db)
    rows = run_eval(questions, agent, top_k=args.top_k)
    print_report(rows, top_k=args.top_k)


if __name__ == "__main__":
    main()
