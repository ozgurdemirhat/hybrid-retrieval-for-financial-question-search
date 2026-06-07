import argparse
from pathlib import Path

from modules.data import load_documents
from modules.evaluation import evaluate_mode
from modules.pipeline import load_pipeline, train_base_models
from modules.preprocess import snippet
from modules.retrievers import BM25Retriever, DenseRetriever


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", add_help=False)
    build_parser.add_argument("--max-docs", type=int, default=None)

    subparsers.add_parser("train", add_help=False)

    search_parser = subparsers.add_parser("search", add_help=False)
    search_parser.add_argument("--max-docs", type=int, default=None)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument(
        "--mode",
        default="weighted_rrf",
    )
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--candidate-k", type=int, default=100)

    eval_parser = subparsers.add_parser("evaluate", add_help=False)
    eval_parser.add_argument("--max-docs", type=int, default=None)
    eval_parser.add_argument("--split", default="dev")
    eval_parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "dense", "hybrid", "weighted_rrf"],
    )
    eval_parser.add_argument("--top-k", type=int, default=100)
    eval_parser.add_argument("--candidate-k", type=int, default=100)
    eval_parser.add_argument("--limit-queries", type=int, default=None)
    return parser


def command_build(args):
    Path("artifacts/cache").mkdir(parents=True, exist_ok=True)
    documents = load_documents("beir/fiqa", max_docs=args.max_docs)
    suffix = f"_{args.max_docs}" if args.max_docs else "_full"
    BM25Retriever(Path(f"artifacts/cache/fiqa_bm25{suffix}.pkl")).build(documents)
    DenseRetriever(
        "sentence-transformers/all-mpnet-base-v2",
        Path(f"artifacts/cache/fiqa_dense_all-mpnet-base-v2{suffix}"),
        64,
    ).build(documents)
    print("Artifacts built successfully.")


def command_train():
    train_base_models()


def command_search(args):
    pipeline = load_pipeline(args.max_docs)
    results = pipeline.search(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    for rank, result in enumerate(results, start=1):
        print(f"{rank:02d}. [{result.doc_id}] score={result.score:.4f}")
        print(f"    {snippet(result.text)}")


def command_evaluate(args):
    pipeline = load_pipeline(args.max_docs)
    summaries = []
    for mode in args.modes:
        output_path = Path("artifacts/runs") / f"{args.split}_{mode}.csv"
        summary = evaluate_mode(
            pipeline=pipeline,
            split=args.split,
            mode=mode,
            output_path=output_path,
            limit_queries=args.limit_queries,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
        summaries.append(summary)

    for summary in summaries:
        print(f"\nMode: {summary.mode}")
        print(f"Run file: {summary.results_path}")
        for metric_name, value in summary.metrics.items():
            print(f"  {metric_name}: {value:.4f}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build":
        command_build(args)
    elif args.command == "train":
        command_train()
    elif args.command == "search":
        command_search(args)
    elif args.command == "evaluate":
        command_evaluate(args)
