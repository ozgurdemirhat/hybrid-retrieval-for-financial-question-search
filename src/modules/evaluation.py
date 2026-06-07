import ir_measures
import pandas as pd
from ir_measures import MAP, P, Recall, nDCG

from modules.data import load_qrels, load_queries


METRICS = [nDCG@10, nDCG@20, MAP, Recall@100, P@10]


class EvaluationSummary:
    def __init__(self, mode, split, results_path, metrics):
        self.mode = mode
        self.split = split
        self.results_path = results_path
        self.metrics = metrics


def build_qrels_frame(split_name):
    return pd.DataFrame(load_qrels(split_name), columns=["query_id", "doc_id", "relevance"])


def evaluate_mode(
    pipeline,
    split,
    mode,
    output_path,
    limit_queries=None,
    top_k=100,
    candidate_k=100,
):
    split_name = f"beir/fiqa/{split}"
    queries = load_queries(split_name, limit_queries=limit_queries)
    query_ids = {query.query_id for query in queries}
    rows = []
    for query in queries:
        results = pipeline.search(
            query=query.text,
            mode=mode,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        for rank, result in enumerate(results, start=1):
            rows.append(
                {
                    "query_id": query.query_id,
                    "doc_id": result.doc_id,
                    "score": result.score,
                    "rank": rank,
                    "system": mode,
                }
            )
    run_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(output_path, index=False)

    qrels_df = build_qrels_frame(split_name)
    qrels_df = qrels_df[qrels_df["query_id"].isin(query_ids)]
    metric_values = ir_measures.calc_aggregate(METRICS, qrels_df, run_df)
    metrics = {str(metric): float(value) for metric, value in metric_values.items()}
    return EvaluationSummary(mode=mode, split=split, results_path=output_path, metrics=metrics)
