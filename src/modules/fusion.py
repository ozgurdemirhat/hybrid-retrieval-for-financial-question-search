from collections import defaultdict


def reciprocal_rank_fusion(
    result_sets,
    k=60,
):
    return weighted_reciprocal_rank_fusion(
        result_sets=result_sets,
        weights=[1.0] * len(result_sets),
        k=k,
    )


def weighted_reciprocal_rank_fusion(
    result_sets,
    weights,
    k=60,
):
    fused_scores = defaultdict(float)
    for weight, results in zip(weights, result_sets):
        for rank, result in enumerate(results, start=1):
            fused_scores[result.doc_id] += weight / (k + rank)
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
