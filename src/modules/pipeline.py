from pathlib import Path

from modules.data import load_documents
from modules.fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from modules.reranker import CrossEncoderReranker
from modules.retrievers import BM25Retriever, DenseRetriever, SearchResult


class RetrievalArtifacts:
    def __init__(self, documents, doc_lookup):
        self.documents = documents
        self.doc_lookup = doc_lookup


class SearchPipeline:
    def __init__(
        self,
        artifacts,
        bm25,
        dense,
        reranker,
    ):
        self.artifacts = artifacts
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

    def search(
        self,
        query,
        mode="weighted_rrf",
        top_k=10,
        candidate_k=100,
    ):
        candidate_k = min(candidate_k, len(self.artifacts.documents))
        top_k = min(top_k, candidate_k)
        if mode == "bm25":
            return self.bm25.search(query, top_k=top_k)
        if mode == "dense":
            return self.dense.search(query, top_k=top_k)
        if mode == "dense_rerank":
            dense_results = self.dense.search(query, top_k=candidate_k)
            return self.reranker.rerank(query, dense_results, top_k=top_k)

        bm25_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        hybrid_results = [
            SearchResult(doc_id=doc_id, score=score, text=self.artifacts.doc_lookup[doc_id])
            for doc_id, score in fused[:candidate_k]
        ]
        if mode == "hybrid":
            return hybrid_results[:top_k]
        if mode == "hybrid_rerank":
            return self.reranker.rerank(query, hybrid_results, top_k=top_k)
        if mode in {"weighted_rrf", "weighted_rrf_rerank"}:
            weighted_fused = weighted_reciprocal_rank_fusion(
                [dense_results, bm25_results],
                weights=[0.95, 0.05],
                k=60,
            )
            weighted_results = [
                SearchResult(doc_id=doc_id, score=score, text=self.artifacts.doc_lookup[doc_id])
                for doc_id, score in weighted_fused[:candidate_k]
            ]
            if mode == "weighted_rrf":
                return weighted_results[:top_k]
            return self.reranker.rerank(query, weighted_results, top_k=top_k)


def load_pipeline(max_docs=None):
    Path("artifacts/cache").mkdir(parents=True, exist_ok=True)
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    documents = load_documents("beir/fiqa", max_docs=max_docs)
    suffix = f"_{max_docs}" if max_docs else "_full"
    bm25 = BM25Retriever(Path(f"artifacts/cache/fiqa_bm25{suffix}.pkl"))
    bm25.load()
    dense = DenseRetriever(
        "sentence-transformers/all-mpnet-base-v2",
        Path(f"artifacts/cache/fiqa_dense_all-mpnet-base-v2{suffix}"),
        64,
    )
    dense.load()
    return SearchPipeline(
        RetrievalArtifacts(
            documents,
            {doc.doc_id: doc.text for doc in documents},
        ),
        bm25,
        dense,
        CrossEncoderReranker("cross-encoder/ms-marco-electra-base"),
    )


def train_base_models():
    from modules.evaluation import evaluate_mode
    from modules.train import load_training_data, train_dense, train_reranker

    documents, doc_lookup, queries, positive_pairs = load_training_data()
    train_dense(doc_lookup, queries, positive_pairs)
    train_reranker(documents, doc_lookup, queries, positive_pairs)

    artifacts = RetrievalArtifacts(
        documents,
        {doc.doc_id: doc.text for doc in documents},
    )
    bm25 = BM25Retriever(Path("artifacts/cache/fiqa_bm25_full.pkl"))
    bm25.build(documents)
    dense = DenseRetriever(
        "artifacts/models/dense",
        Path("artifacts/cache/fiqa_dense_dense_full"),
        64,
    )
    dense.build(documents)
    pipeline = SearchPipeline(
        artifacts,
        bm25,
        dense,
        CrossEncoderReranker("artifacts/models/reranker"),
    )
    for split in ("dev", "test"):
        for mode in (
            "bm25",
            "dense",
            "dense_rerank",
            "hybrid",
            "hybrid_rerank",
            "weighted_rrf",
            "weighted_rrf_rerank",
        ):
            evaluate_mode(
                pipeline,
                split,
                mode,
                Path("artifacts/runs_finetuned_big_models") / f"{split}_{mode}.csv",
                top_k=100,
                candidate_k=100,
            )
