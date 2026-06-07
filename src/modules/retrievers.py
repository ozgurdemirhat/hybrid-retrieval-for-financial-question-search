import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from modules.cache import dump_array, dump_json, dump_pickle, load_array, load_json, load_pickle
from modules.preprocess import tokenize_for_bm25


class SearchResult:
    def __init__(self, doc_id, score, text):
        self.doc_id = doc_id
        self.score = score
        self.text = text


class BM25Retriever:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.doc_ids = []
        self.doc_texts = []
        self.index = None

    def build(self, documents):
        tokens = [tokenize_for_bm25(doc.text) for doc in documents]
        self.doc_ids = [doc.doc_id for doc in documents]
        self.doc_texts = [doc.text for doc in documents]
        self.index = BM25Okapi(tokens)
        dump_pickle(
            self.cache_path,
            {
                "doc_ids": self.doc_ids,
                "doc_texts": self.doc_texts,
                "tokens": tokens,
            },
        )

    def load(self):
        payload = load_pickle(self.cache_path)
        self.doc_ids = payload["doc_ids"]
        self.doc_texts = payload["doc_texts"]
        self.index = BM25Okapi(payload["tokens"])

    def search(self, query, top_k=10):
        tokenized_query = tokenize_for_bm25(query)
        scores = np.asarray(self.index.get_scores(tokenized_query), dtype=np.float32)
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked = sorted(top_indices, key=lambda idx: float(scores[idx]), reverse=True)
        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                score=float(scores[idx]),
                text=self.doc_texts[idx],
            )
            for idx in ranked
        ]


class DenseRetriever:
    def __init__(self, model_name, cache_prefix, batch_size=64):
        self.model_name = model_name
        self.cache_prefix = cache_prefix
        self.batch_size = batch_size
        self.doc_ids = []
        self.doc_texts = []
        self.doc_embeddings = None
        self.model = SentenceTransformer(self.model_name)

    def build(self, documents):
        self.doc_ids = [doc.doc_id for doc in documents]
        self.doc_texts = [doc.text for doc in documents]
        embeddings = self.model.encode(
            self.doc_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.doc_embeddings = embeddings.astype(np.float32)
        dump_array(self.cache_prefix.with_suffix(".npy"), self.doc_embeddings)
        dump_json(
            self.cache_prefix.with_suffix(".json"),
            {
                "doc_ids": self.doc_ids,
                "doc_texts": self.doc_texts,
                "model_name": self.model_name,
            },
        )

    def load(self):
        payload = load_json(self.cache_prefix.with_suffix(".json"))
        self.doc_ids = payload["doc_ids"]
        self.doc_texts = payload["doc_texts"]
        self.doc_embeddings = load_array(self.cache_prefix.with_suffix(".npy")).astype(np.float32)

    def search(self, query, top_k=10):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)
        scores = self.doc_embeddings @ query_embedding
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked = sorted(top_indices, key=lambda idx: float(scores[idx]), reverse=True)
        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                score=float(scores[idx]),
                text=self.doc_texts[idx],
            )
            for idx in ranked
        ]
