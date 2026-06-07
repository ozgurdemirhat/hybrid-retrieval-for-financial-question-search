from sentence_transformers import CrossEncoder

from modules.retrievers import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = CrossEncoder(self.model_name, max_length=256)

    def rerank(self, query, candidates, top_k=10):
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        rescored = [
            SearchResult(doc_id=candidate.doc_id, score=float(score), text=candidate.text)
            for candidate, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
