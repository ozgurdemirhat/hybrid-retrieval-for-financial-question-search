import random

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader

from modules.data import load_documents, load_qrels, load_queries


def load_training_data():
    documents = load_documents("beir/fiqa")
    doc_lookup = {doc.doc_id: doc.text for doc in documents}
    queries = {
        query.query_id: query.text
        for query in load_queries("beir/fiqa/train")
    }
    pairs = [
        (query_id, doc_id)
        for query_id, doc_id, relevance in load_qrels("beir/fiqa/train")
        if relevance > 0
    ]
    return documents, doc_lookup, queries, pairs


def train_dense(doc_lookup, queries, pairs):
    examples = [
        InputExample(texts=[queries[query_id], doc_lookup[doc_id]])
        for query_id, doc_id in pairs
    ]
    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=8,
    )
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.fit(
        train_objectives=[(dataloader, losses.MultipleNegativesRankingLoss(model))],
        epochs=1,
        warmup_steps=int(len(dataloader) * 0.1),
        optimizer_params={"lr": 2e-5},
        output_path=None,
        show_progress_bar=False,
        save_best_model=False,
    )
    model.save("artifacts/models/dense", create_model_card=False)


def train_reranker(documents, doc_lookup, queries, pairs):
    random.seed(596)
    doc_ids = [doc.doc_id for doc in documents]
    positive_docs = {}
    for query_id, doc_id in pairs:
        positive_docs.setdefault(query_id, set()).add(doc_id)

    examples = []
    for query_id, doc_id in pairs:
        query = queries[query_id]
        examples.append(InputExample(texts=[query, doc_lookup[doc_id]], label=1.0))
        negative_id = random.choice(doc_ids)
        while negative_id in positive_docs[query_id]:
            negative_id = random.choice(doc_ids)
        examples.append(InputExample(texts=[query, doc_lookup[negative_id]], label=0.0))

    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=8,
    )
    model = CrossEncoder(
        "cross-encoder/ms-marco-electra-base",
        num_labels=1,
        max_length=256,
    )
    model.fit(
        train_dataloader=dataloader,
        epochs=1,
        warmup_steps=int(len(dataloader) * 0.1),
        optimizer_params={"lr": 2e-5},
        output_path=None,
        show_progress_bar=False,
        save_best_model=False,
    )
    model.save("artifacts/models/reranker", create_model_card=False)
