import ir_datasets

from modules.preprocess import normalize_whitespace


class FiqaDocument:
    def __init__(self, doc_id, text):
        self.doc_id = doc_id
        self.text = text


class FiqaQuery:
    def __init__(self, query_id, text):
        self.query_id = query_id
        self.text = text


def load_dataset(name):
    return ir_datasets.load(name)


def load_documents(dataset_name, max_docs=None):
    docs = []
    dataset = load_dataset(dataset_name)
    for index, doc in enumerate(dataset.docs_iter()):
        docs.append(FiqaDocument(doc_id=doc.doc_id, text=normalize_whitespace(doc.text)))
        if max_docs and index + 1 >= max_docs:
            break
    return docs


def load_queries(split_name, limit_queries=None):
    queries = []
    dataset = load_dataset(split_name)
    for index, query in enumerate(dataset.queries_iter()):
        queries.append(FiqaQuery(query_id=query.query_id, text=normalize_whitespace(query.text)))
        if limit_queries and index + 1 >= limit_queries:
            break
    return queries


def load_qrels(split_name):
    dataset = load_dataset(split_name)
    return [
        (qrel.query_id, qrel.doc_id, int(qrel.relevance))
        for qrel in dataset.qrels_iter()
    ]
