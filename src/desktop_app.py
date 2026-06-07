import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

from modules.data import load_qrels, load_queries
from modules.fusion import weighted_reciprocal_rank_fusion
from modules.retrievers import BM25Retriever, DenseRetriever


def main():
    queries_by_id = {
        query.query_id: query.text
        for query in load_queries("beir/fiqa/test")
    }
    demo_queries = [
        {"id": "622", "text": queries_by_id["622"], "ndcg": 0.6240505200},
        {"id": "2376", "text": queries_by_id["2376"], "ndcg": 0.9098999913},
        {"id": "864", "text": queries_by_id["864"], "ndcg": 0.7618870796},
        {"id": "10674", "text": queries_by_id["10674"], "ndcg": 0.8477774489},
        {"id": "1393", "text": queries_by_id["1393"], "ndcg": 0.6796878192},
        {"id": "1530", "text": queries_by_id["1530"], "ndcg": 0.9674679835},
        {"id": "4265", "text": queries_by_id["4265"], "ndcg": 0.7316362001},
        {"id": "4504", "text": queries_by_id["4504"], "ndcg": 0.8772153153},
        {"id": "2568", "text": queries_by_id["2568"], "ndcg": 0.6101543718},
        {"id": "4844", "text": queries_by_id["4844"], "ndcg": 0.8065735964},
        {"id": "10808", "text": queries_by_id["10808"], "ndcg": 0.9469024295},
        {"id": "6679", "text": queries_by_id["6679"], "ndcg": 0.6978817289},
    ]
    qrels = {}
    for query_id, doc_id, relevance in load_qrels("beir/fiqa/test"):
        if relevance > 0:
            qrels.setdefault(query_id, []).append((doc_id, relevance))

    window = tk.Tk()
    window.title(
        "Papatyalar FiQA Search - "
        "all-mpnet-base-v2 + weighted RRF alpha=0.95"
    )
    window.geometry("1180x720")
    window.minsize(860, 520)
    window.columnconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)

    form = ttk.Frame(window, padding=8)
    form.grid(row=0, column=0, sticky="ew")
    form.columnconfigure(1, weight=1)

    ttk.Label(form, text="Query source").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 8),
    )
    query_options = [
        f"{number}. [{query['id']}] nDCG@10 {query['ndcg']:.4f} - {query['text']}"
        for number, query in enumerate(demo_queries, start=1)
    ]
    query_options.append("Free text query")
    query_var = tk.StringVar(value=query_options[0])
    query_combo = ttk.Combobox(
        form,
        textvariable=query_var,
        values=query_options,
        state="readonly",
        font=("Helvetica", 13),
    )
    query_combo.grid(
        row=0,
        column=1,
        sticky="ew",
        padx=(0, 8),
        ipady=4,
    )
    query_combo.current(0)

    ttk.Label(form, text="Free text").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 8),
        pady=(8, 0),
    )
    free_text_var = tk.StringVar(
        value="How should I diversify my investments before retirement?"
    )
    free_text_entry = ttk.Entry(
        form,
        textvariable=free_text_var,
        font=("Helvetica", 13),
    )
    free_text_entry.grid(
        row=1,
        column=1,
        columnspan=2,
        sticky="ew",
        padx=(0, 8),
        pady=(8, 0),
        ipady=4,
    )

    results_frame = ttk.Frame(window, padding=(8, 0, 8, 8))
    results_frame.grid(row=1, column=0, sticky="nsew")
    results_frame.columnconfigure(0, weight=1)
    results_frame.columnconfigure(1, weight=1)
    results_frame.rowconfigure(0, weight=1)

    judged_frame = ttk.LabelFrame(
        results_frame,
        text="Test qrels ranking",
    )
    judged_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    judged_frame.columnconfigure(0, weight=1)
    judged_frame.rowconfigure(0, weight=1)

    model_frame = ttk.LabelFrame(
        results_frame,
        text=(
            "Best model ranking "
            "(all-mpnet-base-v2 + weighted RRF alpha=0.95)"
        ),
    )
    model_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
    model_frame.columnconfigure(0, weight=1)
    model_frame.rowconfigure(0, weight=1)

    judged_box = scrolledtext.ScrolledText(
        judged_frame,
        wrap=tk.WORD,
        font=("Menlo", 12),
        padx=14,
        pady=14,
    )
    judged_box.grid(row=0, column=0, sticky="nsew")
    judged_box.configure(state=tk.DISABLED)

    model_box = scrolledtext.ScrolledText(
        model_frame,
        wrap=tk.WORD,
        font=("Menlo", 12),
        padx=14,
        pady=14,
    )
    model_box.grid(row=0, column=0, sticky="nsew")
    model_box.configure(state=tk.DISABLED)

    results = queue.Queue()
    bm25 = None
    dense = None

    def set_text(box, text):
        box.configure(state=tk.NORMAL)
        box.delete("1.0", tk.END)
        box.insert(tk.END, text)
        box.configure(state=tk.DISABLED)

    def preview(text):
        text = " ".join(text.split())
        if len(text) <= 520:
            return text
        return f"{text[:517]}..."

    def search(query):
        nonlocal bm25, dense
        if bm25 is None:
            bm25 = BM25Retriever(
                Path(__file__).resolve().parents[1]
                / "artifacts/cache/fiqa_bm25_full.pkl"
            )
            bm25.load()
            dense = DenseRetriever(
                str(
                    Path(__file__).resolve().parents[1]
                    / "artifacts/models/all-mpnet-base-v2"
                ),
                Path(__file__).resolve().parents[1]
                / "artifacts/cache/fiqa_dense_all-mpnet-base-v2_full",
                64,
            )
            dense.load()

        bm25_hits = bm25.search(query["text"], top_k=100)
        dense_hits = dense.search(query["text"], top_k=100)
        texts = {
            hit.doc_id: hit.text
            for hit in [*bm25_hits, *dense_hits]
        }
        model_hits = weighted_reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            [0.95, 0.05],
            60,
        )[:10]

        if query["id"] == "free-text":
            header = [
                "Free text query",
                "nDCG@10: not available",
                query["text"],
                "-" * 72,
            ]
            judged_lines = [
                *header,
                "Free text queries do not have FiQA test qrels.",
                "",
            ]
            relevance_by_doc = {}
        else:
            header = [
                f"Query {query['id']}",
                f"nDCG@10: {query['ndcg']:.4f}",
                query["text"],
                "-" * 72,
            ]
            judged_lines = list(header)
            doc_lookup = dict(zip(bm25.doc_ids, bm25.doc_texts))
            for rank, (doc_id, relevance) in enumerate(
                sorted(
                    qrels[query["id"]],
                    key=lambda item: (-item[1], item[0]),
                )[:10],
                start=1,
            ):
                judged_lines.append(
                    f"{rank:02d}. Document {doc_id} - "
                    f"Test relevance {relevance}"
                )
                judged_lines.append(preview(doc_lookup[doc_id]))
                judged_lines.append("")
            relevance_by_doc = dict(qrels[query["id"]])

        model_lines = list(header)
        for rank, (doc_id, score) in enumerate(model_hits, start=1):
            relevance = relevance_by_doc.get(doc_id)
            if relevance is None:
                relevance_text = " - Not judged relevant in test qrels"
            else:
                relevance_text = f" - Test relevance {relevance}"
            model_lines.append(
                f"{rank:02d}. Document {doc_id} - "
                f"Score {score:.6f}{relevance_text}"
            )
            model_lines.append(preview(texts[doc_id]))
            model_lines.append("")

        results.put(
            (
                "\n".join(judged_lines).strip(),
                "\n".join(model_lines).strip(),
            )
        )

    def run_query():
        search_button.configure(state=tk.DISABLED)
        set_text(judged_box, "Loading test qrels ranking...")
        set_text(model_box, "Running best model ranking...")
        if query_combo.current() == len(demo_queries):
            query = {
                "id": "free-text",
                "text": free_text_var.get().strip(),
            }
        else:
            query = demo_queries[query_combo.current()]
        threading.Thread(target=search, args=(query,), daemon=True).start()

    def show_results():
        if not results.empty():
            left_text, right_text = results.get()
            set_text(judged_box, left_text)
            set_text(model_box, right_text)
            search_button.configure(state=tk.NORMAL)
        window.after(100, show_results)

    search_button = ttk.Button(
        form,
        text="Run selected",
        command=run_query,
    )
    search_button.grid(row=0, column=2, padx=(0, 8), ipady=4)
    query_combo.bind("<Return>", lambda event: run_query())
    free_text_entry.bind(
        "<FocusIn>",
        lambda event: query_combo.current(len(demo_queries)),
    )
    free_text_entry.bind("<Return>", lambda event: run_query())

    window.after(100, show_results)
    run_query()
    window.mainloop()


if __name__ == "__main__":
    main()
