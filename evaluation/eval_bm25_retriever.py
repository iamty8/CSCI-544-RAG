import json
import os
import sys
from tqdm import tqdm
import evaluate
import torch
from rapidfuzz import fuzz

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.bm25_retriever import BM25Retriever
from query.strategies import QueryRewriter
from utils.metric_utils import (
    measure_retrieval_speed_and_memory,
    reciprocal_rank,
    recall_at_k,
    compute_rouge,
    compute_bertscore,
    retrieval_precision
)
from utils.retriever_utils import load_ms_marco_corpus
from configs.config import DATA_PATH


def load_query_answer_pairs(msmacro_path, max_queries=None):
    with open(msmacro_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data["query"]
    answers = data["answers"]
    pairs = [(q, a[0]) for q, a in zip(queries, answers) if a]
    if max_queries:
        pairs = pairs[:max_queries]
    return pairs


def evaluate_retriever(retriever, qa_pairs, rewriter=None, top_k=10):
    print("✅ Starting Evaluation...")
    print(f"🔢 Total QA pairs: {len(qa_pairs)}")

    total_recall, total_precision, total_rr = 0.0, 0.0, 0.0
    total_rouge, total_bertscore, total_retrieval_speed_and_memory = [], [], []
    global_bertscore = evaluate.load("bertscore")

    skipped_empty, skipped_invalid, skipped_nodoc, skipped_norelevant = 0, 0, 0, 0
    fuzzy_threshold = 40  # ✅ 调低 threshold

    for idx, (query, answer) in enumerate(tqdm(qa_pairs, desc="Evaluating", unit="query")):
        original_query = query
        if rewriter:
            query = rewriter.rewrite(query)
            if not query.strip():
                skipped_empty += 1
                continue

        results = retriever.retrieve(query, top_k=top_k)

        if not results or not isinstance(results, list):
            skipped_invalid += 1
            continue

        retrieved_texts = [
            doc.text for doc, _ in results
            if hasattr(doc, "text") and isinstance(doc.text, str) and doc.text.strip()
        ]
        if len(retrieved_texts) == 0:
            skipped_nodoc += 1
            continue

        # ✅ fuzzy match
        relevant_ids = set()
        for doc, _ in results:
            if hasattr(doc, "text") and doc.text:
                sim = fuzz.token_sort_ratio(answer, doc.text)
                if sim >= fuzzy_threshold or answer.lower() in doc.text.lower():
                    relevant_ids.add(doc.doc_id)

        if not relevant_ids:
            skipped_norelevant += 1
            # ✅ optional: 打印前几个失败例子以 debug
            if idx < 3:
                print(f"[❌ No fuzzy match] Q: {query}")
                print(f"A: {answer}")
                print("Top retrieved texts:")
                for doc, _ in results[:3]:
                    print(f"- {doc.text[:200]}...\n")
            continue

        total_recall += recall_at_k(results, relevant_ids, k=top_k)
        total_precision += retrieval_precision(results, relevant_ids, k=top_k)
        total_rr += reciprocal_rank(results, relevant_ids)
        total_retrieval_speed_and_memory.append(measure_retrieval_speed_and_memory(retriever, query))

        try:
            total_rouge.append(compute_rouge(answer, retrieved_texts[0]))
            total_bertscore.append(compute_bertscore(global_bertscore, [retrieved_texts[0]], [answer]))
        except Exception:
            skipped_nodoc += 1
            continue

        torch.cuda.empty_cache()

    total = len(qa_pairs)
    n = total - (skipped_empty + skipped_invalid + skipped_nodoc + skipped_norelevant)
    print(f"\n✅ Total evaluated: {n} / {total}")
    print(f"⛔ Skipped - Empty Query: {skipped_empty}, Invalid: {skipped_invalid}, No .text: {skipped_nodoc}, No Relevant: {skipped_norelevant}")

    if n == 0:
        print("❌ No queries successfully evaluated.")
        return

    print(f"\n📊 Final Evaluation:")
    print(f"Average Recall@{top_k}: {total_recall / n:.4f}")
    print(f"Average Precision@{top_k}: {total_precision / n:.4f}")
    print(f"Average Reciprocal Rank: {total_rr / n:.4f}")
    print(f"Avg Retrieval Time: {sum(d['retrieval_time'] for d in total_retrieval_speed_and_memory) / n:.4f}s")
    print(f"Avg Peak Memory: {sum(d['peak_memory_bytes'] for d in total_retrieval_speed_and_memory) / n / 1024:.2f} KB")
    print(f"Avg ROUGE: {sum(d['rouge1'].fmeasure for d in total_rouge) / n:.4f}")
    print(f"Avg BERTScore F1: {sum(d['f1'] for d in total_bertscore) / n:.4f}")

if __name__ == "__main__":
    corpus = load_ms_marco_corpus(data_path=os.path.join(DATA_PATH, "ms_marco.json"), max_passages=10000)
    ground_truth_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    retriever = BM25Retriever(corpus)
    query_answer_pairs = load_query_answer_pairs(ground_truth_path, max_queries=1000)
    rewriter = QueryRewriter(method="truncate")  # or "refine", "paraphrase", "none"
    evaluate_retriever(retriever, query_answer_pairs, rewriter=rewriter, top_k=10)