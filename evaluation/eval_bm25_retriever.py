import json
import os
import sys
from tqdm import tqdm
import evaluate
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.bm25_retriever import BM25Retriever
from query.strategies import QueryRewriter
from utils.metric_utils import measure_retrieval_speed_and_memory, reciprocal_rank, recall_at_k, compute_rouge, compute_bertscore, retrieval_precision
from utils.retriever_utils import load_ms_marco_corpus
from configs.config import DATA_PATH
from rapidfuzz import fuzz

def load_query_answer_pairs(msmacro_path, max_queries=None):
    """
    Load query-answer pairs from the MS MARCO dataset.
    
    Parameters:
        msmacro_path (str): Path to the MS MARCO dataset.
        max_queries (int): Maximum number of queries to load. If None, load all queries.
    
    Returns:
        list: List of tuples containing query and answer.
    """
    with open(msmacro_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["query"]
    answers = data["answers"]
    pairs = [(q, a[0]) for q, a in zip(queries, answers) if a]  # Take the first answer
    if max_queries:
        pairs = pairs[:max_queries]
    return pairs

def evaluate_retriever(retriever, qa_pairs, rewriter=None, top_k=10):

    print("✅ Starting Evaluation...")
    print(f"🔢 Total QA pairs: {len(qa_pairs)}")

    total_recall, total_precision, total_rr = 0.0, 0.0, 0.0
    total_rouge, total_bertscore, total_retrieval_speed_and_memory = [], [], []

    global_bertscore = evaluate.load("bertscore")
    skipped = 0

    for idx, (query, answer) in enumerate(tqdm(qa_pairs, desc="Evaluating", unit="query")):
        original_query = query
        if rewriter:
            query = rewriter.rewrite(query)
            if not query.strip():
                print(f"[⛔ Skip {idx}] Rewritten query is empty: '{original_query}' → ''")
                skipped += 1
                continue

        results = retriever.retrieve(query, top_k=top_k)

        # 🧱 Check if results are valid tuples
        if not results or not isinstance(results, list) or not all(isinstance(r, tuple) and len(r) == 2 for r in results):
            print(f"[⚠️ Skip {idx}] Invalid or empty retrieval results for query: {original_query}")
            skipped += 1
            continue

        retrieved_texts = [doc.text for doc, _ in results if hasattr(doc, "text") and isinstance(doc.text, str) and doc.text.strip()]
        if len(retrieved_texts) == 0:
            print(f"[⚠️ Skip {idx}] No non-empty .text in retrieved docs: {original_query}")
            skipped += 1
            continue

        # ✅ Fuzzy match based relevant_ids
        relevant_ids = set()
        for doc, _ in results:
            if hasattr(doc, "text") and doc.text:
                sim = fuzz.token_sort_ratio(answer, doc.text)
                if sim >= 70:
                    relevant_ids.add(doc.doc_id)

        if not relevant_ids:
            # print(f"[⚠️ Skip {idx}] No relevant_ids found for answer (fuzzy match < 70): {answer}")
            skipped += 1
            continue

        total_recall += recall_at_k(results, relevant_ids, k=top_k)
        total_precision += retrieval_precision(results, relevant_ids, k=top_k)
        total_rr += reciprocal_rank(results, relevant_ids)
        total_retrieval_speed_and_memory.append(measure_retrieval_speed_and_memory(retriever, query))

        try:
            total_rouge.append(compute_rouge(answer, retrieved_texts[0]))
            total_bertscore.append(compute_bertscore(global_bertscore, [retrieved_texts[0]], [answer]))
        except Exception as e:
            print(f"[❌ Error @ {idx}] compute_rouge/bertscore failed: {e}")
            skipped += 1
            continue

        torch.cuda.empty_cache()

    n = len(qa_pairs) - skipped
    print(f"\n✅ Total evaluated: {n} | Skipped: {skipped}")
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
    rewriter = QueryRewriter(method="truncate")  # can modify to truncate/refine/paraphrase/none
    evaluate_retriever(retriever, query_answer_pairs, rewriter=rewriter, top_k=10)



