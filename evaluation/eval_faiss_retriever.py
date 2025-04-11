import json
import os
import sys
from tqdm import tqdm
import evaluate
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.faiss_retriever import FAISSRetriever
from query.strategies import QueryRewriter
from utils.metric_utils import measure_retrieval_speed_and_memory, reciprocal_rank, recall_at_k, compute_rouge, compute_bertscore, retrieval_precision
from utils.retriever_utils import load_ms_marco_corpus
from configs.config import DATA_PATH


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
    passages = data["passages"]
    qa_passage_triples = []
    for q, a, p in zip(queries, answers, passages):
        if a:
            qa_passage_triples.append((q, a[0], p['passage_text']))
    if max_queries:
        qa_passage_triples = qa_passage_triples[:max_queries]
    return qa_passage_triples

def evaluate_retriever(retriever, qa_pairs, rewriter=None, top_k=10):
    total_recall, total_precision, total_rr = 0.0, 0.0, 0.0
    total_rouge = []
    total_bertscore = []
    total_retrieval_speed_and_memory = []

    global_bertscore = evaluate.load("bertscore")

    for query, answer, passage_texts in tqdm(qa_pairs, desc="Evaluating", unit="query"):
        if rewriter:
            query = rewriter.rewrite(query)

        # Retrieve documents
        results = retriever.retrieve(query, top_k=top_k)

        # Determine which documents are relevant (based on passage substring matching)
        relevant_ids = {
            doc.doc_id for doc in retriever.documents
            if any(gt_passage.strip() in doc.text for gt_passage in passage_texts)
        }

        # Compute metrics
        retrieved_texts = [doc.text for doc, _ in results]
        total_recall += recall_at_k(results, relevant_ids, k=top_k)
        total_precision += retrieval_precision(results, relevant_ids, k=top_k)
        total_rr += reciprocal_rank(results, relevant_ids)
        total_retrieval_speed_and_memory.append(measure_retrieval_speed_and_memory(retriever, query))
        total_rouge.append(compute_rouge(answer, retrieved_texts[0]))

        bert_score_result = compute_bertscore(global_bertscore, [retrieved_texts[0]], [answer])
        total_bertscore.append(bert_score_result)
        torch.cuda.empty_cache()

    n = len(qa_pairs)
    print(f"Average Recall@{top_k}: {total_recall / n:.4f}")
    print(f"Average Precision@{top_k}: {total_precision / n:.4f}")
    print(f"Average Reciprocal Rank: {total_rr / n:.4f}")
    print(f"Average Retrieval Speed and Memory: {sum(d['retrieval_time'] for d in total_retrieval_speed_and_memory) / n:.4f} seconds")
    print(f"Average Peak Memory Usage: {sum(d['peak_memory_bytes'] for d in total_retrieval_speed_and_memory) / n / 1024:.4f} KB")
    print(f"Average ROUGE: {sum(d['rouge1'].fmeasure for d in total_rouge) / n:.4f}")
    print(f"Average BERTScore: {sum(d['f1'] for d in total_bertscore) / n:.4f}")

if __name__ == "__main__":
    corpus = load_ms_marco_corpus(data_path=os.path.join(DATA_PATH, "ms_marco.json"))
    ground_truth_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    retriever = FAISSRetriever(corpus)
    query_answer_pairs = load_query_answer_pairs(ground_truth_path, max_queries=1000)
    rewriter = QueryRewriter(method="paraphrase")  # change to truncate/refine/none if needed
    evaluate_retriever(retriever, query_answer_pairs, rewriter=rewriter, top_k=10)