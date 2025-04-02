import time
import tracemalloc

from rouge_score import rouge_scorer
import evaluate

def measure_retrieval_speed_and_memory(query_engine, query: str):
    tracemalloc.start()
    start_time = time.time()
    
    response = query_engine.query(query)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "response": response.response,
        "retrieval_time": end_time - start_time,
        "peak_memory_bytes": peak
    }

def reciprocal_rank(results, relevant_ids):
    for rank, result in enumerate(results, start=1):
        if result.metadata['doc_id'] in relevant_ids:
            return 1 / rank
    return 0

def recall_at_k(results, relevant_ids, k=10):
    top_k_ids = {r.metadata['doc_id'] for r in results[:k]}
    return len(top_k_ids & set(relevant_ids)) / len(relevant_ids)



def compute_rouge(reference: str, prediction: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)


def compute_bertscore(predictions: list, references: list):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return {
        "precision": sum(results["precision"]) / len(results["precision"]),
        "recall": sum(results["recall"]) / len(results["recall"]),
        "f1": sum(results["f1"]) / len(results["f1"]),
    }

def retrieval_precision(results, relevant_ids, k=10):
    top_k_ids = {r.metadata['doc_id'] for r in results[:k]}
    return len(top_k_ids & set(relevant_ids)) / k
