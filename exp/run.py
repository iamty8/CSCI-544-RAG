import os
import json
import argparse
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import Document
import evaluate
from tqdm import tqdm
import torch

from configs.config import RETRIEVERS, DATA_PATH, QUERY_METHODS
from retrieval.retriever_base import RetrieverBase
from query.strategies import QueryRewriter
from utils.retriever_utils import parse_kv_args, setup_logger, get_git_commit_info, sync_logs_to_drive
from utils.metric_utils import measure_retrieval_speed_and_memory, reciprocal_rank, recall_at_k, compute_rouge, compute_bertscore, retrieval_precision


class Exp:
    def __init__(self, retriever:str, retriever_args:dict, corpus_path:str, query_methods:list[str]):
        self.corpus:list = self._load_corpus(corpus_path) if retriever != "bm25" else self._load_corpus(corpus_path, max_passages=10000)
        self.qa_pairs:list = self._load_query_answer_pairs(corpus_path) if retriever != "bm25" else self._load_query_answer_pairs(corpus_path, max_queries=10000)
        self.retriever_name:str = retriever
        print(retriever_args)
        self.retriever:RetrieverBase = RETRIEVERS[retriever](corpus=self.corpus, **retriever_args)
        self.rewriters:dict[str, QueryRewriter] = dict(
            zip([query_method for query_method in query_methods], [QueryRewriter(method=query_method) for query_method in query_methods]))

    def _load_corpus(self, corpus_path:str, max_passages:int=None) -> list:
        with open(corpus_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        # Extract individual passages
        passages = dataset["passages"]

        # Flatten all passage_text entries (each is a list of paragraphs)
        corpus:list = []
        for p in passages:
            if isinstance(p["passage_text"], list):
                corpus.extend(p["passage_text"])  # Add each paragraph to corpus
            else:
                corpus.append(p["passage_text"])  # In case it's a single string

        if max_passages:
            corpus = corpus[:max_passages]
        return corpus
    
    def _load_query_answer_pairs(self, corpus_path:str, max_queries:int=None) -> list:
        """
        Load query-answer pairs from the MS MARCO dataset.
        
        Parameters:
            max_queries (int): Maximum number of queries to load. If None, load all queries.
        
        Returns:
            list: List of tuples containing query and answer.
        """
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data["query"]
        answers = data["answers"]
        pairs = [(q, a[0]) for q, a in zip(queries, answers) if a]
        if max_queries:
            pairs = pairs[:max_queries]
        return pairs

    def query(
            self, 
            query: str, 
            rewrite_method:str=None, 
            top_k:int=10
        ) -> tuple[list[tuple[Document, float]], str]:
        """
        Query the retriever with an optional rewrite method."
        """
        if rewrite_method not in self.rewriters.keys():
            raise ValueError(f"Rewrite method '{rewrite_method}' not found.")
        rewriter = self.rewriters[rewrite_method] if rewrite_method else None
        rewritten_query = rewriter.rewrite(query)
        return self.retriever.retrieve(rewritten_query, top_k=top_k), rewritten_query
    
    def evaluate(
                self, 
                rewrite_method:str=None, 
                top_k:int=10,
                verbose:bool=True,
                max_queries:int=-1
            ) -> dict[str, float]:
        logger = setup_logger(verbose=verbose)

        commit_hash, commit_message = get_git_commit_info()
        if commit_hash:
            logger.info(f"Git Commit: {commit_hash}")
        else:
            logger.warning("Unable to retrieve git commit information.")

        total_recall, total_precision, total_rr = 0.0, 0.0, 0.0
        total_rouge = []
        total_bertscore = []
        total_retrieval_speed_and_memory = []

        global_bertscore = evaluate.load("bertscore")

        for query, answer in tqdm(self.qa_pairs[:max_queries], desc="Evaluating"):
            results, _ = self.query(query, rewrite_method=rewrite_method, top_k=top_k)

            # Compute metrics
            retrieved_texts = [doc.text for doc, _ in results]
            relevant_ids = {self.retriever.text_to_doc_id[answer]} if answer in self.retriever.text_to_doc_id else set()
            total_recall += recall_at_k(results, relevant_ids, k=top_k)
            total_precision += retrieval_precision(results, relevant_ids, k=top_k) 
            total_rr += reciprocal_rank(results, relevant_ids)
            total_retrieval_speed_and_memory.append(measure_retrieval_speed_and_memory(self.retriever, query))
            total_rouge.append(compute_rouge(answer, retrieved_texts[0]))

            bert_score_result = compute_bertscore(global_bertscore, [retrieved_texts[0]], [answer])
            total_bertscore.append(bert_score_result)
            # total_bertscore.append(compute_bertscore([retrieved_texts[0]], [answer]))
            torch.cuda.empty_cache()

        n = len(self.qa_pairs)

        metrics = {
            "Recall@k": total_recall / n,
            "Precision@k": total_precision / n,
            "MRR": total_rr / n,
            "RetrievalSpeedAvg": sum(d['retrieval_time'] for d in total_retrieval_speed_and_memory) / n,
            "MemoryUsageAvg(KB)": sum(d['peak_memory_bytes'] for d in total_retrieval_speed_and_memory) / n / 1024,
            "ROUGE-1": sum(d['rouge1'].fmeasure for d in total_rouge) / n,
            "BERTScore-F1": sum(d['f1'] for d in total_bertscore) / n
        }
        logger.info(f"{self.retriever_name} - {rewrite_method}")
        logger.info("-" * 40)

        max_key_len = max(len(key) for key in metrics.keys())
        for key, value in metrics.items():
            logger.info(f"{key:<{max_key_len}} : {value:.4f}")

        logger.info("-" * 40)
        logger.info("\n")

        sync_logs_to_drive()

        return metrics

    
if __name__ == "__main__":
    if True:
        parser = argparse.ArgumentParser(description="Query an experiment with optional rewrite method.")
        parser.add_argument("--corpus_path", type=str, default=os.path.join(DATA_PATH, "ms_marco.json"),
                            help="Path to the corpus JSON file.")
        parser.add_argument("--retriever", type=str, default="ann", help="Retriever name (e.g. 'bm25', 'ann').")
        parser.add_argument("--retriever_args", nargs='*', default=None, # default=["method=hnsw"],
                            help="Retriever arguments as key=value pairs, e.g., method=hnsw ef=200",)
        # parser.add_argument("--query", type=str, required=True, help="Query string.")
        parser.add_argument("--rewrite_method", type=str, default=None,
                            help="Query rewrite method: truncation, refine, paraphrase.")
        parser.add_argument("--top_k", type=int, default=10, help="Number of top documents to retrieve.")
        parser.add_argument("--query_methods", nargs='+', default=["truncation", "refine", "paraphrase", "no-rewrite"],
                            help="List of all available query rewriting methods.")

        parsed_args = parser.parse_args()
        retriever_args = parse_kv_args(parsed_args.retriever_args)
        exp = Exp(parsed_args.retriever, retriever_args, parsed_args.corpus_path, parsed_args.query_methods)
        # results, rewritten_query = exp.query(parsed_args.query, rewrite_method=parsed_args.rewrite_method, top_k=parsed_args.top_k)
        # print(f"\nOriginal Query: {parsed_args.query}\n")
    else:
        corpus_path = os.path.join(DATA_PATH, "ms_marco.json")
        query_methods = QUERY_METHODS
        exp = Exp("ann", {'method':"hnsw"}, corpus_path, query_methods)
    # query = "How long can I wait to cook my frozen ground turkey which I put in my fridge?"
    # results, rewritten_query = exp.query(query, rewrite_method="truncation", top_k=10)
    # print(f"\nOriginal Query: {query}\n")
    # print(f"Rewritten Query: {rewritten_query}\n")
    for rewrite_method in parsed_args.query_methods:
        print(f"\nEvaluating with rewrite method: {rewrite_method}\n")
        exp.evaluate(rewrite_method=rewrite_method, top_k=10, verbose=False)
    # exp.evaluate(rewrite_method="truncation", top_k=10, verbose=False)
    
    # print("Top retrieval results:")
    # for i, (doc, score) in enumerate(results):
    #     print(f"[{i+1}] Document ID: {doc.doc_id}, Score: {score:.4f}")
    #     print(f"Text: {doc.text}\n")