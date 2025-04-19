import os
import json
import argparse
import logging
import sys

from llama_index.core import Document
import evaluate
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import RETRIEVERS, DATA_PATH, QUERY_METHODS
from retrieval.retriever_base import RetrieverBase
from query.strategies import QueryRewriter
from utils.retriever_utils import parse_kv_args, setup_logger, get_git_commit_info, sync_logs_to_drive
from utils.metric_utils import measure_retrieval_speed_and_memory, reciprocal_rank, recall_at_k, compute_rouge, compute_bertscore, retrieval_precision


class Exp:
    def __init__(
            self, 
            retriever:str, 
            retriever_args:dict, 
            corpus_path:str, 
            query_methods:list[str],
            batch_mode:bool=False,
            debug:bool=False
        ):
        self.corpus:list = self._load_corpus(corpus_path, max_passages=1000 if debug else -1) if retriever != "bm25" else self._load_corpus(corpus_path, max_passages=10000)
        self.qa_pairs:list = self._load_query_answer_pairs(corpus_path, max_queries=1000 if debug else -1) if retriever != "bm25" else self._load_query_answer_pairs(corpus_path, max_queries=10000)
        self.retriever_name:str = retriever
        self.batch_mode:bool = batch_mode
        self.retriever:RetrieverBase = RETRIEVERS[retriever](corpus=self.corpus, **retriever_args)
        self.rewriters:dict[str, QueryRewriter] = dict(
            zip([query_method for query_method in query_methods], 
                [QueryRewriter(method=query_method, batch_mode=self.batch_mode) 
                 for query_method in query_methods]
                )
        )

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
        passages = data["passages"]
        qa_passage_triples = []
        for q, a, p in zip(queries, answers, passages):
            if a:
                qa_passage_triples.append((q, a[0], p['passage_text']))  
        if max_queries:
            qa_passage_triples = qa_passage_triples[:max_queries]
        return qa_passage_triples

    def query(
        self, 
        query: str | list[str], 
        rewrite_method: str = None, 
        top_k: int = 10, 
        batch_mode: bool = False
    ) -> tuple[list[tuple[Document, float]], str] | list[tuple[list[tuple[Document, float]], str]]:
        """
        Query the retriever, with optional rewrite and batch mode support.
        
        Args:
            query (str or List[str]): Single query or batch of queries.
            rewrite_method (str): Optional rewrite method.
            top_k (int): Number of top documents to retrieve.
            batch (bool): If True, input is treated as a batch of queries.
        
        Returns:
            Single query: (retrieved_documents, rewritten_query)
            Batch queries: List[(retrieved_documents, rewritten_query)]
        """
        if rewrite_method and rewrite_method not in self.rewriters.keys():
            raise ValueError(f"Rewrite method '{rewrite_method}' not found.")
        
        rewriter = self.rewriters[rewrite_method] if rewrite_method else None

        if not batch_mode:
            rewritten_query = rewriter.rewrite(query) if rewriter else query
            retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=top_k)
            return retrieved_docs, rewritten_query
        else:
            queries = query
            if not isinstance(queries, list):
                raise ValueError("In batch mode, 'query' must be a list of strings.")
            
            results = []
            if rewriter and hasattr(rewriter, 'batch_rewrite'):
                rewritten_queries = rewriter.batch_rewrite(queries)
            elif rewriter:
                rewritten_queries = [rewriter.rewrite(q) for q in queries]
            else:
                rewritten_queries = queries
            
            for rewritten_query in rewritten_queries:
                retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=top_k)
                results.append((retrieved_docs, rewritten_query))
            
            return results

    
    def evaluate(
                self, 
                rewrite_method:str=None, 
                top_k:int=10,
                verbose:bool=True,
                max_queries:int=-1
            ) -> dict[str, float]:
        logger = setup_logger(verbose=verbose)

        logger.info(f"{self.retriever_name} - {rewrite_method}")
        logger.info("-" * 40)

        commit_hash, commit_message = get_git_commit_info()
        if commit_hash:
            logger.info(f"Git Commit: {commit_hash}")
        else:
            logger.warning("Unable to retrieve git commit information.")

        total_recall, total_precision, total_rr = 0.0, 0.0, 0.0
        total_rouge = []
        total_bertscore = []
        total_retrieval_speed_and_memory = []

        global_bertscore = evaluate.load("bertscore", trust_remote_code=True)# evaluate.load("bertscore")

        skipped_empty, skipped_invalid, skipped_nodoc, skipped_norelevant = 0, 0, 0, 0

        for idx, (query, answer, passage_texts) in enumerate(tqdm(self.qa_pairs[:max_queries], desc="Evaluating")):
            results, rewritten_query = self.query(query, rewrite_method=rewrite_method, top_k=top_k)

            if not rewritten_query.strip():
                skipped_empty += 1
                continue

            if not results or not isinstance(results, list):
                skipped_invalid += 1
                continue

            # Compute metrics
            retrieved_texts, relevant_ids = self.retriever.result_processing(results, query, answer, passage_texts, idx)

            if retrieved_texts is None:
                skipped_nodoc += 1
                continue

            if relevant_ids is None:
                skipped_norelevant += 1
                continue

            total_recall += recall_at_k(results, relevant_ids, k=top_k)
            total_precision += retrieval_precision(results, relevant_ids, k=top_k) 
            total_rr += reciprocal_rank(results, relevant_ids)
            total_retrieval_speed_and_memory.append(measure_retrieval_speed_and_memory(self.retriever, query))
            try:
                total_rouge.append(compute_rouge(answer, retrieved_texts[0]))
                total_bertscore.append(compute_bertscore(global_bertscore, [retrieved_texts[0]], [answer]))
            except Exception:
                logger.warning(f"Error computing metrics for query: {query}")
                logger.warning(f"Answer: {answer}")
                logger.warning(f"Retrieved Texts: {retrieved_texts}")
                logger.warning(f"Rewritten Query: {rewritten_query}")
                logger.warning(f"Results: {results}")
                skipped_nodoc += 1
                continue
            torch.cuda.empty_cache()

        total = len(self.qa_pairs[:max_queries])
        n = total - (skipped_empty + skipped_invalid + skipped_nodoc + skipped_norelevant)
        logger.info(f"Evaluated {n} / {total} queries.")
        logger.info(f"Skipped - Empty Query: {skipped_empty}, Invalid: {skipped_invalid}, No Text: {skipped_nodoc}, No Relevant: {skipped_norelevant}")

        if n == 0:
            logger.error("No successful evaluations.")
            return {}
        
        metrics = {
            "Recall@k": total_recall / n,
            "Precision@k": total_precision / n,
            "MRR": total_rr / n,
            "RetrievalSpeedAvg": sum(d['retrieval_time'] for d in total_retrieval_speed_and_memory) / n,
            "MemoryUsageAvg(KB)": sum(d['peak_memory_bytes'] for d in total_retrieval_speed_and_memory) / n / 1024,
            "ROUGE-1": sum(d['rouge1'].fmeasure for d in total_rouge) / n,
            "BERTScore-F1": sum(d['f1'] for d in total_bertscore) / n
        }

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
        parser.add_argument("--debug", action="store_true", help="Enable debug mode for quick testing.")

        parsed_args = parser.parse_args()
        retriever_args = parse_kv_args(parsed_args.retriever_args)
        if parsed_args.debug:
            exp = Exp(parsed_args.retriever, retriever_args, parsed_args.corpus_path, parsed_args.query_methods, debug=True)
        else:
            exp = Exp(parsed_args.retriever, retriever_args, parsed_args.corpus_path, parsed_args.query_methods)
    else:
        corpus_path = os.path.join(DATA_PATH, "ms_marco.json")
        query_methods = QUERY_METHODS
        exp = Exp("ann", {'method':"hnsw"}, corpus_path, query_methods)
    for rewrite_method in parsed_args.query_methods:
        print(f"\nEvaluating with rewrite method: {rewrite_method}\n")
        exp.evaluate(rewrite_method=rewrite_method, top_k=10, verbose=False)