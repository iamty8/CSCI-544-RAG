import os
import json
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import Document

from configs.config import RETRIEVERS, DATA_PATH, QUERY_METHODS
from retrieval.retriever_base import RetrieverBase
from query.strategies import QueryRewriter
from utils.retriever_utils import parse_kv_args


class Exp:
    def __init__(self, retriever:str, retriever_args:dict, corpus_path:str, query_methods:list[str]):
        self.corpus:list = self._load_corpus(corpus_path) if retriever != "bm25" else self._load_corpus(corpus_path, max_passages=10000)
        self.retriever:RetrieverBase = RETRIEVERS[retriever](corpus=self.corpus, **retriever_args)
        self.rewriters:dict[str, QueryRewriter] = dict(
            zip(query_methods, [QueryRewriter(method=query_method) for query_method in query_methods]))

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

    def query(self, query: str, rewrite_method:str=None, top_k:int=10) -> tuple[list[tuple[Document, float]], str]:
        """
        Query the retriever with an optional rewrite method."
        """
        if rewrite_method not in self.rewriters.keys() and rewrite_method is not None:
            raise ValueError(f"Rewrite method '{rewrite_method}' not found.")
        rewriter = self.rewriters[rewrite_method] if rewrite_method else None
        if rewriter:
            rewritten_query = rewriter.rewrite(query)
        else:
            rewritten_query = query
        return self.retriever.retrieve(rewritten_query, top_k=top_k), rewritten_query
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query an experiment with optional rewrite method.")
    parser.add_argument("--corpus_path", type=str, default=os.path.join(DATA_PATH, "ms_marco.json"),
                        help="Path to the corpus JSON file.")
    parser.add_argument("--retriever", type=str, default="ann", help="Retriever name (e.g. 'bm25', 'ann').")
    parser.add_argument("--retriever_args", nargs='*', default=["method=hnsw"],
                        help="Retriever arguments as key=value pairs, e.g., method=hnsw ef=200",)
    parser.add_argument("--query", type=str, required=True, help="Query string.")
    parser.add_argument("--rewrite_method", type=str, default=None,
                        help="Query rewrite method: truncation, refine, paraphrase.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top documents to retrieve.")
    parser.add_argument("--query_methods", nargs='+', default=["truncation", "refine", "paraphrase"],
                        help="List of all available query rewriting methods.")

    parsed_args = parser.parse_args()
    retriever_args = parse_kv_args(parsed_args.retriever_args)
    exp = Exp(parsed_args.retriever, retriever_args, parsed_args.corpus_path, parsed_args.query_methods)
    results, rewritten_query = exp.query(parsed_args.query, rewrite_method=parsed_args.rewrite_method, top_k=parsed_args.top_k)
    # corpus_path = os.path.join(DATA_PATH, "ms_marco.json")
    # query_methods = QUERY_METHODS
    # exp = Exp("ann", {'method':"hnsw"}, corpus_path, query_methods)
    # query = "How long can I wait to cook my frozen ground turkey which I put in my fridge?"
    # results = exp.query(query, rewrite_method="truncation", top_k=10)
    print(f"\nOriginal Query: {parsed_args.query}\n")
    print(f"Rewritten Query: {rewritten_query}\n")
    
    print("Top retrieval results:")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Document ID: {doc.doc_id}, Score: {score:.4f}")
        print(f"Text: {doc.text}\n")