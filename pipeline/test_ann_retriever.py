import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.ann_retriever import ANNRetriever
from query.strategies import QueryRewriter
from utils.retriever_utils import load_ms_marco_corpus
from configs.config import DATA_PATH


def main():
    print("Loading MS MARCO corpus...")
    corpus = load_ms_marco_corpus(data_path=os.path.join(DATA_PATH, "ms_marco.json"))
    print(f"Loaded {len(corpus)} passages.")

    method = "hnsw"  # 🔁 Change to "pq" to test PQ variant

    print(f"Initializing ANNRetriever with method: {method} ...")
    retriever = ANNRetriever(corpus, method=method)

    query = "What is the capital of France?"
    query2 = "How long can I wait to cook my frozen ground turkey which I put in my fridge?"
    rewriter = QueryRewriter(method="refine")  # can modify to truncate/refine/paraphrase/none
    rewritten_query = rewriter.rewrite(query2)
    print(f"\nOriginal_Query: {query2}\n")
    print(f"\nRewritten_Query: {rewritten_query}\n")

    results = retriever.retrieve(query2, top_k=10)

    print("Top retrieval results:")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Document ID: {doc.doc_id}, Score: {score:.4f}")
        print(f"Text: {doc.text}\n")


if __name__ == "__main__":
    main()