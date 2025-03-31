import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.ann_retriever import ANNRetriever


def load_ms_marco_corpus(max_passages=None):
    """
    Loads MS MARCO passages from ../data/ms_marco.json.
    Optionally limit to first N passages.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Fallback key detection
    if "passage" in dataset:
        corpus = dataset["passage"]
    elif "passage_text" in dataset:
        corpus = dataset["passage_text"]
    else:
        corpus = []
        for key, value in dataset.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                corpus = value
                break

    if max_passages:
        corpus = corpus[:max_passages]
    return corpus


def main():
    print("Loading MS MARCO corpus...")
    corpus = load_ms_marco_corpus()
    print(f"Loaded {len(corpus)} passages.")

    method = "hnsw"  # 🔁 Change to "pq" to test PQ variant

    print(f"Initializing ANNRetriever with method: {method} ...")
    retriever = ANNRetriever(corpus, method=method)

    query = "What is the capital of France?"
    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query, top_k=10)

    print("Top retrieval results:")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Document ID: {doc.doc_id}, Score: {score:.4f}")
        print(f"Text: {doc.text}\n")


if __name__ == "__main__":
    main()