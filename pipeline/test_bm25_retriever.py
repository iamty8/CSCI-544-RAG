import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.bm25_retriever import BM25Retriever


def load_ms_marco_corpus(max_passages=None):
    """
    Loads MS MARCO from ../data/ms_marco.json and returns a list of passages.
    If max_passages is set, only returns that many for testing.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if "passage" in dataset:
        corpus = dataset["passage"]
    elif "passage_text" in dataset:
        corpus = dataset["passage_text"]
    else:
        # Default fallback
        corpus = []
        for key, value in dataset.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                corpus = value
                break

    # Limit to first N passages if max_passages is set
    if max_passages:
        corpus = corpus[:max_passages]
    return corpus


def main():
    print("Loading MS MARCO corpus...")
    # corpus = load_ms_marco_corpus()
    corpus = load_ms_marco_corpus(max_passages=10000)  # ✅ set to 10000 for quick test
    
    print(f"Loaded {len(corpus)} passages.")

    print("Initializing BM25Retriever...")
    retriever = BM25Retriever(corpus)

    query = "What is the capital of France?"
    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query, top_k=5)
    print("Top results:")
    for i, (doc, _) in enumerate(results):
        print(f"[{i+1}] Document ID: {doc.doc_id}")
        print(f"Text: {doc.text}\n")


if __name__ == "__main__":
    main()