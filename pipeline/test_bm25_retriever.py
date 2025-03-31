import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.bm25_retriever import BM25Retriever


def load_ms_marco_corpus(max_passages=None):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Extract individual passages
    passages = dataset["passages"]

    # Flatten all passage_text entries (each is a list of paragraphs)
    corpus = []
    for p in passages:
        if isinstance(p["passage_text"], list):
            corpus.extend(p["passage_text"])  # Add each paragraph to corpus
        else:
            corpus.append(p["passage_text"])  # In case it's a single string

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