import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.dense_passage_retriever import DensePassageRetriever

def load_ms_marco_corpus(max_passages=None):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    passages = dataset["passages"]
    corpus = []
    for p in passages:
        if isinstance(p["passage_text"], list):
            corpus.extend(p["passage_text"])
        else:
            corpus.append(p["passage_text"])

    return corpus[:max_passages]

def main():
    print("Loading MS MARCO corpus...")
    corpus = load_ms_marco_corpus()
    print(f"Loaded {len(corpus)} passages.")

    print("Initializing DPRRetriever...")
    retriever = DensePassageRetriever(corpus)

    query = "What is the capital of France?"
    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query, top_k=10)

    print("Top retrieval results:")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Document ID: {doc.doc_id}, Score: {score:.4f}")
        print(f"Text: {doc.text}\n")

if __name__ == "__main__":
    main()