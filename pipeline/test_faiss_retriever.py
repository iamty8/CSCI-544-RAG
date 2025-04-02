import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.faiss_retriever import FAISSRetriever

def load_ms_marco_corpus(max_passages=None):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ms_marco.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Extract individual passages from the dataset structure.
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
    corpus = load_ms_marco_corpus()
    print(f"Loaded {len(corpus)} passages.")

    print("Initializing FAISSRetriever...")
    retriever = FAISSRetriever(corpus)

    query = "What is the capital of France?"
    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query, top_k=10)

    # print("Top retrieval results:")
    # for i, (passage_text, score) in enumerate(results):
    #     print(f"[{i+1}] Score: {score:.4f}")
    #     print(f"Passage: {passage_text}\n")
    
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Document ID: {doc.doc_id}, Score: {score:.4f}")
        print(f"Text: {doc.text}\n")

if __name__ == "__main__":
    main()
