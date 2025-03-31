import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.faiss_retriever import FAISSRetriever


def load_ms_marco_corpus():
    """
    Loads the entire MS MARCO dataset from ../data/ms_marco.json and extracts a list of passages.
    It attempts to use the "passage" field if available, otherwise "passage_text".
    """
    # Construct path to ms_marco.json relative to the current file (/pipeline/test_faiss_retriever.py)
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "ms_marco.json")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # The dataset is stored as a dictionary with keys mapping to lists.
    if "passage" in dataset:
        corpus = dataset["passage"]
    elif "passage_text" in dataset:
        corpus = dataset["passage_text"]
    else:
        # Fallback: Look for the first key that is a list of strings
        corpus = None
        for key, value in dataset.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                corpus = value
                break
        if corpus is None:
            raise ValueError("No suitable text field found in the dataset.")

    return corpus


def main():
    # Load the entire MS MARCO corpus from ../data/ms_marco.json
    print("Loading MS MARCO corpus from ../data/ms_marco.json...")
    corpus = load_ms_marco_corpus()
    print(f"Loaded {len(corpus)} passages.")

    # Initialize the FAISSRetriever with the loaded corpus.
    print("Initializing FAISSRetriever...")
    retriever = FAISSRetriever(corpus)

    # Sample query for testing.
    query = "What is the capital of France?"

    # Retrieve top 10 documents for the query.
    print(f"\nQuery: {query}\n")
    results = retriever.retrieve(query, top_k=10)

    # Print the retrieved results with document IDs and scores.
    print("Top retrieval results:")
    for doc, score in results:
        print(f"Document ID: {doc.doc_id}, Score: {score:.4f}")
        print(f"Text: {doc.text}\n")


if __name__ == "__main__":
    main()
