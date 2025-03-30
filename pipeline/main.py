import argparse
import json
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from utils.preprocess import Preprocessor


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_corpus(data, retriever_type):
    passages = []
    for i in range(len(data["query"])):
        passage_list = data["passages"][i]
        for p in passage_list:
            if not p.strip():
                continue
            if retriever_type == "bm25":
                tokens = Preprocessor.preprocess_text_for_sparse_methods(p)
                passages.append(" ".join(tokens))  # BM25: clean token string
            else:
                processed = Preprocessor.preprocess_text_for_dense_methods(p)
                passages.append(processed)         # FAISS: keep natural text
    return list(set(passages))


def select_retriever(method, corpus):
    if method == "bm25":
        return BM25Retriever(corpus)
    elif method == "faiss":
        return FAISSRetriever(corpus)
    else:
        raise ValueError(f"Unsupported retriever type: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/ms_marco.json")
    parser.add_argument("--retriever", type=str, choices=["bm25", "faiss"], default="bm25")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data}")
    data = load_data(args.data)

    print(f"Preprocessing data for: {args.retriever}")
    corpus = extract_corpus(data, args.retriever)

    print(f"Initializing {args.retriever.upper()} retriever...")
    retriever = select_retriever(args.retriever, corpus)

    print("Query interface ready. Type 'exit' to quit.\n")
    while True:
        query = input("Enter query: ")
        if query.strip().lower() == "exit":
            break
        result = retriever.query(query)
        print("\n🔍 Retrieved Result:\n", result)
        print("-" * 60)


if __name__ == "__main__":
    main()