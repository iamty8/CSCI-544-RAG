import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, corpus, model_name="msmarco-MiniLM-L-6-v2"):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Use Inner Product (cosine similarity if normalized)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def query(self, query_str, top_k=5):
        query_embedding = self.model.encode([query_str], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = [self.corpus[i] for i in indices[0]]
        return "\n".join(results)