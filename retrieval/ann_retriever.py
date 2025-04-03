import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from llama_index.core import Document

from utils.preprocess import Preprocessor
from retrieval.retriever_base import RetrieverBase


class ANNRetriever(RetrieverBase):
    def __init__(self, corpus, model_name="sentence-transformers/msmarco-MiniLM-L-6-v3", method="hnsw"):
        """
        Initializes the retriever with FAISS and a sentence embedding model.

        Parameters:
            corpus (list of str): Raw input text documents (passages)
            model_name (str): Sentence transformer model name
            method (str): Type of FAISS index to use ('hnsw' or 'pq')
        """
        super().__init__(corpus)
        self.method = method

        # Determine device for embedding: use cuda if available, else cpu.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for embedding: {device}")

        # Load the embedding model on the selected device.
        self.model = SentenceTransformer(model_name, device=device)

        # Preprocess each passage using dense-friendly cleaning.
        self.cleaned_corpus = [
            Preprocessor.preprocess_text_for_dense_methods(text) for text in corpus
        ]

        # Encode all documents into dense vectors (embeddings).
        self.embeddings = self.model.encode(
            self.cleaned_corpus,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize vectors to unit length (important for cosine similarity).
        faiss.normalize_L2(self.embeddings)
        self.dim = self.embeddings.shape[1]

        # Build the FAISS index based on the chosen ANN method.
        if method == "hnsw":
            # HNSW = Hierarchical Navigable Small World graph (fast and accurate).
            self.index = faiss.IndexHNSWFlat(self.dim, 32)  # 32 = number of neighbors.
        elif method == "pq":
            # PQ = Product Quantization (compact index, lower memory usage).
            nlist = 100  # number of clusters.
            m = 16  # sub-vector size.
            quantizer = faiss.IndexFlatL2(self.dim)  # Used to group vectors.
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8)
            self.index.train(self.embeddings)  # Must train PQ before adding data.
        else:
            raise ValueError("Unsupported method: choose 'hnsw' or 'pq'")

        # Add all document vectors to the index.
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        """
        Retrieves top_k most similar documents to the input query.

        Parameters:
            query (str): The user's input question or keyword phrase.
            top_k (int): Number of top results to return.

        Returns:
            List of (Document, similarity score) tuples.
        """
        query_vec = self.model.encode(
            [Preprocessor.preprocess_text_for_dense_methods(query)],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_vec)

        # Search for nearest neighbors in the FAISS index.
        scores, indices = self.index.search(query_vec, top_k)

        # Collect and return top matching documents with their similarity scores.
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results
