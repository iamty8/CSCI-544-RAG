import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Import Document from the correct location.
from llama_index.core import Document

from utils.preprocess import Preprocessor

class FAISSRetriever:
    def __init__(self, corpus, model_name="sentence-transformers/msmarco-MiniLM-L-6-v3"):
        """
        Initialize the FAISSRetriever.

        Parameters:
          corpus (list of str): The list of original passage texts.
          model_name (str): Name of the SentenceTransformer model to use.
        """
        self.corpus = corpus
        # Preprocess the corpus for dense retrieval (minimal cleaning)
        self.cleaned_corpus = [Preprocessor.preprocess_text_for_dense_methods(text) for text in corpus]

        # Create Document objects for later retrieval output.
        self.documents = [Document(text=doc, doc_id=str(idx)) for idx, doc in enumerate(corpus)]

        # Determine device: cuda if available, else cpu.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Initialize the SentenceTransformer model with the appropriate device.
        self.model = SentenceTransformer(model_name, device=device)
        # Encode the cleaned corpus with batching (you can adjust batch_size if needed)
        self.embeddings = self.model.encode(self.cleaned_corpus, show_progress_bar=True, convert_to_numpy=True, batch_size=64)

        # Normalize embeddings for cosine similarity (using inner product on L2-normalized vectors)
        faiss.normalize_L2(self.embeddings)

        # Build a FAISS index (using Inner Product metric) on CPU
        dim = self.embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(self.embeddings)

        # Check if GPUs are available; if so, move the index to GPU. (not working)
        # num_gpus = faiss.get_num_gpus()
        # if num_gpus > 0:
        #     print("Using FAISS GPU with", num_gpus, "GPUs available.")
        #     self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        # else:
        #     print("No GPU found. Using CPU index.")
        #     self.index = cpu_index

    def retrieve(self, query, top_k=10):
        """
        Retrieve top_k passages given a query.

        Parameters:
          query (str): The query string.
          top_k (int): Number of top passages to retrieve.

        Returns:
          List of tuples (passage_text, score)
        """
        # Preprocess the query using the dense method (minimal cleaning)
        cleaned_query = Preprocessor.preprocess_text_for_dense_methods(query)
        # Encode the query
        query_vector = self.model.encode(cleaned_query, convert_to_numpy=True)
        # Normalize the query vector
        faiss.normalize_L2(np.expand_dims(query_vector, axis=0))
        # Search the FAISS index
        scores, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        scores = scores.flatten()
        indices = indices.flatten()

        # Retrieve corresponding passages and return with scores.
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(self.documents):
                results.append((self.documents[idx].text, float(score)))
        return results
