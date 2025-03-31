import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import Document from llama-index
from llama_index.core import Document

from utils.preprocess import Preprocessor


class FAISSRetriever:
    def __init__(self, corpus, model_name="sentence-transformers/msmarco-MiniLM-L-6-v3"):
        """
        Initialize the FAISSRetriever.

        Parameters:
          corpus (list of str): The list of original document texts.
          model_name (str): Name of the SentenceTransformer model to use.
        """
        self.corpus = corpus
        # Preprocess the corpus for dense retrieval (minimal cleaning)
        self.cleaned_corpus = [Preprocessor.preprocess_text_for_dense_methods(text) for text in corpus]

        # Create LlamaIndex Document objects for later retrieval output.
        self.documents = [Document(text=doc, doc_id=str(idx)) for idx, doc in enumerate(corpus)]

        # Initialize the SentenceTransformer model with the correct model identifier.
        self.model = SentenceTransformer(model_name)
        # Encode the cleaned corpus
        self.embeddings = self.model.encode(self.cleaned_corpus, show_progress_bar=True, convert_to_numpy=True)

        # Normalize embeddings for cosine similarity (using inner product on L2-normalized vectors)
        faiss.normalize_L2(self.embeddings)

        # Build a FAISS index (using Inner Product metric)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top_k documents given a query.

        Parameters:
          query (str): The query string.
          top_k (int): Number of top documents to retrieve.

        Returns:
          List of tuples (Document, score)
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

        # Retrieve corresponding documents and return along with scores
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results
