import faiss
import torch
import numpy as np
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer  # Import DPR models and tokenizers
from llama_index.core import Document
from utils.preprocess import Preprocessor

class DensePassageRetriever:
    def __init__(self, corpus):
        # Note: only GPU can be used for DPR models, CPU will generate error
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load pre-trained DPR question encoder model
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(self.device)
        # Load pre-trained DPR context encoder model
        self.p_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(self.device)

        # Load tokenizers for question and passage encoding
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.p_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        # Convert corpus to Document objects with unique IDs
        self.documents = [Document(text=doc, doc_id=str(i)) for i, doc in enumerate(corpus)]
        # Preprocess each document text for dense retrieval
        texts = [Preprocessor.preprocess_text_for_dense_methods(doc.text) for doc in self.documents]
        
        # Encode all passages in the corpus
        with torch.no_grad():  # Disable gradient calculation for inference
            # Tokenize and pad all texts, move to device
            inputs = self.p_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            # Generate embeddings for all passages
            self.embeddings = self.p_encoder(**inputs).pooler_output.cpu().numpy()

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        # Get embedding dimension
        dim = self.embeddings.shape[1]
        # Create FAISS index for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        # Add embeddings to the index
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        # Preprocess the query text
        query = Preprocessor.preprocess_text_for_dense_methods(query)
        with torch.no_grad():  # Disable gradient calculation for inference
            # Tokenize the query and move to device
            q_inputs = self.q_tokenizer(query, return_tensors='pt').to(self.device)
            # Generate embedding for the query
            q_embedding = self.q_encoder(**q_inputs).pooler_output.cpu().numpy()

        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(q_embedding)
        # Search for top-k most similar passages
        scores, indices = self.index.search(q_embedding, top_k)

        # Prepare results with documents and similarity scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc = self.documents[idx]
            results.append((doc, float(score)))
        return results  # Return list of (document, score) tuples