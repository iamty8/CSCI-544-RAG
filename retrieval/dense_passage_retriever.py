import faiss
import torch
import numpy as np
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer  # Import DPR models and tokenizers
from llama_index.core import Document
from utils.preprocess import Preprocessor

from retrieval.retriever_base import RetrieverBase

class DensePassageRetriever(RetrieverBase):
    def __init__(self, corpus, batch_size=64, max_length=256):
        super().__init__(corpus)
        # Note: only GPU can be used for DPR models; using CPU will generate an error
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
        # self.documents = [Document(text=doc, doc_id=str(i)) for i, doc in enumerate(corpus)]
        # Preprocess each document text for dense retrieval
        texts = [Preprocessor.preprocess_text_for_dense_methods(doc.text) for doc in self.documents]

        # Batch process the texts to encode passages without overloading GPU memory.
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.p_tokenizer(
                batch_texts,
                max_length=max_length,      # Apply truncation to limit sequence length
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                embeddings_batch = self.p_encoder(**inputs).pooler_output.cpu().numpy()
            all_embeddings.append(embeddings_batch)

        # Concatenate all embedding batches together
        self.embeddings = np.concatenate(all_embeddings, axis=0)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        # Preprocess the query text
        query = Preprocessor.preprocess_text_for_dense_methods(query)
        with torch.no_grad():
            q_inputs = self.q_tokenizer(query, return_tensors='pt').to(self.device)
            q_embedding = self.q_encoder(**q_inputs).pooler_output.cpu().numpy()

        faiss.normalize_L2(q_embedding)
        scores, indices = self.index.search(q_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc = self.documents[idx]
            results.append((doc, float(score)))
        return results  # Return list of (document, score) tuples
