from llama_index.core.indices.keyword_table import GPTKeywordTableIndex
from llama_index.core import Document
from llama_index.core import ServiceContext
from llama_index.core.retrievers import KeywordTableSimpleRetriever

from utils.preprocess import Preprocessor
from retrieval.retriever_base import RetrieverBase


class BM25Retriever(RetrieverBase):
    def __init__(self, corpus):
        """
        Initializes BM25Retriever by applying sparse preprocessing and creating a BM25 index.

        Parameters:
          corpus (list of str): Original passage texts (raw from dataset)
        """
        # Apply sparse preprocessing
        self.cleaned_docs = [
            " ".join(Preprocessor.preprocess_text_for_sparse_methods(text))
            for text in corpus
        ]
        # Create LlamaIndex Document objects
        self.documents = [Document(text=doc, doc_id=str(idx)) for idx, doc in enumerate(self.cleaned_docs)]

        # Build BM25 index using GPTKeywordTableIndex
        self.index = GPTKeywordTableIndex.from_documents(
            self.documents,
            service_context=ServiceContext.from_defaults(llm=None, embed_model=None) # disables OpenAI
        )

    def retrieve(self, query, top_k=10):
        """
        Retrieves top_k documents using keyword-based BM25 retrieval.

        Parameters:
          query (str): The query string.
          top_k (int): Number of top documents to retrieve.

        Returns:
          List of tuples (Document, score=None)
        """
        # Preprocess the query
        tokens = Preprocessor.preprocess_text_for_sparse_methods(query)
        preprocessed_query = " ".join(tokens)

        # Retrieve actual documents from BM25 index
        retriever = KeywordTableSimpleRetriever(index=self.index)
        nodes = retriever.retrieve(preprocessed_query)

        # Wrap and return
        results = [
            (Document(text=node.node.text, doc_id=node.node.node_id), node.score)
            for node in nodes[:top_k]
        ]
        return results