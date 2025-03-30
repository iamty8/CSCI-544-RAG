from llama_index.indices.keyword_table import GPTKeywordTableIndex
from llama_index.schema import Document

class BM25Retriever:
    def __init__(self, corpus):
        self.docs = [Document(text=text) for text in corpus]
        self.index = GPTKeywordTableIndex.from_documents(self.docs)

    def query(self, query_str, top_k=5):
        response = self.index.query(query_str, similarity_top_k=top_k)
        return response.response if hasattr(response, 'response') else str(response)