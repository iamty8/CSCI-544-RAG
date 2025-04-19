from llama_index.core.indices.keyword_table import GPTKeywordTableIndex
from llama_index.core import Document
from llama_index.core import ServiceContext
from llama_index.core.retrievers import KeywordTableSimpleRetriever

from utils.preprocess import Preprocessor
from retrieval.retriever_base import RetrieverBase
from llama_index.core import Settings

from rapidfuzz import fuzz


class BM25Retriever(RetrieverBase):
    def __init__(self, corpus, fuzzy_threshold=40):
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

        self.text_to_doc_id = {doc.text: doc.doc_id for doc in self.documents}

        Settings.llm = None
        Settings.embed_model = None

        # Build BM25 index using GPTKeywordTableIndex
        self.index = GPTKeywordTableIndex.from_documents(
            self.documents
            # service_context=ServiceContext.from_defaults(llm=None, embed_model=None) # disables OpenAI
        )

        self.fuzzy_threshold = fuzzy_threshold

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
    
    def result_processing(
            self, 
            results:list[tuple[Document, float]], 
            query:str, 
            answer:str, 
            passage_texts:str, 
            idx:int
        ) -> tuple[set[str], list[str]]:

        retrieved_texts = [
            doc.text for doc, _ in results
            if hasattr(doc, "text") and isinstance(doc.text, str) and doc.text.strip()
        ]
        if len(retrieved_texts) == 0:
            retrieved_texts = None

        # ✅ fuzzy match
        relevant_ids = set()
        for doc, _ in results:
            if hasattr(doc, "text") and doc.text:
                sim = fuzz.token_sort_ratio(answer, doc.text)
                if sim >= self.fuzzy_threshold or answer.lower() in doc.text.lower():
                    relevant_ids.add(doc.doc_id)

        if not relevant_ids:
            skipped_norelevant += 1
            # ✅ optional: 打印前几个失败例子以 debug
            if idx < 3:
                print(f"[❌ No fuzzy match] Q: {query}")
                print(f"A: {answer}")
                print("Top retrieved texts:")
                for doc, _ in results[:3]:
                    print(f"- {doc.text[:200]}...\n")
            relevant_ids = None
        
        return retrieved_texts, relevant_ids