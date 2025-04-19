from abc import ABC, abstractmethod
from llama_index.core import Document

class RetrieverBase(ABC):
    """
    Base class for all retrievers.
    """

    def __init__(self, corpus:list):
        """
        Initialize the retriever with the given parameters.
        """
        self.corpus:list = corpus
        # Create Document objects for easy retrieval + display.
        self.documents:list[Document] = [Document(text=doc, doc_id=str(idx)) for idx, doc in enumerate(corpus)]
        self.text_to_doc_id:dict[str, str] = {doc.text: doc.doc_id for doc in self.documents}

    @abstractmethod
    def retrieve(self, query) -> list[tuple[Document, float]]:
        """
        Retrieve documents based on the query.
        """
        pass

    @abstractmethod
    def result_processing(
            self, 
            results:list[tuple[Document, float]], 
            query:str, 
            answer:str, 
            passage_texts:str, 
            idx:int
        ) -> tuple[set[str], list[str]]:
        pass