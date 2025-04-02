from abc import ABC, abstractmethod
from llama_index.core import Document

class RetrieverBase(ABC):
    """
    Base class for all retrievers.
    """

    def __init__(self, **kwargs):
        """
        Initialize the retriever with the given parameters.
        """
        self.params = kwargs

    @abstractmethod
    def retrieve(self, query) -> list[tuple[Document, float]]:
        """
        Retrieve documents based on the query.
        """
        pass