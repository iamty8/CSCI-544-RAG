import retrieval as retrieval
import retrieval.ann_retriever
import retrieval.bm25_retriever
import retrieval.dense_passage_retriever
import retrieval.faiss_retriever
import retrieval.retriever_base

DATA_PATH:str = './data'

RETRIEVERS:dict[str, retrieval.retriever_base.RetrieverBase] = {
    'faiss': retrieval.faiss_retriever.FAISSRetriever,
    'ann': retrieval.ann_retriever.ANNRetriever,
    'bm25': retrieval.bm25_retriever.BM25Retriever,
    'dense': retrieval.dense_passage_retriever.DensePassageRetriever
}

QUERY_METHODS:list[str] = ["truncation", "refine", "paraphrase"]

LOG_PATH:str = './exp/exp.log'
RCLONE_REMOTE_NAME:str = 'csci544'