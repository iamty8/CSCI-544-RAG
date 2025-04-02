import re
from utils.preprocess import Preprocessor

class QueryTruncator:
    @staticmethod
    def truncate_query(query: str, max_tokens: int = 20) -> str:
        """
        简单地将查询按空格分词，保留最多 max_tokens 个关键词。
        """
        tokens = query.strip().split()
        truncated = tokens[:max_tokens]
        return " ".join(truncated)

    @staticmethod
    def truncate_query_keywords_only(query: str, max_tokens: int = 10) -> str:
        """
        只保留关键词（去除停用词和标点），再截断前 max_tokens 个。
        """
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = Preprocessor.remove_stopwords(words)
        return " ".join(keywords[:max_tokens])
