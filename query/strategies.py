from query.paraphrase import paraphrase_query
from query.truncation import QueryTruncator
from query.embedding_refinement import EmbeddingRefiner



class QueryRewriter:
    def __init__(self, method="paraphrase"):
        self.method = method.lower()

    def rewrite(self, query: str) -> str:
        if self.method == "paraphrase":
            return paraphrase_query(query)
        elif self.method == "truncation":
            # return QueryTruncator.truncate_query(query, max_tokens=20)
            return QueryTruncator.truncate_query_keywords_only(query, max_tokens=10)
        elif self.method == "refine":
            refiner = EmbeddingRefiner()
            return refiner.refine_query(query)
        else:
            return query  # 默认不改写