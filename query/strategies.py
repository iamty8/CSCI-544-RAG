from query.paraphrase import paraphrase_query, batch_paraphrase_queries
from query.truncation import QueryTruncator
from query.embedding_refinement import EmbeddingRefiner



class QueryRewriter:
    def __init__(self, method="paraphrase", batch_mode:bool=False):
        self.method = method.lower() if method else "no-rewrite"
        self.batch_mode:bool = batch_mode

    def rewrite(self, query: str|list[str]) -> str:
        if not self.batch_mode:
            if self.method == "paraphrase":
                return paraphrase_query(query)
            elif self.method == "truncation":
                # return QueryTruncator.truncate_query(query, max_tokens=20)
                return QueryTruncator.truncate_query_keywords_only(query, max_tokens=10)
            elif self.method == "refine":
                refiner = EmbeddingRefiner()
                return refiner.refine_query(query)
            else:
                self.method = "no-rewrite"
                return query
        else:
            if self.method == "paraphrase":
                return batch_paraphrase_queries(query)
            elif self.method == "truncation":
                # return QueryTruncator.truncate_query(query, max_tokens=20)
                return QueryTruncator.truncate_query_keywords_only(query, max_tokens=10)
            elif self.method == "refine":
                refiner = EmbeddingRefiner()
                return refiner.refine_query(query)
            else:
                self.method = "no-rewrite"
                return query