from query.paraphrase import paraphrase_query
# from query.truncation import truncate_query
# from query.embedding_refinement import refine_query



class QueryRewriter:
    def __init__(self, method="paraphrase"):
        self.method = method.lower()

    def rewrite(self, query: str) -> str:
        if self.method == "paraphrase":
            return paraphrase_query(query)
        # elif self.method == "truncate":
            # return truncate_query(query, max_tokens=10)
        # elif self.method == "refine":
            # return refine_query(query)
        else:
            return query  # 默认不改写