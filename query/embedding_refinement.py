from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from utils.preprocess import Preprocessor

class EmbeddingRefiner:
    def __init__(self, model_name="sentence-transformers/msmarco-MiniLM-L-6-v3"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def refine_query(self, query: str, top_k: int = 10) -> str:
        """
        使用注意力机制提取对 embedding 贡献最大的 token，构建更紧凑的 query。
        """
        # 原始预处理
        cleaned_query = Preprocessor.preprocess_text_for_dense_methods(query)

        # Tokenize
        inputs = self.tokenizer(cleaned_query, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model(**inputs, output_attentions=True)

        # 获取 attention：shape = (layers, heads, tokens, tokens)
        attentions = torch.stack(outputs.attentions)  # [L, B, H, T, T]
        mean_attn = attentions.mean(dim=(0, 1, 2))     # 平均所有层和头部 → [T, T]
        token_scores = mean_attn.mean(dim=0)           # 每个 token 的被关注程度

        # 选取 top_k token id
        topk_ids = torch.topk(token_scores, top_k).indices
        topk_ids = topk_ids.sort().values  # 保持原始顺序

        refined_tokens = [self.tokenizer.decode([inputs.input_ids[0][i]], skip_special_tokens=True) for i in topk_ids]
        return " ".join(refined_tokens)
