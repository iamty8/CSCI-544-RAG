# query/paraphrasing.py

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from parrot import Parrot
# import torch
import warnings
warnings.filterwarnings("ignore")

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

def paraphrase_query(query: str) -> str:
        para_phrases = parrot.augment(
            input_phrase=query,
            diversity_ranker="levenshtein",
            do_diverse=True,
            max_return_phrases=5,
            max_length=64,
            adequacy_threshold=0.85,
            fluency_threshold=0.80
        )
        if para_phrases:
            # print(para_phrases)
            return para_phrases[0][0]  # 取第一个候选文本
        else:
            return query