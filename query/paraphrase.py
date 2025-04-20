# query/paraphrasing.py

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from parrot import Parrot
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
        
def batch_paraphrase_queries(
    queries: list[str],
    max_length: int = 64,
    num_beams: int = 5,
    diversity_penalty: float = 0.0,
    temperature: float = 1.0
) -> list[str]:
    """
    Paraphrase a list of queries at once (no internal batching).
    
    Args:
        queries (list of str): Input queries.
    
    Returns:
        list of paraphrased strings.
    """
    tokenizer = parrot.tokenizer
    model = parrot.model
    device = model.device

    model.eval()
    with torch.no_grad():
        encoded_inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        generated_outputs = model.generate(
            **encoded_inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            diversity_penalty=diversity_penalty,
            do_sample=False,
            early_stopping=True
        )

        decoded_texts = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

        paraphrased_outputs = []
        for idx, original_query in enumerate(queries):
            if idx < len(decoded_texts) and decoded_texts[idx].strip():
                paraphrased_outputs.append(decoded_texts[idx])
            else:
                paraphrased_outputs.append(original_query)

    return paraphrased_outputs