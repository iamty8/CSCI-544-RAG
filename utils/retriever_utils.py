import os
import json
import argparse

def load_ms_marco_corpus(data_path:str, max_passages:int=None):
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Extract individual passages
    passages = dataset["passages"]

    # Flatten all passage_text entries (each is a list of paragraphs)
    corpus = []
    for p in passages:
        if isinstance(p["passage_text"], list):
            corpus.extend(p["passage_text"])  # Add each paragraph to corpus
        else:
            corpus.append(p["passage_text"])  # In case it's a single string

    if max_passages:
        corpus = corpus[:max_passages]
    return corpus

def parse_kv_args(kv_list: list[str]) -> dict:
    args = {}
    for kv in kv_list:
        if '=' not in kv:
            raise argparse.ArgumentTypeError(f"Invalid format for retriever_args: '{kv}'. Use key=value.")
        key, value = kv.split('=', 1)
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        args[key] = value
    return args
