import os
import json
import argparse
import logging
import subprocess

from configs.config import LOG_PATH, RCLONE_REMOTE_NAME

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
    if kv_list is None:
        return {}
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

class SingleTimestampFormatter(logging.Formatter):
    def __init__(self, fmt_with_time, fmt_no_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fmt_with_time = fmt_with_time
        self.fmt_no_time = fmt_no_time
        self.first = True

    def format(self, record):
        if self.first:
            self._style._fmt = self.fmt_with_time
            self.first = False
        else:
            self._style._fmt = self.fmt_no_time
        return super().format(record)


def setup_logger(log_path: str = LOG_PATH, verbose: bool = True, logger_name="retriever_eval") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = SingleTimestampFormatter(
        fmt_with_time="%(asctime)s - %(message)s",
        fmt_no_time="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

    return logger

def get_git_commit_info():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=format:%s"]).decode("utf-8").strip()
        return commit_hash, commit_message
    except Exception as e:
        return None, None
    
def sync_logs_to_drive():
    try:
        print("Syncing with Google Drive ...")
        subprocess.run([
            "rclone", "copy", LOG_PATH, f"{RCLONE_REMOTE_NAME}:Experiments/logs/", "--update", "--quiet"
        ], check=True)
        print("Synced.")
    except subprocess.CalledProcessError:
        print("Failed to sync.")