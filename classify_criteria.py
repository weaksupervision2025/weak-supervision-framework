import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from datasets import load_dataset
from tqdm import tqdm

from src.llm_client import LLMQueryClient
from src.classifier import DialogueCriteriaClassifier
import config


def load_samples(dataset_name: str, split: str) -> list[str]:
    """Load dataset texts for classification."""
    logger.info(f"Loading dataset {dataset_name} split {split}")
    dataset = load_dataset(dataset_name, split=split)
    text_col = next((c for c in ["text", "sentence", "utterance"] if c in dataset.column_names), dataset.column_names[0])
    texts = list(dataset[text_col])
    return texts

def load_criteria(path: str) -> dict[str, str]:
    logger.info(f"Loading criteria from {path}")
    criteria = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                criteria[item["criterion"]] = item["description"]
    return criteria

def save_result(handle, obj: dict) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()

def run_parallel_requests(
        texts: list[str], classifier: DialogueCriteriaClassifier, output_path: str, num_workers: int,
        start_idx: int = 0, domain_info: str | None = None) -> None:
    logger.info(f"Classfying {len(texts)} samples with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(output_path, "a", encoding="utf-8") as out_file:
        futures = {
            executor.submit(classifier.classify_text, text, config.CLASSIFY_MODE, domain_info): (idx, text) for idx, text in enumerate(texts, start=start_idx)
            }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, text = futures[future]
            try:
                labels = future.result()
            except Exception as e:  # pragma: no cover - best effort log
                logger.error(f"Error processing sample {idx}: {repr(e)}")
                labels = {label: None for label in classifier.criteria_dict}
            result = {"index": idx, "text": text, "labels": labels}
            save_result(out_file, result)

def classify_criteria() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    texts = load_samples(args.dataset, args.split)
    criteria = load_criteria(args.criteria)
    llm_client = LLMQueryClient(config.API_BASE_URL, config.API_KEY, config.VLLM_BASE_URL, config.VLLM_KEY)
    classifier = DialogueCriteriaClassifier(llm_client, criteria, config.CLASSIFY_PROMPT_FILE, config.LLM_FOR_CLASSIFY)

    start_idx = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx:
            logger.info(f"Resuming from {start_idx} already processed samples")
    texts_to_process = texts[start_idx:]

    if not texts_to_process:
        logger.info("Nothing to process")
        return

    run_parallel_requests(texts_to_process, classifier, args.output, args.num_workers, start_idx=start_idx)


if __name__ == "__main__":
    classify_criteria()
