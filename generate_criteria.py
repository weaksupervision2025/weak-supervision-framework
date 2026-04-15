import argparse
import json
from loguru import logger
from datasets import load_dataset

from src.criteria_generator import CriteriaGenerator
from src.llm_client import LLMQueryClient
import config


def load_samples(
    dataset_name: str,
    num_samples: int,
    split: str = "test",
    create_empty: bool = False,
) -> tuple[list[str], list[str]]:
    logger.debug(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    text_col = next(
        (c for c in ["text", "sentence", "utterance"] if c in dataset.column_names),
        dataset.column_names[0],
    )
    label_col = next(
        (c for c in ["label", "labels", "intent"] if c in dataset.column_names),
        dataset.column_names[-1],
    )

    if create_empty:
        logger.debug("Returning empty dataset")
        return [], []

    texts = dataset[text_col][:num_samples]
    raw_labels = dataset[label_col][:num_samples]
    label_feature = dataset.features.get(label_col)

    if hasattr(label_feature, "int2str"):
        labels = [label_feature.int2str(v) for v in raw_labels]
    else:
        labels = [str(v) for v in raw_labels]

    return texts, labels


def read_criteria(path: str) -> list[dict[str, str]]:
    logger.debug(f"Reading existing criteria from {path}")
    criteria = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            criteria.append(json.loads(line))
    return criteria


def generate_criteria(dataset: str, output: str, existing: str | None = None, samples: int = 100) -> None:
    logger.info(
        f"Generating criteria for dataset '{dataset}' using {samples} samples"
    )

    texts, labels = load_samples(dataset, samples)
    logger.debug(f"Loaded {len(texts)} texts")

    existing_list = read_criteria(existing) if existing else []
    if existing_list:
        logger.info(f"Loaded {len(existing_list)} existing criteria from {existing}")

    llm_client = LLMQueryClient(config.API_BASE_URL, config.API_KEY, config.VLLM_BASE_URL, config.VLLM_KEY)

    generator = CriteriaGenerator(
        llm_client,
        config.GENERATION_PROMPT_FILE,
        config.DEDUPLICATION_PROMPT_FILE,
        config.LLM_FOR_GENERATION
    )

    new_criteria = generator.get_new_criteria(
        dataset,
        texts,
        labels,
        existing_criteria={c["criterion"]: c["description"] for c in existing_list}
        if existing_list
        else None,
    )

    logger.info(f"Generated {len(new_criteria)} new criteria")

    if existing_list:
        final = generator.deduplicate_new_criteria(existing_list, new_criteria)
        logger.info(f"After deduplication {len(final)} criteria remain")
    else:
        final = new_criteria

    logger.info(f"Writing {len(final)} criteria to {output}")

    with open(output, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing", default=None)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    generate_criteria(
        dataset=args.dataset,
        output=args.output,
        existing=args.existing,
        samples=args.samples
    )
