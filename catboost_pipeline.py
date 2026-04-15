import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from iterative_pipeline import load_dataset_df, compute_metrics
from prepare_datasets import prepare_banking77_dataset, prepare_chemprot_dataset, prepare_claude9_dataset, prepare_tarif_dataset


def train_catboost(dataset: str, train_size: int, iters: int = 3) -> tuple[CatBoostClassifier, dict, dict]:
    metrics = []
    for _ in range(iters):
        if dataset == "banking77":
            prepare_banking77_dataset(train_size)
        if dataset == "chemprot":
            prepare_chemprot_dataset(train_size)
        if dataset == "claude9":
            prepare_claude9_dataset(train_size)
        if dataset == "tarif":
            prepare_tarif_dataset(train_size)
        train_df = load_dataset_df(dataset, "dev")
        test_df = load_dataset_df(dataset, "test")
        train_pool = Pool(train_df["text"], label=train_df["label"], text_features=[0])
        test_pool = Pool(test_df["text"], label=test_df["label"], text_features=[0])
        model = CatBoostClassifier(loss_function="MultiClass", iterations=500, verbose=True)
        model.fit(train_pool, verbose=True)
        test_preds = model.predict(test_pool)
        test_metrics = compute_metrics(test_df["label"], test_preds)
        metrics.append(test_metrics)

    result_metrics = {}
    stat = {}
    for i, m in enumerate(metrics):
        for k, v in m.items():
            result_metrics[k + "_iter" + str(i)] = v
            if stat.get(k, None) is None:
                stat[k] = []
            stat[k].append(v)
    for k, v in stat.items():
        result_metrics[k + "_mean"] = np.mean(v)
        result_metrics[k + "_std"] = np.std(v)

    return result_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="catboost")
    parser.add_argument("--train_size", type=int, default=1000)
    args = parser.parse_args()

    train_size = args.train_size

    for train_size in range(5, 10, 5):
        print('TRAIN SIZE: ' + str(train_size))
        test_metrics = train_catboost(args.dataset, train_size)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(train_size, test_metrics)
        (out_dir / ("test_metrics_{}_ts{}.json".format(args.dataset, train_size))).write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
