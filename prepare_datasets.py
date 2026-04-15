import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
from collections import defaultdict, deque
import random


def custom_train_test_split(df, target_column, train_size=100):
    class_indices = defaultdict(list)
    for idx, row in df.iterrows():
        class_label = row[target_column]
        if hasattr(class_label, 'item'):
            class_label = class_label.item()
        elif pd.isna(class_label):
            class_label = 'NaN'
        class_indices[class_label].append(idx)
    for class_label in class_indices:
        random.shuffle(class_indices[class_label])
    class_queues = {class_label: deque(indices) for class_label, indices in class_indices.items()}
    train_indices = []
    classes = list(class_queues.keys())
    while len(train_indices) < train_size and classes:
        random.shuffle(classes)
        added_in_cycle = False
        for class_label in classes[:]:
            if (len(train_indices) < train_size and
                    class_queues[class_label]):
                idx = class_queues[class_label].popleft()
                train_indices.append(idx)
                if len(train_indices) >= train_size:
                    break
                added_in_cycle = True
                if not class_queues[class_label]:
                    classes.remove(class_label)
        if len(train_indices) >= train_size:
            break
        if not added_in_cycle:
            break
    test_indices = []
    for class_label, queue in class_queues.items():
        test_indices.extend(queue)
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def load_dataset_df_for_prepare(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    records = []
    for idx, entry in raw.items():
        text = entry.get("data", {}).get("text")
        label = entry.get("label")
        weak_labels = entry.get("weak_labels", None)
        records.append({"text": text, "label": label, "weak_labels": weak_labels})
    df = pd.DataFrame(records)
    return df


def prepare_chemprot_dataset(train_size: int) -> None:
    train_df = load_dataset_df_for_prepare('data/chemprot/raw/train.json')
    valid_df = load_dataset_df_for_prepare('data/chemprot/raw/valid.json')
    test_df = load_dataset_df_for_prepare('data/chemprot/raw/test.json')
    train_df = pd.concat([train_df, valid_df])
    train_df.reset_index(inplace=True)
    dev_df, train_df = custom_train_test_split(train_df, "label", train_size=train_size)
    for (fname, df) in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        data = {}
        for i, row in enumerate(df.values.tolist()):
            data[str(i)] = {
                "label": row[1] if fname == "test" else row[2],
                "data": {"text": row[0] if fname == "test" else row[1]},
            }
        assert (len(set([x["label"] for k, x in data.items()])) == 10)
        json.dump(data, open('data/chemprot/source/{}.json'.format(fname), 'w'))


def prepare_banking77_dataset(train_size: int) -> None:
    train_df = load_dataset_df_for_prepare('data/banking77/raw/train.json')
    valid_df = load_dataset_df_for_prepare('data/banking77/raw/valid.json')
    test_df = load_dataset_df_for_prepare('data/banking77/raw/test.json')
    train_df = pd.concat([train_df, valid_df])
    train_df.reset_index(inplace=True)
    dev_df, train_df = custom_train_test_split(train_df, "label", train_size=train_size)
    for (fname, df) in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        data = {}
        for i, row in enumerate(df.values.tolist()):
            data[str(i)] = {
                "label": row[1] if fname == "test" else row[2],
                "data": {"text": row[0] if fname == "test" else row[1]},
            }
        assert (len(set([x["label"] for k, x in data.items()])) == 77)
        # json.dump(data, open('data/banking77/source/{}.json'.format(fname), 'w'))
    # for (fname, df) in [('train', train_df), ('dev', valid_df), ('test', test_df)]:
    #     data = {}
    #     for i, row in enumerate(df.values.tolist()):
    #         data[str(i)] = {
    #             "label": row[1],
    #             "data": {"text": row[0]},
    #         }
    #     assert (len(set([x["label"] for k, x in data.items()])) == 77)
    #     json.dump(data, open('data/banking77/source/{}.json'.format(fname), 'w'))


def prepare_claude9_dataset(train_size: int) -> None:
    train_df = load_dataset_df_for_prepare('data/claude9/raw/train.json')
    valid_df = load_dataset_df_for_prepare('data/claude9/raw/valid.json')
    test_df = load_dataset_df_for_prepare('data/claude9/raw/test.json')
    train_df = pd.concat([train_df, valid_df])
    train_df.reset_index(inplace=True)
    dev_df, train_df = custom_train_test_split(train_df, "label", train_size=train_size)
    for (fname, df) in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        data = {}
        for i, row in enumerate(df.values.tolist()):
            data[str(i)] = {
                "label": row[1] if fname == "test" else row[2],
                "data": {"text": row[0] if fname == "test" else row[1]},
            }
        assert (len(set([x["label"] for k, x in data.items()])) == 9)
        json.dump(data, open('data/claude9/source/{}.json'.format(fname), 'w'))


def prepare_tarif_dataset(train_size: int) -> None:
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {e}")
                        continue
        return data
    def read_tarif_dataset(path):
        data = read_jsonl(path)
        records = []
        for i, item in enumerate(data):
            try:
                text = item.get("masked_formatted_dialogue")
                label = int(item.get("Тариф", 0))
                records.append({"text": text, "label": label, "weak_labels": None})
            except:
                continue
        df = pd.DataFrame(records)
        return df
    train_df = read_tarif_dataset('data/tarif/raw/train.jsonl')
    valid_df = read_tarif_dataset('data/tarif/raw/dev.jsonl')
    test_df = read_tarif_dataset('data/tarif/raw/test.jsonl')
    dev_df, train2_df = custom_train_test_split(valid_df, "label", train_size=train_size)
    for (fname, df) in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        data = {}
        for i, row in enumerate(df.values.tolist()):
            data[str(i)] = {
                "label": row[1],
                "data": {"text": row[0]},
            }
        assert (len(set([x["label"] for k, x in data.items()])) == 2)
        json.dump(data, open('data/tarif/source/{}.json'.format(fname), 'w'))
