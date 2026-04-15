import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def compute_metrics_bert(eval_pred, average="macro"):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average=average),
        "precision": precision_score(labels, preds, average=average),
        "recall": recall_score(labels, preds, average=average),
    }

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=512):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return item


def train_bert(train_texts, train_labels, test_texts,
               valid_texts=None, valid_labels=None,
               model_name=None, model=None, tokenizer=None, label_encoder=None,
               epochs=20, max_len=512):

    inter = list(set(valid_labels).intersection(set(train_labels)))
    valid_labels = [x if x in inter else 1 for x in valid_labels]

    if not label_encoder:
        label_encoder = LabelEncoder()
        if valid_labels is not None:
            label_encoder = label_encoder.fit(train_labels)
            train_label_ids = label_encoder.transform(train_labels)
            valid_label_ids = label_encoder.transform(valid_labels)
        else:
            train_label_ids = label_encoder.fit_transform(train_labels)
    else:
        train_label_ids = label_encoder.transform(train_labels)
        if valid_labels is not None:
            valid_label_ids = label_encoder.transform(valid_labels)
    num_labels = len(set(train_label_ids))

    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    test_dataset = TextDataset(
        texts=test_texts,
        labels=None,
        tokenizer=tokenizer,
        max_len=max_len
    )

    if model is None:
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    training_args = TrainingArguments(
        output_dir="./bert_cls_out",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
    )

    if valid_texts is None:
        train_texts_split, val_texts_split, train_label_ids_split, val_label_ids_split = train_test_split(
            train_texts,
            train_label_ids,
            test_size=0.1,
            random_state=42,
            # stratify=train_label_ids
        )
    else:
        train_texts_split, val_texts_split, train_label_ids_split, val_label_ids_split = train_texts, valid_texts, train_label_ids, valid_label_ids

    train_dataset = TextDataset(
        texts=train_texts_split,
        labels=train_label_ids_split,
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_dataset = TextDataset(
        texts=val_texts_split,
        labels=val_label_ids_split,
        tokenizer=tokenizer,
        max_len=max_len
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_bert,
    )

    trainer.train()

    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    pred_ids = np.argmax(logits, axis=-1)
    pred_labels = label_encoder.inverse_transform(pred_ids)

    return model, tokenizer, label_encoder, pred_labels
