from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model.label_model import LabelModel


@dataclass
class ClassInfo:
    name: str
    index: int


class SnorkelTrainer:
    """Train a Snorkel label model for an arbitrary number of classes."""

    def __init__(
        self,
        lf_descriptions: dict[str, str],
        lf_classes: dict[str, str],
        class_names: Iterable[str],
        device: str = 'cpu',
    ):
        self.class_to_index = {name: idx for idx, name in enumerate(class_names)}
        self.abstain = -1

        self.lf_descriptions = lf_descriptions
        self.lf_classes = lf_classes
        self.device = device
        self.lfs = [self._create_labeling_function(field) for field in lf_descriptions.keys()]
        self.applier = PandasLFApplier(self.lfs)
        self.label_model = LabelModel(cardinality=len(self.class_to_index), verbose=True, device=device)
        if 'cuda' in device:
            self.label_model.cuda(device)

    def _create_labeling_function(self, field: str):
        class_name = self.lf_classes[field]
        class_index = self.class_to_index.get(class_name, self.abstain)

        @labeling_function(name=field)
        def lf(x):
            llm_pred = x[field]
            return class_index if llm_pred else self.abstain

        return lf

    def fit(self, train_df: pd.DataFrame) -> None:
        logger.info(f"Training label model on {len(train_df)} samples")
        L_train = self.applier.apply(train_df)
        self.label_model.fit(L_train, n_epochs=1000, log_freq=100, l2=2, lr=1e-3, optimizer="adam")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        L = self.applier.apply(df)
        return self.label_model.predict(L)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        L = self.applier.apply(df)
        return self.label_model.predict_proba(L)

    def get_weak_labels(self, df: pd.DataFrame, threshold: float = 0.5):
        L = self.applier.apply(df)
        pred_probas = self.label_model.predict_proba(L)
        preds = np.argmax(pred_probas, axis=1)
        max_prob = np.max(pred_probas, axis=1)
        mask = max_prob > threshold
        return df[mask], preds[mask]
