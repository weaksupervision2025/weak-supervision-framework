import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Union


class MajorityVoteTrainer:
    def __init__(self, strategy: str = 'hard', handle_ties: str = 'random', random_state: int = None, classes: list = None):
        self.strategy = strategy
        self.handle_ties = handle_ties
        self.random_state = random_state

        self.classes_ = classes
        self.n_classes_ = len(self.classes_) if self.classes_ else None
        self.prior_probs_ = None
        self.annotator_weights_ = None

    def fit(self, df: pd.DataFrame, annotator_weights: Optional[Dict[str, float]] = None) -> 'MajorityVoteTrainer':
        df = df.fillna(-1).astype(int)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if not self.classes_:
            all_labels = np.unique(df.values)
            self.classes_ = all_labels[all_labels != -1]
            self.n_classes_ = len(self.classes_)

        self._compute_prior_probs(df)

        self.annotator_weights_ = annotator_weights or {col: 1.0 for col in df.columns}

        for annotator in df.columns:
            if annotator not in self.annotator_weights_:
                self.annotator_weights_[annotator] = 1.0

        return self

    def _compute_prior_probs(self, df: pd.DataFrame) -> None:
        all_valid_labels = []
        for col in df.columns:
            valid_labels = df[col][df[col] != -1]
            all_valid_labels.extend(valid_labels.tolist())

        if not all_valid_labels:
            self.prior_probs_ = np.ones(self.n_classes_) / self.n_classes_
        else:
            counter = Counter(all_valid_labels)
            total = sum(counter.values())
            self.prior_probs_ = np.array([counter.get(cls, 0) for cls in self.classes_]) / total

    def _count_votes(self, row: pd.Series) -> Dict[int, float]:
        vote_counts = {cls: 0.0 for cls in self.classes_}

        for annotator, label in row.items():
            if label != -1 and annotator in self.annotator_weights_:
                weight = self.annotator_weights_[annotator]
                vote_counts[label] += weight

        return vote_counts

    def _resolve_tie(self, max_classes: List[int]) -> int:
        if self.handle_ties == 'random':
            return np.random.choice(max_classes)
        elif self.handle_ties == 'first':
            return max_classes[0]
        elif self.handle_ties == 'prior':
            prior_probs = [self.prior_probs_[np.where(np.array(self.classes_) == cls)[0][0]] for cls in max_classes]
            return max_classes[np.argmax(prior_probs)]
        else:
            raise ValueError(f"Unknown: {self.handle_ties}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = df.fillna(-1).astype(int)
        n_objects = len(df)
        probabilities = np.zeros((n_objects, self.n_classes_))

        for i, (idx, row) in enumerate(df.iterrows()):
            vote_counts = self._count_votes(row)
            total_votes = sum(vote_counts.values())

            if total_votes == 0:
                probabilities[i] = self.prior_probs_
            else:
                for j, cls in enumerate(self.classes_):
                    probabilities[i, j] = vote_counts[cls] / total_votes

        return probabilities

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.fillna(-1).astype(int)
        if self.strategy == 'hard':
            return self._predict_hard(df)
        elif self.strategy == 'soft':
            return self._predict_soft(df)
        else:
            raise ValueError(f"Unknown: {self.strategy}")

    def _predict_hard(self, df: pd.DataFrame) -> np.ndarray:
        predictions = []

        for i, (idx, row) in enumerate(df.iterrows()):
            vote_counts = self._count_votes(row)

            max_count = max(vote_counts.values())
            max_classes = [cls for cls, count in vote_counts.items() if count == max_count]

            if len(max_classes) == 1:
                predictions.append(max_classes[0])
            else:
                predictions.append(self._resolve_tie(max_classes))

        return np.array(predictions)

    def _predict_soft(self, df: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(df)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]

    def get_annotator_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = []

        for annotator in df.columns:
            annotator_data = df[annotator]
            valid_annotations = annotator_data[annotator_data != -1]
            n_valid = len(valid_annotations)
            n_total = len(annotator_data)
            missing_rate = (n_total - n_valid) / n_total if n_total > 0 else 0

            class_dist = {}
            for cls in self.classes_:
                count = (valid_annotations == cls).sum()
                class_dist[f'class_{cls}_count'] = count
                class_dist[f'class_{cls}_prop'] = count / n_valid if n_valid > 0 else 0

            stats.append({
                'annotator': annotator,
                'n_annotations': n_total,
                'n_valid': n_valid,
                'missing_rate': missing_rate,
                'weight': self.annotator_weights_.get(annotator, 1.0),
                **class_dist
            })

        return pd.DataFrame(stats)

    def get_consensus_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        n_objects = len(df)
        total_agreement = 0
        total_votes = 0
        objects_with_consensus = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            vote_counts = self._count_votes(row)
            total_votes_for_object = sum(vote_counts.values())

            if total_votes_for_object > 0:
                total_votes += total_votes_for_object
                max_count = max(vote_counts.values())

                if len([c for c in vote_counts.values() if c > 0]) == 1:
                    total_agreement += total_votes_for_object

                if max_count > total_votes_for_object / 2:
                    objects_with_consensus += 1

        metrics = {
            'total_objects': n_objects,
            'objects_with_consensus': objects_with_consensus,
            'consensus_rate': objects_with_consensus / n_objects if n_objects > 0 else 0,
            'total_agreement_rate': total_agreement / total_votes if total_votes > 0 else 0
        }

        return metrics
