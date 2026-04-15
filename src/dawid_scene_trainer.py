import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
import numba
from typing import Dict, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


@numba.jit(nopython=True, fastmath=True)
def sparse_softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


@numba.jit(nopython=True, parallel=True)
def batched_dot_product(A, B, batch_size=1000):
    n, k = A.shape
    _, m = B.shape
    result = np.zeros((n, m))

    for i in numba.prange(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_A = A[i:end_i]
        result[i:end_i] = batch_A @ B

    return result


class DawidSkeneTrainer:
    def __init__(self,
                 max_iter: int = 30,
                 tol: float = 1e-4,
                 random_state: int = None,
                 prior_strength: float = 50.0,
                 confusion_smoothing: float = 20.0,
                 min_class_support: float = 0.1,
                 batch_size: int = 1000,
                 use_numba: bool = True,
                 sparse_threshold: float = 0.01,
                 diversity_penalty: float = 0.1,
                 gradient_clip: float = 5.0,
                 classes: list = None):

        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.prior_strength = prior_strength
        self.confusion_smoothing = confusion_smoothing
        self.min_class_support = min_class_support
        self.batch_size = batch_size
        self.use_numba = use_numba
        self.sparse_threshold = sparse_threshold
        self.diversity_penalty = diversity_penalty
        self.gradient_clip = gradient_clip

        self.prior_probs_ = None
        self.confusion_matrices_ = None
        self.classes_ = classes
        self.n_classes_ = len(self.classes_) if self.classes_ else None
        self.class_to_idx_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _create_sparse_structures(self, df: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        n_objects, n_annotators = df.shape

        data, rows, cols = [], [], []
        label_indices = []

        for i in range(n_objects):
            for j in range(n_annotators):
                label = df.iloc[i, j]
                if label != -1:
                    label_idx = self.class_to_idx_.get(label, 0)
                    data.append(1.0)
                    rows.append(i)
                    cols.append(j * self.n_classes_ + label_idx)
                    label_indices.append((i, j, label_idx))

        annotation_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_objects, n_annotators * self.n_classes_)
        )

        presence_data, presence_rows, presence_cols = [], [], []
        for i in range(n_objects):
            for j in range(n_annotators):
                if df.iloc[i, j] != -1:
                    presence_data.append(1.0)
                    presence_rows.append(i)
                    presence_cols.append(j)

        presence_matrix = sparse.csr_matrix(
            (presence_data, (presence_rows, presence_cols)),
            shape=(n_objects, n_annotators)
        )

        return annotation_matrix, presence_matrix

    def _vectorized_e_step(self, annotation_matrix: sparse.csr_matrix,
                           presence_matrix: sparse.csr_matrix) -> np.ndarray:
        n_objects = annotation_matrix.shape[0]
        posterior_probs = np.zeros((n_objects, self.n_classes_))

        for start_idx in range(0, n_objects, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_objects)
            batch_size = end_idx - start_idx

            batch_annotations = annotation_matrix[start_idx:end_idx].toarray()
            batch_annotations = batch_annotations.reshape(batch_size, len(self.df_columns_), self.n_classes_)

            batch_presence = presence_matrix[start_idx:end_idx].toarray()

            log_prior = np.log(self.prior_probs_ + 1e-15)
            batch_log_posterior = np.tile(log_prior, (batch_size, 1))

            for j, annotator in enumerate(self.df_columns_):
                cm = self.confusion_matrices_[annotator]
                presence_mask = batch_presence[:, j:j + 1]

                if presence_mask.sum() > 0:
                    annotations_j = batch_annotations[:, j, :]

                    log_likelihood = np.log(cm.T @ annotations_j.T + 1e-15).T

                    batch_log_posterior += log_likelihood * presence_mask

            for i in range(batch_size):
                if self.use_numba:
                    posterior_probs[start_idx + i] = sparse_softmax(batch_log_posterior[i])
                else:
                    posterior_probs[start_idx + i] = self._stable_softmax(batch_log_posterior[i])

            if self.diversity_penalty > 0:
                batch_probs = posterior_probs[start_idx:end_idx]
                entropy = -np.sum(batch_probs * np.log(batch_probs + 1e-15), axis=1)
                diversity_loss = self.diversity_penalty * (1 - entropy / np.log(self.n_classes_))
                batch_log_posterior -= diversity_loss.reshape(-1, 1)

                for i in range(batch_size):
                    posterior_probs[start_idx + i] = sparse_softmax(batch_log_posterior[i])

        return posterior_probs

    def _vectorized_m_step(self, df: pd.DataFrame, posterior_probs: np.ndarray,
                           annotation_matrix: sparse.csr_matrix, presence_matrix: sparse.csr_matrix) -> None:
        n_objects = len(df)
        n_annotators = len(df.columns)

        class_weights = posterior_probs.sum(axis=0)
        min_weight = self.min_class_support * n_objects
        class_weights = np.maximum(class_weights, min_weight)

        self.prior_probs_ = (class_weights + self.prior_strength) / \
                            (n_objects + self.prior_strength * self.n_classes_)

        posterior_expanded = np.repeat(posterior_probs[:, :, np.newaxis], self.n_classes_, axis=2)

        for j, annotator in enumerate(self.df_columns_):
            presence_j = presence_matrix[:, j].toarray().ravel()
            present_indices = np.where(presence_j > 0)[0]

            if len(present_indices) == 0:
                self.confusion_matrices_[annotator] = self._create_global_confusion_matrix()
                continue

            start_col = j * self.n_classes_
            end_col = (j + 1) * self.n_classes_
            annotations_j = annotation_matrix[:, start_col:end_col].toarray()

            # cm[true_class, pred_class] = sum_over_objects posterior[object, true_class] * annotation[object, pred_class]
            weight_matrix = posterior_expanded[present_indices]
            annotation_matrix_j = annotations_j[present_indices]

            cm_update = np.zeros((self.n_classes_, self.n_classes_))
            for true_idx in range(self.n_classes_):
                weights_true = weight_matrix[:, true_idx, true_idx]
                cm_update[true_idx] = weights_true @ annotation_matrix_j

            row_sums = cm_update.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_update = cm_update / row_sums

            current_cm = self.confusion_matrices_[annotator]
            global_cm = self._create_global_confusion_matrix()

            data_weight = min(0.8, len(present_indices) / (100 * self.n_classes_))
            smooth_weight = 0.1

            new_cm = (data_weight * cm_update +
                      (1 - data_weight - smooth_weight) * current_cm +
                      smooth_weight * global_cm)

            new_cm = (new_cm + self.confusion_smoothing) / (1 + self.confusion_smoothing * self.n_classes_)
            new_cm = new_cm / new_cm.sum(axis=1, keepdims=True)

            new_cm = np.clip(new_cm, 1e-6, 1 - 1e-6)
            new_cm = new_cm / new_cm.sum(axis=1, keepdims=True)

            self.confusion_matrices_[annotator] = new_cm

    def _create_global_confusion_matrix(self) -> np.ndarray:
        base_accuracy = 0.7

        cm = np.ones((self.n_classes_, self.n_classes_)) * (1 - base_accuracy) / (self.n_classes_ - 1)
        np.fill_diagonal(cm, base_accuracy)

        return cm

    def _initialize_parameters_vectorized(self, df: pd.DataFrame) -> None:
        if not self.classes_:
            all_labels = np.unique(df.values)
            self.classes_ = all_labels[all_labels != -1]
            self.n_classes_ = len(self.classes_)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.df_columns_ = df.columns.tolist()

        all_valid_labels = []
        for col in df.columns:
            valid_labels = df[col][df[col] != -1]
            all_valid_labels.extend(valid_labels.tolist())

        if all_valid_labels:
            label_counts = np.zeros(self.n_classes_)
            for label in all_valid_labels:
                label_counts[self.class_to_idx_[label]] += 1

            total = len(all_valid_labels)
            self.prior_probs_ = (label_counts + self.prior_strength) / (total + self.prior_strength * self.n_classes_)
        else:
            self.prior_probs_ = np.ones(self.n_classes_) / self.n_classes_

        self.prior_probs_ = np.maximum(self.prior_probs_, self.min_class_support / self.n_classes_)
        self.prior_probs_ = self.prior_probs_ / self.prior_probs_.sum()

        self.confusion_matrices_ = {}
        global_cm = self._create_global_confusion_matrix()

        for annotator in df.columns:
            annotator_data = df[annotator][df[annotator] != -1]
            n_valid = len(annotator_data)

            if n_valid < 5:
                cm = global_cm.copy()
            else:
                label_counts = np.zeros(self.n_classes_)
                for label in annotator_data:
                    label_counts[self.class_to_idx_[label]] += 1

                expected_counts = self.prior_probs_ * n_valid
                agreement = 1.0 - np.abs(label_counts - expected_counts).sum() / (2 * n_valid)
                accuracy = np.clip(agreement, 0.3, 0.9)

                cm = np.ones((self.n_classes_, self.n_classes_)) * (1 - accuracy) / (self.n_classes_ - 1)
                np.fill_diagonal(cm, accuracy)

            cm = (cm + self.confusion_smoothing / 10) / (1 + self.confusion_smoothing / 10 * self.n_classes_)
            cm = cm / cm.sum(axis=1, keepdims=True)

            self.confusion_matrices_[annotator] = cm

    def _stable_softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-15)

    def fit(self, df: pd.DataFrame) -> 'DawidSkeneTrainer':
        df = df.fillna(-1).astype(int)

        self._initialize_parameters_vectorized(df)

        annotation_matrix, presence_matrix = self._create_sparse_structures(df)

        prev_log_likelihood = -np.inf
        best_diversity = 0

        for iteration in tqdm(range(self.max_iter), total=self.max_iter):
            posterior_probs = self._vectorized_e_step(annotation_matrix, presence_matrix)

            self._vectorized_m_step(df, posterior_probs, annotation_matrix, presence_matrix)

            current_log_likelihood = np.sum(np.log(np.max(posterior_probs, axis=1) + 1e-10))
            diversity = len(np.unique(np.argmax(posterior_probs, axis=1)))

            if diversity > best_diversity:
                best_diversity = diversity

            if iteration > 5:
                improvement = current_log_likelihood - prev_log_likelihood

                if abs(improvement) < self.tol and diversity >= self.n_classes_ * 0.1:
                    print(f"Done on iteration {iteration + 1}")
                    break

            prev_log_likelihood = current_log_likelihood

            if (iteration + 1) % 5 == 0:
                print(f"ITER {iteration + 1}: loglike={current_log_likelihood:.2f}, "
                      f"diversity={diversity}/{self.n_classes_}")

        print(f"DONE. Best diversity: {best_diversity}/{self.n_classes_}")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = df.fillna(-1).astype(int)
        self.df_columns_ = df.columns.tolist()
        annotation_matrix, presence_matrix = self._create_sparse_structures(df)
        return self._vectorized_e_step(annotation_matrix, presence_matrix)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.fillna(-1).astype(int)
        probabilities = self.predict_proba(df)
        predicted_indices = np.argmax(probabilities, axis=1)

        unique, counts = np.unique(predicted_indices, return_counts=True)

        return np.array(self.classes_)[predicted_indices]

    def get_annotator_quality(self) -> pd.DataFrame:
        stats = []
        for annotator, cm in self.confusion_matrices_.items():
            accuracy = np.diag(cm).mean()
            entropy = -np.sum(cm * np.log(cm + 1e-15)) / (self.n_classes_ * self.n_classes_)
            stats.append({
                'annotator': annotator,
                'accuracy': accuracy,
                'entropy': entropy,
                'n_diagonal_strong': (np.diag(cm) > 0.5).sum()
            })

        return pd.DataFrame(stats).sort_values('accuracy', ascending=False)
