import numba
import numpy as np
from numba import prange
from sklearn.base import BaseEstimator, ClassifierMixin

from valiml.utils.numba_utils import entropy


@numba.jit(nopython=True, cache=True)
def normalize(v):
    return v / v.sum()


@numba.jit(nopython=True, parallel=True, cache=True)
def fit_stump(X, Y, feature_ids, n_labels, reg_entropy=0, sample_weight=None):
    n_features = len(feature_ids)

    weighted_errors = np.zeros((n_features,), dtype=np.float64)
    split_values = np.zeros((n_features,), dtype=np.float64)
    left_probabilities = np.zeros((n_features, n_labels), dtype=np.float64)
    right_probabilities = np.zeros((n_features, n_labels), dtype=np.float64)

    for fidx in prange(n_features):
        feature_idx = feature_ids[fidx]

        sorted_data_mask = X[:, feature_idx].argsort()
        x = X[:, feature_idx][sorted_data_mask]
        y = Y[sorted_data_mask]
        sorted_sample_weight = sample_weight[sorted_data_mask]

        sample_sum_left = np.zeros(n_labels)
        sample_sum_right = np.zeros(n_labels)
        for idx in range(0, y.shape[0]):
            sample_sum_right[y[idx]] += sorted_sample_weight[idx]

        total_weighted_sum = sample_sum_right.sum()
        if reg_entropy == 0:
            prob_left = normalize(np.random.random((n_labels,)))
        else:
            prob_left = np.zeros((n_labels,), dtype=np.float64)
            prob_left[np.random.choice(n_labels, size=1)] = 1.

        prob_right = normalize(sample_sum_right)
        weighted_error = total_weighted_sum - sample_sum_right.max()
        weighted_error += entropy(prob_right)

        best_stump = (weighted_error, prob_left, prob_right, x[0])

        for idx in range(1, y.shape[0]):
            sample_sum_left[y[idx]] += sorted_sample_weight[idx]
            sample_sum_right[y[idx]] -= sorted_sample_weight[idx]

            if x[idx - 1] != x[idx]:
                prob_left = normalize(sample_sum_left)
                prob_right = normalize(sample_sum_right)

                left_label = prob_left.argmax()
                right_label = prob_right.argmax()

                weighted_error = total_weighted_sum - sample_sum_left[left_label] - sample_sum_right[right_label]

                weighted_error += reg_entropy * (
                    entropy(prob_left) * sample_sum_left.sum() / total_weighted_sum +
                    entropy(prob_right) * sample_sum_right.sum() / total_weighted_sum
                )

                if weighted_error < best_stump[0]:
                    best_split_value = x[idx - 1] + (x[idx] - x[idx - 1]) / 2
                    best_stump = (weighted_error, prob_left, prob_right, best_split_value)

        weighted_errors[fidx], left_probabilities[fidx], right_probabilities[fidx], split_values[fidx] = best_stump

    best_idx = weighted_errors.argmin()
    return (weighted_errors[best_idx], feature_ids[best_idx],
            split_values[best_idx],
            left_probabilities[best_idx], right_probabilities[best_idx])


class DecisionStump(BaseEstimator, ClassifierMixin):
    def __init__(self, n_random_features=None, reg_entropy=0):
        self.feature_idx = -1
        self.left_probabilities = None
        self.right_probabilities = None
        self.split_value = 0
        self.n_labels = 0
        self.n_random_features = n_random_features
        self.classes_ = 0
        self.reg_entropy = reg_entropy
        self.left_label = 0
        self.right_label = 0

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.n_labels = len(self.classes_)

        if self.n_random_features is not None:
            n_random_features = min(self.n_random_features, X.shape[1])
            features_subset = np.random.choice(X.shape[1], n_random_features, replace=False)
        else:
            features_subset = list(range(X.shape[1]))

        if sample_weight is None:
            sample_weight = np.zeros(X.shape[0])
            sample_weight.fill(1 / X.shape[0])

        error, self.feature_idx, self.split_value, self.left_probabilities, self.right_probabilities = fit_stump(
            X, y, features_subset, self.n_labels, reg_entropy=self.reg_entropy, sample_weight=sample_weight
        )

        self.left_label = self.left_probabilities.argmax()
        self.right_label = self.right_probabilities.argmax()

        return self

    def predict_proba(self, X):
        probability = np.zeros((X.shape[0], self.n_labels))
        mask = X[:, self.feature_idx] < self.split_value

        probability[mask, :] = self.left_probabilities
        probability[~mask, :] = self.right_probabilities

        np.clip(probability, np.finfo(probability.dtype).eps, None, out=probability)

        return probability

    def predict(self, X):
        prediction = np.zeros(X.shape[0], dtype=int)
        mask = X[:, self.feature_idx] < self.split_value

        prediction[mask] = self.left_label
        prediction[~mask] = self.right_label
        return prediction

    def __lt__(self, other):
        return self.split_value < other.split_value
