import numba
import numpy as np
from numba import prange
from sklearn.base import BaseEstimator, ClassifierMixin


@numba.jit(nopython=True, cache=True)
def normalize(v):
    return v / v.sum()


@numba.jit(nopython=True, parallel=False, cache=True)
def fit_stump_samme(X, Y, feature_ids, n_labels, sample_weight=None):
    n_features = len(feature_ids)

    weighted_errors = np.zeros((n_features,), dtype=np.float64)
    split_values = np.zeros((n_features,), dtype=np.float64)
    left_labels = np.zeros((n_features,), dtype=np.int32)
    right_labels = np.zeros((n_features,), dtype=np.int32)

    for fidx in prange(n_features):
        feature_idx = feature_ids[fidx]

        sorted_data_mask = X[:, feature_idx].argsort()
        x = X[:, feature_idx][sorted_data_mask]
        y = Y[sorted_data_mask]
        sorted_sample_weights = sample_weight[sorted_data_mask]

        sample_sum_left = np.zeros(n_labels)
        sample_sum_right = np.zeros(n_labels)
        for idx in range(y.shape[0]):
            sample_sum_right[y[idx]] += sorted_sample_weights[idx]

        total_weighted_sum = sample_sum_right.sum()

        best_split_value = x[0]
        best_left_label = np.random.choice(n_labels)
        best_right_label = sample_sum_right.argmax()
        best_weighted_accuracy = sample_sum_right[best_right_label]

        for idx in range(y.shape[0]):
            if idx > 0 and x[idx - 1] != x[idx]:
                left_label = sample_sum_left.argmax()
                right_label = sample_sum_right.argmax()
                weighted_accuracy = sample_sum_left[left_label] + sample_sum_right[right_label]

                if weighted_accuracy > best_weighted_accuracy:
                    best_split_value = x[idx - 1] + (x[idx] - x[idx - 1]) / 2
                    best_left_label = left_label
                    best_right_label = right_label
                    best_weighted_accuracy = weighted_accuracy

            sample_sum_left[y[idx]] += sorted_sample_weights[idx]
            sample_sum_right[y[idx]] -= sorted_sample_weights[idx]

        weighted_errors[fidx] = total_weighted_sum - best_weighted_accuracy
        split_values[fidx] = best_split_value
        left_labels[fidx] = best_left_label
        right_labels[fidx] = best_right_label

    best_idx = weighted_errors.argmin()

    return (weighted_errors[best_idx], feature_ids[best_idx],
            split_values[best_idx],
            left_labels[best_idx], right_labels[best_idx])


@numba.jit(nopython=True, parallel=True, cache=True)
def fit_stump_sammer(X, Y, feature_ids, n_labels, sample_weight=None):
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
        sorted_sample_weights = sample_weight[sorted_data_mask]

        sample_sum_left = np.zeros(n_labels)
        sample_sum_right = np.zeros(n_labels)
        for idx in range(0, y.shape[0]):
            sample_sum_right[y[idx]] += sorted_sample_weights[idx]
        total_weighted_sum = sample_sum_right.sum()

        best_split_value = x[0]
        best_left_probabilities = normalize(np.random.random((n_labels,)))
        best_right_probabilities = normalize(sample_sum_right)
        best_weighted_accuracy = sample_sum_right[sample_sum_right.argmax()]

        for idx in range(0, y.shape[0]):
            if idx > 0 and x[idx - 1] != x[idx]:
                if x[idx] != best_split_value:
                    left_label = sample_sum_left.argmax()
                    right_label = sample_sum_right.argmax()
                    weighted_accuracy = sample_sum_left[left_label] + sample_sum_right[right_label]

                    if weighted_accuracy > best_weighted_accuracy:
                        best_split_value = x[idx - 1] + (x[idx] - x[idx - 1]) / 2
                        best_left_probabilities = normalize(sample_sum_left)
                        if sample_sum_right.sum() == 0:
                            best_right_probabilities = normalize(np.random.random((n_labels,)))
                        else:
                            best_right_probabilities = normalize(sample_sum_right)
                        best_weighted_accuracy = weighted_accuracy

            sample_sum_left[y[idx]] += sorted_sample_weights[idx]
            sample_sum_right[y[idx]] -= sorted_sample_weights[idx]

        weighted_errors[fidx] = total_weighted_sum - best_weighted_accuracy
        split_values[fidx] = best_split_value
        left_probabilities[fidx] = best_left_probabilities
        right_probabilities[fidx] = best_right_probabilities

    best_idx = weighted_errors.argmin()
    return (weighted_errors[best_idx], feature_ids[best_idx],
            split_values[best_idx],
            left_probabilities[best_idx], right_probabilities[best_idx])


class DecisionStumpSamme(BaseEstimator, ClassifierMixin):
    def __init__(self, n_random_features=None):
        self.feature_idx = -1
        self.left_label = 0
        self.right_label = 0
        self.split_value = 0
        self.n_labels = 0
        self.classes_ = 0
        self.n_random_features = n_random_features

    def fit(self, X, y, sample_weights=None):
        self.classes_ = np.unique(y)
        self.n_labels = len(self.classes_)

        if self.n_random_features is not None:
            features_subset = np.random.choice(X.shape[1], self.n_random_features, replace=False)
        else:
            features_subset = list(range(X.shape[1]))

        error, self.feature_idx, self.split_value, self.left_label, self.right_label = fit_stump_samme(
            X, y, features_subset, self.n_labels, sample_weight=sample_weights
        )

        return self

    def decision_function(self, X):
        prediction = np.zeros((X.shape[0], self.n_labels))
        prediction.fill(-1 / (self.n_labels - 1))
        mask = X[:, self.feature_idx] < self.split_value

        prediction[mask, self.left_label] = 1
        prediction[~mask, self.right_label] = 1
        return prediction

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)

    def __lt__(self, other):
        return self.split_value < other.split_value


class DecisionStumpSammeR(BaseEstimator, ClassifierMixin):
    def __init__(self, n_random_features=None):
        self.feature_idx = -1
        self.left_probabilities = None
        self.right_probabilities = None
        self.split_value = 0
        self.n_labels = 0
        self.n_random_features = n_random_features
        self.classes_ = 0

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.n_labels = len(self.classes_)

        if self.n_random_features is not None:
            features_subset = np.random.choice(X.shape[1], self.n_random_features, replace=False)
        else:
            features_subset = list(range(X.shape[1]))

        error, self.feature_idx, self.split_value, self.left_probabilities, self.right_probabilities = fit_stump_sammer(
            X, y, features_subset, self.n_labels, sample_weight=sample_weight
        )

        return self

    def predict_proba(self, X):
        probability = np.zeros((X.shape[0], self.n_labels))
        mask = X[:, self.feature_idx] < self.split_value

        probability[mask, :] = self.left_probabilities
        probability[~mask, :] = self.right_probabilities

        np.clip(probability, np.finfo(probability.dtype).eps, None, out=probability)

        return probability

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def decision_function(self, X):
        probabilities = self.predict_proba(X)

        np.clip(probabilities, np.finfo(probabilities.dtype).eps, None, out=probabilities)
        log_proba = np.log(probabilities)

        return (self.n_labels - 1) * (log_proba - (1. / self.n_labels) * log_proba.sum(axis=1)[:, np.newaxis])

    def __lt__(self, other):
        return self.split_value < other.split_value
