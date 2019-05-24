import numba
import numpy as np
from numba import prange
from sklearn.base import BaseEstimator, ClassifierMixin


@numba.jit(nopython=True, cache=True, parallel=True)
def fit_stump(X, Y, feature_ids, sample_weights=None):
    n_features = len(feature_ids)

    weighted_errors = np.zeros((n_features,), dtype=np.float64)
    split_values = np.zeros((n_features,), dtype=np.float64)
    labels = np.zeros((n_features,), dtype=np.int32)

    for fidx in prange(n_features):
        feature_idx = feature_ids[fidx]

        sorted_data_mask = X[:, feature_idx].argsort()
        x = X[:, feature_idx][sorted_data_mask]
        y = Y[sorted_data_mask]
        sample_distribution = np.cumsum((2 * y - 1) * sample_weights[sorted_data_mask])

        best_weighted_error = 0.5 - np.abs(sample_distribution[-1]) / 2
        split_value = x[0]
        label = 1 if sample_distribution[-1] < 0 else -1

        for idx in range(1, y.shape[0]):
            if x[idx] != x[idx - 1]:
                margin = sample_distribution[-1] - 2 * sample_distribution[idx - 1]
                weighted_error = 0.5 - np.abs(margin) / 2
                if weighted_error < best_weighted_error:
                    split_value = x[idx - 1] + (x[idx] - x[idx - 1]) / 2
                    label = 1 if margin < 0 else -1
                    best_weighted_error = weighted_error

        weighted_errors[fidx] = best_weighted_error
        split_values[fidx] = split_value
        labels[fidx] = label

    best_idx = weighted_errors.argmin()
    return weighted_errors[best_idx], feature_ids[best_idx], split_values[best_idx], labels[best_idx]


class DecisionStump(BaseEstimator, ClassifierMixin):
    def __init__(self, n_random_features=None):
        self.error = 1
        self.feature_idx = -1
        self.split_value = 0
        self.label = 0
        self.n_random_features = n_random_features

    def fit(self, X, y, sample_weight=None):
        if self.n_random_features is not None:
            features_subset = np.random.choice(X.shape[1], self.n_random_features, replace=False)
        else:
            features_subset = list(range(X.shape[1]))

        error, self.feature_idx, self.split_value, self.label = fit_stump(X, y, features_subset,
                                                                          sample_weights=sample_weight)
        return self

    def decision_function(self, X):
        return np.sign(2 * self.predict(X) - 1)

    def predict(self, X):
        if self.label == 1:
            return (X[:, self.feature_idx] < self.split_value).astype('int')
        return (X[:, self.feature_idx] >= self.split_value).astype('int')

    def __lt__(self, other):
        return self.split_value < other.split_value
