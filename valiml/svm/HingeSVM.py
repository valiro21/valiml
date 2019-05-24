import numba
import numpy as np
import pandas as pd
from numba import prange
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

from valiml.utils.numba_utils import create_progbar_numba


@numba.jit(nopython=True, parallel=True)
def optimize(X, y, alpha, max_iters):
    t = 0
    w = np.zeros((X.shape[1] + 1,), dtype=np.float32)
    bar = create_progbar_numba(max_iters)
    for n_iter in range(max_iters):
        for idx in range(X.shape[0]):
            t += 1
            lr = 1 / (t * alpha)
            if y[idx] * np.dot(w[1:], X[idx, :]) < 1:
                w[1:] -= lr * (alpha * w[1:] - y[idx] * X[idx])
            else:
                w[1:] -= lr * alpha * w[1:]
        bar.add(1)
    return w


def make_compute_kernel_matrix(kernel):
    @numba.jit(nopython=True, parallel=True)
    def _compute(X, kernel_matrix):
        for idx in range(X.shape[0]):
            kernel_matrix[idx:, idx] = kernel(X[idx:, :], X[idx, :])
            kernel_matrix[idx, idx:] = kernel_matrix[idx:, idx]

        return kernel_matrix
    return _compute


@numba.jit(nopython=True, parallel=True)
def optimize_kernel(X, y, alpha, max_iters, kernel_matrix):
    t = 0
    a = np.zeros((X.shape[0]), dtype=np.float32)
    bar = create_progbar_numba(max_iters)
    for n_iter in range(max_iters):
        for idx in range(X.shape[0]):
            t += 1
            lr = 1 / (t * alpha)

            value = lr * y[idx] * (a * y * kernel_matrix[idx, :]).sum()
            if value < 1:
                a[idx] = a[idx] + 1
        bar.add(1)
    return a


def make_rbf(gamma):
    @numba.jit(nopython=True, parallel=True)
    def rbf(X, w):
        result = np.zeros((X.shape[0],), dtype=np.float32)
        for idx in prange(X.shape[0]):
            result[idx] = -np.linalg.norm(X[idx, :] - w)**2 / (2 * gamma**2)
        return np.exp(result)
    return rbf


def make_apply_kernel(kernel):
    @numba.jit(nopython=True, parallel=True)
    def _kernel(X_train, X_test, a):
        result = np.zeros((X_test.shape[0],), dtype=np.float32)
        for idx in range(X_test.shape[0]):
            result[idx] = (kernel(X_train, X_test[idx]) * a).sum()
        return result
    return _kernel


class HingeSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='pegasos', max_iters=1000, alpha=0.1, kernel='linear', verbose=0, gamma=0.1):
        """

        :param optimizer: The optimizer to use. Choices include lm and gd.
        :param max_iters: Maximum number of iterations for convergence.
        :param alpha: The weight of the squared norm of w.
        :param n_inits: The number of initial starting position to try.
        :param regularizer: The regularizer of the respective optimizer.
                            Step size for gd and l2 regularization for lm.
        """
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.alpha = alpha
        self.kernel = kernel
        self.verbose = verbose
        self.gamma = gamma

    def fit(self, X, y):
        self.coef_ = np.zeros((1, X.shape[1]))
        self.intercept_ = np.zeros((1, 1))

        X = X.astype('float32')
        y = 2 * y - 1
        if self.kernel == 'linear':
            w = optimize(X, y, self.alpha, self.max_iters)

            self.intercept_[0] = w[0]
            self.coef_[:] = w[1:]

            hinge_loss = np.clip(1 - self.decision_function(X) * y, 0, np.inf)
            self.support_vectors_ = X[hinge_loss > 0]
            self.dual_coef_ = hinge_loss[hinge_loss > 0]

        elif self.kernel == 'rbf':
            kernel = make_rbf(self.gamma)
            kernel_matrix = np.zeros((X.shape[0], X.shape[0]), dtype='float32')
            make_compute_kernel_matrix(kernel)(X, kernel_matrix)
            self.a = optimize_kernel(X, y, self.alpha, self.max_iters, kernel_matrix) * y
            self.X_train = X

        return self

    def decision_function(self, X):
        X = X.astype('float32')
        if self.kernel == 'linear':
            result = np.matmul(X, self.coef_[0])
        elif self.kernel == 'rbf':
            kernel = make_rbf(self.gamma)
            result = make_apply_kernel(kernel)(self.X_train, X, self.a)
        return result

    def predict(self, X):
        return np.sign(self.decision_function(X))


@numba.jit(nopython=True)
def test_progb():
    bar = create_progbar_numba(25)
    a = np.random.rand(2000, 10000)
    b = np.random.rand(10000, 2000)
    c = 0
    for idx in range(25):
        c += np.sum(a + b.T)

        bar.add(1)
    return c
