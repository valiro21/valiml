from valiml.optimizers import lm
import numpy as np
from functools import partial

from valiml.utils.numba_utils import create_progbar_numba


class LSTSQ():
    def __init__(self, l2_alpha=None, constraints=None):
        self.l2_alpha = l2_alpha
        self.constraints = constraints
    
    def fit(self, X, Y):
        if self.l2_alpha is not None:
            l2_x = np.sqrt(self.l2_alpha) * np.identity(X.shape[1])
            l2_y = np.zeros((X.shape[1]))
            X = np.concatenate([X, l2_x])
            Y = np.concatenate([Y, l2_y])
                    
        if self.constraints is None:
            self.x = np.linalg.lstsq(X, Y, rcond=None)[0]
        else:
            zero_matrix = np.zeros((self.constraints[0].shape[0], self.constraints[0].shape[0]))
            Y = np.hstack([2 * np.matmul(X.T, Y), self.constraints[1]])
            X = np.block([
                [np.matmul(X.T, X), self.constraints[0].T],
                [self.constraints[0], zero_matrix]
            ])
            
            self.x = np.linalg.solve(X, Y)[:-self.constraints[1].shape[0]]
        
        return self
    
    def predict(self, X):
        return np.matmul(X, self.x)

    
class SigmaLSTSQ():
    def __init__(self, n_iters=1):
        self.n_iters = n_iters
    
    def predict(self, X):
        k = np.matmul(X, self.x)
        return k
    
    def _callback(self, X, Y, n_iter, x, y, alpha):
        self.x = x
        error = np.sum(np.sign(self.predict(X)) != Y) / Y.shape[0]

        metrics = "|".join(['alpha', 'loss', 'error', 'acc'])
        values = [alpha, y, error, 1 - error]
        self._bar.update(n_iter, metrics, values)
    
    def _sigma_derivative(self, x):
        s = np.exp(2 * x)
        return (4 * s) / (1 + s)**2

    def mse(self, X, Y, x, jacobian=True):
        z = np.matmul(X, x)
        y = np.tanh(z)
        loss = np.array([np.linalg.norm(y - Y)])**2

        if jacobian:
            derivative = np.matmul(2 * (y - Y) * np.vectorize(self._sigma_derivative)(z), X)
            return loss, derivative
        return loss

    def _fit(self, X, Y, l2_alpha=1e3, alpha=1e10, n_iters=20, eps=1e-6, verbose=True):        
        self._bar = create_progbar_numba(target=n_iters, stateful_metrics="|".join(['alpha', 'loss', 'error']))
        
        self.x = np.random.normal(scale=1/np.sqrt(X.shape[1]), size=X.shape[1])
        
        self.x = lm(self.x,
                    partial(self.mse, X, Y),
                    alpha=alpha,
                    max_iters=n_iters,
                    eps=eps,
                    callback=partial(self._callback, X, Y)
                    )
            
        return self
    
    def fit(self, X, Y, **kwargs):
        best_error = float('inf')
        best_x = None
        for n_iter in range(self.n_iters):
            self._fit(X, Y, **kwargs)
            
            error = (np.sign(self.predict(X)) != Y).sum() / Y.shape[0]
            if error < best_error:
                best_x = self.x
                best_error = error
        self.x = best_x
        print(f'Selecting best classifier with error {best_error}')
        return self
