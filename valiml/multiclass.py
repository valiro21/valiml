from copy import deepcopy
import numpy as np
from scipy.stats import mode
from itertools import combinations

from sklearn.base import BaseEstimator, ClassifierMixin, clone


def one_vs_rest_split(X, Y):
    classes = np.unique(Y)
    for class_val in classes:
        yield class_val, X, (Y == class_val).astype('int')


def one_vs_one_split(X, Y):
    classes = np.unique(Y)
    for class_vals in combinations(classes, 2):
        mask = np.isin(Y, class_vals)
        yield class_vals, X[mask], 2 * (Y[mask] == class_vals[0]) - 1


class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, verbose=0):
        self.classifier = classifier
        self.verbose = verbose
    
    def fit(self, X, Y, *args, **kwargs):
        self.classifiers = {}
        self.classes = np.unique(Y)
        for class_val, x, y in one_vs_rest_split(X, Y):
            classifier = clone(self.classifier)
            classifier.fit(x, y, *args, **kwargs)
            self.classifiers[class_val] = classifier
            if self.verbose:
                print(f'Trained classifier for class {class_val}')
        return self
    
    def predict(self, x):
        predictions = np.zeros((x.shape[0], len(self.classifiers)))
        for idx, class_vals in enumerate(self.classifiers):
            classifier = self.classifiers[class_vals]
            predictions[:, idx] = classifier.decision_function(x)
        
        return np.vectorize(lambda x: self.classes[x])(predictions.argmax(axis=1)).reshape(-1)

    
class OneVsOneClassifier():
    def __init__(self, classifier):
        self.classifier = classifier
    
    def fit(self, X, Y, *args, **kwargs):
        self.classifiers = {}
        self.classes = np.unique(Y)
        for class_vals, x, y in one_vs_one_split(X, Y):
            classifier = deepcopy(self.classifier)
            classifier.fit(x, y, *args, **kwargs)
            self.classifiers[class_vals] = classifier
        return self
    
    def predict(self, x):
        predictions = np.zeros((x.shape[0], len(self.classifiers)))
        for idx, class_vals in enumerate(self.classifiers):
            classifier = self.classifiers[class_vals]
            prediction = classifier.predict(x)
            predictions[:, idx] = np.vectorize(lambda value: class_vals[0 if value > 0 else 1])(prediction)
        
        return mode(predictions, axis=1).mode.reshape(-1)