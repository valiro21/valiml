import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array, check_X_y
from valiml.boosting import DecisionStump
from sklearn.model_selection import train_test_split


# TODO: Think of a way to use inheritance for multiple modes
from valiml.utils import create_progbar, normalize


def make_prediction_sammer(estimator, X, n_labels):
    estimator_proba = estimator.predict_proba(X)
    np.clip(estimator_proba, np.finfo(estimator_proba.dtype).eps, None, out=estimator_proba)
    log_proba = np.log(estimator_proba)

    estimator_decision = (log_proba - (1. / n_labels) * log_proba.sum(axis=1)[:, np.newaxis])
    estimator_decision *= n_labels - 1

    estimator_prediction = estimator_decision.argmax(axis=1)

    return estimator_proba, estimator_decision, estimator_prediction


def one_hot_encode_samme(labels, n_labels):
    one_hot_encoded_labels = np.zeros((labels.shape[0], n_labels))
    one_hot_encoded_labels.fill(-1 / (n_labels - 1))

    for idx in range(labels.shape[0]):
        one_hot_encoded_labels[idx, labels[idx]] = 1
    return one_hot_encoded_labels


def get_train_and_validation(x_train, y_train, validation_split):
    if isinstance(validation_split, tuple):
        x_test, y_test = validation_split
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=validation_split,
                                                            shuffle=True)
    return x_train, x_test, y_train, y_test


class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=200, algorithm='SAMME',
                 verbose=False, learning_rate=1.0, valid_boost=0):
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.base_estimator = base_estimator
        self.valid_boost = valid_boost
        self.initialize()

    def _get_alpha(self, weighted_error):
        if self.algorithm == 'SAMME.R':
            return 1
        elif self.algorithm == 'SAMME':
            return np.log((1 - weighted_error) / weighted_error) + np.log(self.n_labels - 1)

    def initialize(self):
        self.validation_errors = None
        self.n_labels = -1
        self.losses = []
        self.errors = []
        self.weighted_errors = []
        self.estimators = []
        
    def _compute_loss(self, classifier_decision, y):
        return np.exp(-classifier_decision * y / self.n_labels).sum() / y.shape[0]
    
    def fit(self, X, y, validation_split=None):
        """
        :param x_train: features
        :param y_train: labels
        """

        x_train, y_train = check_X_y(X, y, multi_output=False, y_numeric=True, estimator='AdaBoost')

        check_array(x_train, accept_sparse=False, dtype='numeric', force_all_finite=True, ensure_2d=True,
                    estimator='AdaBoost')
        check_array(y_train, accept_sparse=False, dtype='numeric', force_all_finite=True, ensure_2d=False,
                    estimator='AdaBoost')

        self.initialize()

        if self.verbose:
            metrics = ['w_err', 'loss', 'err', 'acc']
            if validation_split is not None:
                metrics.append('val_acc')
            _bar = create_progbar(self.n_estimators, stateful_metrics=metrics)

        self.n_labels = len(np.unique(y_train))
        one_hot_encoded_labels = one_hot_encode_samme(y_train, self.n_labels)
        sample_weight = np.ones(x_train.shape[0]) / x_train.shape[0]
        classifier_decision = np.zeros((x_train.shape[0], self.n_labels))
        if validation_split is not None:
            x_train, x_test, y_train, y_test = get_train_and_validation(x_train, y_train, validation_split)
            self.validation_errors = []
            validation_decision = np.zeros((x_test.shape[0], self.n_labels))

        n_iter = 1
        retry_count = 0
        random_chance_error = (self.n_labels - 1) / self.n_labels
        used_samples = x_train.shape[0]

        if self.valid_boost:
            used_samples = int(x_train.shape[0] / 2)
            sample_weight[:used_samples] *= 2

        while n_iter <= self.n_estimators:
            if self.valid_boost and n_iter <= self.valid_boost * self.n_estimators:
                max_iter = self.n_estimators * self.valid_boost
                t = int(x_train.shape[0] / 2 + int((np.log(n_iter) / np.log(max_iter)) * x_train.shape[0] / 2))

                if t - used_samples > 0:
                    sample_weight[:used_samples] *= used_samples / t
                    sample_weight[used_samples:t] *= (t - used_samples) / t

                used_samples = t

            estimator = clone(self.base_estimator).fit(x_train[:used_samples], y_train[:used_samples],
                                                       sample_weight=sample_weight[:used_samples])
            if self.algorithm == 'SAMME':
                prediction_train = estimator.predict(x_train)
                decision_train = one_hot_encode_samme(prediction_train, self.n_labels)

                prediction_test = estimator.predict(x_test)
                decision_test = one_hot_encode_samme(prediction_test, self.n_labels)
            elif self.algorithm == 'SAMME.R':
                proba_train, decision_train, prediction_train = make_prediction_sammer(estimator, x_train, self.n_labels)
                _, decision_test, _ = make_prediction_sammer(estimator, x_test, self.n_labels)

            errors = prediction_train[:used_samples] != y_train[:used_samples]
            weighted_error = sample_weight[:used_samples][errors].sum()

            if weighted_error == 0:
                self.estimators.append((1.0, estimator))
                if self.verbose:
                    _bar.update(self.n_estimators, "|".join(metrics), [0, 0, 0, 1])
                break
            elif weighted_error >= random_chance_error:
                retry_count += 1
                if retry_count == 10:
                    break
                continue

            self.weighted_errors.append(weighted_error)

            alpha = self.learning_rate * self._get_alpha(weighted_error)
            assert alpha >= 0

            self.estimators.append((alpha, estimator))

            classifier_decision += alpha * decision_train
            classifier_error = 1 - accuracy_score(y_train, classifier_decision.argmax(axis=1))
            self.errors.append(classifier_error)
            if validation_split is not None:
                validation_decision += alpha * decision_test
                validation_error = 1 - accuracy_score(y_test, validation_decision.argmax(axis=1))
                self.validation_errors.append(validation_error)

            if self.algorithm == 'SAMME':
                sample_weight[:used_samples] *= np.exp(errors * alpha)
            elif self.algorithm == 'SAMME.R':
                result = one_hot_encoded_labels[:used_samples] * np.log(proba_train[:used_samples])
                result *= -(self.n_labels - 1) / self.n_labels
                sample_weight[:used_samples] *= np.exp(result.sum(axis=1))
            sample_weight[:used_samples] = normalize(sample_weight[:used_samples])

            self.losses.append(self._compute_loss(classifier_decision, one_hot_encoded_labels[:used_samples]))

            if self.verbose:
                values = {
                    'w_err': weighted_error,
                    'loss': self.losses[-1],
                    'err': classifier_error,
                    'acc': 1 - classifier_error
                }
                if validation_split is not None:
                    values['val_acc'] = 1 - validation_error

                _bar.update(
                    n_iter,
                    **values
                )

            n_iter += 1

        if self.verbose:
            _bar.update()
            _bar.finish(dirty=True)

        return self

    def decision_function(self, X):
        prediction = np.zeros((X.shape[0], self.n_labels))

        for alpha, estimator in self.estimators:
            if self.algorithm == 'SAMME':
                decision = one_hot_encode_samme(estimator.predict(X), self.n_labels)
                prediction += alpha * decision
            elif self.algorithm == 'SAMME.R':
                _, decision, _ = make_prediction_sammer(estimator, X, self.n_labels)
                prediction += alpha * decision

        return prediction

    def predict(self, X):
        decision = self.decision_function(X)
        return decision.argmax(axis=1)

    def predict_proba(self, X):
        prediction = self.decision_function(X)
        prediction /= self.n_labels
        prediction = np.exp((1. / (self.n_labels - 1)) * prediction)

        normalizer = prediction.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        prediction /= normalizer
        return prediction


if __name__ == '__main__':
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    b = AdaBoost(base_estimator=DecisionStump(n_random_features=5, reg_entropy=0), n_estimators=400, verbose=True, algorithm='SAMME.R')
    b.fit(x_train, y_train, validation_split=(x_test, y_test))
