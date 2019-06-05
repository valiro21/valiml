import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y

from valiml.utils.numba_utils import create_progbar_numba

from valiml.boosting.MulticlassDecisionStump import DecisionStumpSamme, DecisionStumpSammeR
from valiml.optimizers import lm
from sklearn.model_selection import train_test_split


# TODO: Think of a way to use inheritance for multiple modes
from valiml.utils.utils import normalize


class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, n_random_features=None, mode='exponential', type='discrete',
                 verbose=False, learning_rate=1.0, entropy_reg=0.1, valid_boost=0):
        self.n_estimators = n_estimators
        self.mode = mode
        self.n_random_features = n_random_features
        self.type = type
        self.verbose = verbose
        self.learninig_rate = learning_rate
        self.entropy_reg = entropy_reg
        self.valid_boost = valid_boost

        if self.mode == 'exponential':
            self._Z = 1
            
    def _get_weak_classifier(self, instance_importance, x, y):
        if self.n_labels > 2 and self.mode == 'exponential':
            if self.type == 'discrete':
                return DecisionStumpSamme(n_random_features=self.n_random_features, alpha=self.entropy_reg)\
                    .fit(x, y, sample_weight=instance_importance)
            else:
                return DecisionStumpSammeR(n_random_features=self.n_random_features, alpha=self.entropy_reg)\
                    .fit(x, y, sample_weight=instance_importance)
        elif self.n_labels > 2 and self.mode == 'quadratic':
            return DecisionStumpSamme(n_random_features=self.n_random_features, alpha=self.entropy_reg)\
                .fit(x, y, sample_weight=instance_importance)
        else:
            return DecisionStumpSamme(n_random_features=self.n_random_features, alpha=self.entropy_reg)\
                .fit(x, y, sample_weight=instance_importance)
            
    def _get_updated_instance_importance(self, instance_importance, weighted_error,
                                         alpha, stump_prediction, classifier_decision, y):
        missclassified_samples = (stump_prediction != y)
        if self.mode == 'exponential':
            if self.n_labels > 2:
                if self.type == 'discrete':
                    return normalize(instance_importance * np.exp(missclassified_samples * alpha))
                else:
                    result = np.array([np.dot(self._y[idx], self._log_proba[idx]) for idx in range(y.shape[0])])
                    return normalize(instance_importance * np.exp(-result * (self.n_labels - 1) / self.n_labels))
            else:
                instance_importance[missclassified_samples] /= 2 * weighted_error
                instance_importance[~missclassified_samples] /= 2 * (1 - weighted_error)
                return instance_importance
        if self.mode == 'quadratic':
            if self.n_labels > 2:
                return 2 * (classifier_decision - y) / y.shape[0]
            else:
                return - 2 * (1 - classifier_decision * y)
        elif self.mode == 'logit':
            return normalize(- 1 / (1 + np.exp(classifier_decision * y)) / y.shape[0])

    def _get_alpha(self, n_iter, instance_importance, weighted_error,
                   stump_decision, previous_classifier_decision, y):
        if self.mode == 'exponential':
            if self.n_labels > 2:
                if self.type == 'discrete':
                    return np.log((1 - weighted_error) / weighted_error) + np.log(self.n_labels - 1)
                else:
                    return 1
            else:
                return np.log((1 - weighted_error) / weighted_error) / 2
        elif self.mode == 'quadratic':
            if self.n_labels > 2:
                return ((y - previous_classifier_decision) * stump_decision).sum() / (stump_decision != 0).sum()
            else:
                return ((y - previous_classifier_decision) * stump_decision).sum() / y.shape[0]
        elif self.mode == 'logit':
            previous_sum = previous_classifier_decision * y
            stump = stump_decision * y

            def logit(alpha, jacobian=False):
                loss = np.log(1 + np.exp(-previous_sum - alpha * stump)) / y.shape[0]
                if jacobian:
                    derivative = (-(1 / (1 + np.exp(previous_sum + alpha * stump))) * stump / y.shape[0]).sum()
                    return np.array([loss.sum()]), np.array([derivative])
                return np.array([loss.sum()])

            return lm(np.array([0]), logit, max_iters=100, alpha=1e1)[0][0]
        
    def _compute_loss(self, weighted_error, classifier_decision, y):
        if self.mode == 'exponential':
            if self.n_labels > 2:
                linear_combination = (self._y[:y.shape[0]] * classifier_decision[:y.shape[0]]).sum(axis=1)
                linear_combination /= -self.n_labels
                return np.exp(linear_combination).sum() / y.shape[0]
            if self.n_labels == 2:
                self._Z *= 2 * np.sqrt(weighted_error * (1 - weighted_error))
                return self._Z
        elif self.mode == 'quadratic':
            if self.n_labels > 2:
                return ((classifier_decision - y) ** 2).sum() / y.shape[0]
            else:
                return ((1 - y * classifier_decision) ** 2).sum() / y.shape[0]
        elif self.mode == 'logit':
            return (np.log(1 + np.exp(-classifier_decision * y))).sum() / y.shape[0]
    
    def fit(self, X, Y, validation_split=None):
        """
        :param X: features
        :param Y: labels
        """
        X, Y = check_X_y(X, Y, multi_output=False, y_numeric=True, estimator='AdaBoost')

        check_array(X, accept_sparse=False, dtype='numeric', force_all_finite=True, estimator='AdaBoost')
        check_array(Y, accept_sparse=False, dtype='numeric', force_all_finite=True, ensure_2d=False,
                    estimator='AdaBoost')

        if self.valid_boost > 0:
            indexes = np.arange(X.shape[0])
            np.random.shuffle(indexes)
            X = X[indexes, :]
            Y = Y[indexes]

        if self.n_random_features is None:
            self.n_random_features = X.shape[1]

        if self.verbose:
            metrics = ['w_err', 'loss', 'err', 'acc']
            if validation_split is not None:
                metrics.append('val_acc')
            _bar = create_progbar_numba(self.n_estimators, stateful_metrics="|".join(metrics))

        if validation_split is not None:
            if isinstance(validation_split, tuple):
                x_test, y_test = validation_split
            else:
                X, x_test, Y, y_test = train_test_split(X, Y, test_size=validation_split, shuffle=True)
        
        self.losses = []
        self.errors = []
        self.weighted_errors = []
        self.decision_stumps = []
        self.instance_importance = np.ones((X.shape[0],)) / X.shape[0]
        if validation_split is not None:
            self.validation_errors = []

        self.n_labels = len(np.unique(Y))
        if self.n_labels > 2 and self.mode == 'exponential':
            classifier_decision = np.zeros((X.shape[0], self.n_labels))
            if validation_split is not None:
                validation_decision = np.zeros((x_test.shape[0], self.n_labels))
        else:
            classifier_decision = np.zeros((X.shape[0],))
            if validation_split is not None:
                validation_decision = np.zeros((x_test.shape[0],))

        if self.n_labels > 2 and self.mode == 'exponential':
            self._y = np.zeros((Y.shape[0], self.n_labels))
            self._y.fill(-1/(self.n_labels-1))
            for idx in range(Y.shape[0]):
                self._y[idx, Y[idx]] = 1
        elif self.n_labels > 2 and self.mode == 'quadratic':
            Y = 2 * Y

        n_iter = 1

        used_samples = X.shape[0]
        if self.valid_boost:
            used_samples = int(X.shape[0] / 2)
            self.instance_importance[:used_samples] *= 2

        while n_iter <= self.n_estimators:
            if self.valid_boost and n_iter <= self.valid_boost * self.n_estimators:
                max_iter = self.n_estimators * self.valid_boost
                t = int(X.shape[0] / 2 + int((np.log(n_iter) / np.log(max_iter)) * X.shape[0] / 2))

                if t - used_samples > 0:
                    self.instance_importance[:used_samples] *= used_samples / t

                    misclassified = self._predict(classifier_decision[used_samples:t]) != Y[used_samples:t]
                    n_misclassified = misclassified.sum()
                    if n_misclassified:
                        p = 1 / (2 * n_misclassified)
                        self.instance_importance[used_samples:t][misclassified] = p
                    if t - used_samples - n_misclassified:
                        p = 1 / (2 * (t - used_samples - n_misclassified))
                        self.instance_importance[used_samples:t][~misclassified] = p
                    self.instance_importance[used_samples:t] *= (t - used_samples) / t

                    self._y[:used_samples] *= used_samples / t
                    self._y[used_samples:t] *= (t - used_samples) / t

                used_samples = t

            decision_stump = self._get_weak_classifier(self.instance_importance[:used_samples],
                                                       X[:used_samples, :],
                                                       Y[:used_samples])
            stump_prediction = decision_stump.predict(X[:used_samples, :])
            weighted_error = self.instance_importance[:used_samples][stump_prediction != Y[:used_samples]].sum()

            if weighted_error == 0:
                self.decision_stumps.append((1.0, decision_stump))
                if self.verbose:
                    values = [0, 0, 0, 1]

                    _bar.update(
                        self.n_estimators,
                        "|".join(metrics),
                        values
                    )

                break
            elif weighted_error >= 0.5 and self.n_labels <= 2:
                break

            stump_decision = decision_stump.decision_function(X)
            if self.n_labels > 2 and self.mode == 'exponential' and self.type != 'discrete':
                self._log_proba = np.log(decision_stump.predict_proba(X[:used_samples]))

            self.weighted_errors.append(weighted_error)

            alpha = self.learninig_rate *\
                    self._get_alpha(n_iter,
                                    self.instance_importance[:used_samples],
                                    weighted_error,
                                    stump_decision[:used_samples],
                                    classifier_decision[:used_samples],
                                    Y[:used_samples])
            if not (self.n_labels > 0 and self.mode == 'quadratic'):
                assert alpha >= 0
            self.decision_stumps.append((alpha, decision_stump))

            classifier_decision += alpha * stump_decision
            if validation_split is not None:
                validation_decision += alpha * decision_stump.decision_function(x_test)

            classifier_error = (self._predict(classifier_decision) != Y).sum() / Y.shape[0]
            if validation_split is not None:
                validation_error = (self._predict(validation_decision) != y_test).sum() / y_test.shape[0]

            self.errors.append(classifier_error)
            if validation_split is not None:
                self.validation_errors.append(validation_error)

            self.instance_importance[:used_samples] = self._get_updated_instance_importance(
                self.instance_importance[:used_samples],
                weighted_error,
                alpha,
                stump_prediction[:used_samples],
                classifier_decision[:used_samples],
                Y[:used_samples]
            )

            loss = self._compute_loss(weighted_error, classifier_decision, Y[:used_samples])
            self.losses.append(loss)
            if self.verbose:
                values = [
                    weighted_error,
                    loss,
                    classifier_error,
                    1 - classifier_error
                ]
                if validation_split is not None:
                    values.append(1 - validation_error)

                _bar.update(
                    n_iter,
                    "|".join(metrics),
                    values
                )

            n_iter += 1

        return self

    def decision_function(self, X):
        if self.n_labels > 2 and self.mode == 'exponential':
            prediction = np.zeros((X.shape[0], self.n_labels))
        else:
            prediction = np.zeros((X.shape[0],))

        for alpha, decision_stump in self.decision_stumps:
            prediction += alpha * decision_stump.decision_function(X)

        return prediction

    def _predict(self, decision):
        if self.n_labels > 2 and self.mode == 'exponential':
            return decision.argmax(axis=1)
        elif self.n_labels == 2:
            return (decision > 0).astype('int')
        else:
            return NotImplemented()

    def predict(self, X):
        decision = self.decision_function(X)
        return self._predict(decision)

    def predict_proba(self, X):
        if self.mode == 'exponential':
            if self.n_labels == 2:
                alpha_sum = [alpha for alpha, prediction in self.decision_stumps]
                return (self.decision_function(X) + alpha_sum) / (2 * alpha_sum)
            else:
                if self.type == 'discrete':
                    raise NotImplemented()
                else:
                    prediction = self.decision_function(X)
                    prediction /= self.n_labels
                    prediction = np.exp((1. / (self.n_labels - 1)) * prediction)

                    normalizer = prediction.sum(axis=1)[:, np.newaxis]
                    normalizer[normalizer == 0.0] = 1.0
                    prediction /= normalizer
                    return prediction
