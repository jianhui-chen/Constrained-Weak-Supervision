import sys
import numpy as np

from scipy.optimize import minimize

from BaseClassifier import BaseClassifier

class GECriterion(BaseClassifier):
    """
    GE Criterion class for implementation
    """

    def __init__(self, max_iter=None, log_name=None):
        # based on args

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            """
            self.logger = Logger("logs/Baseline/" + log_name + "/" + 
                                 str(weak_signals_proba.shape[0]) + 
                                 "_weak_signals/")      # this can be modified to include date and time in file name
            """
        else:
            sys.exit("Not of string type")

        # not based on args bc based on feature number
        self.weights = None
    
    # Uses prediction stuff in parent class

    def _compute_reference_distribution(self, y, weak_signal_proba):
        """
        Computes the score value of the reference expectation

        :param labels: size n labels for each instance in the dataset
        :type labels: array
        :param weak_signal: weak signal trained using one dimensional feature
        :type  weak_signal: array
        :return: tuple containing scalar values of positive and negative reference probability distribution
        :rtype: float
        """
        threshold = 0.5
        positive_index = np.where(weak_signal_proba >= threshold)
        negative_index = np.where(weak_signal_proba < threshold)
        pos_feature_labels = y[positive_index]
        neg_feature_labels = y[negative_index]

        try:
            with np.errstate(all='ignore'):
                reference_pos_probability = np.sum(pos_feature_labels) / pos_feature_labels.size
                reference_neg_probability = np.sum(neg_feature_labels) / neg_feature_labels.size
        except:
            reference_pos_probability = np.nan_to_num(np.sum(pos_feature_labels) / pos_feature_labels.size) + 0
            reference_neg_probability = np.nan_to_num(np.sum(neg_feature_labels) / neg_feature_labels.size) + 0

        return reference_pos_probability, reference_neg_probability

    def _compute_empirical_distribution(self, est_probability, weak_signal_proba):
        """
        Computes the score value of the emperical distribution

        :param est_probability: size n estimated probabtilities for the instances
        :type labels: array
        :param weak_signal_proba: weak signal trained using one dimensional feature
        :type  weak_signal: array
        :return: (tuple of scalar values of the empirical distribution, tuple of index of instances)
        :rtype: tuple
        """
        threshold = 0.5
        positive_index = np.where(weak_signal_proba >= threshold)
        negative_index = np.where(weak_signal_proba < threshold)
        pos_feature_labels = est_probability[positive_index]
        neg_feature_labels = est_probability[negative_index]

        try:
            with np.errstate(all='ignore'):
                empirical_pos_probability = np.sum(pos_feature_labels) / pos_feature_labels.size
                empirical_neg_probability = np.sum(neg_feature_labels) / neg_feature_labels.size
        except:
            empirical_pos_probability = np.nan_to_num(np.sum(pos_feature_labels) / pos_feature_labels.size) + 0
            empirical_neg_probability = np.nan_to_num(np.sum(neg_feature_labels) / neg_feature_labels.size) + 0

        empirical_probability = empirical_pos_probability, empirical_neg_probability
        instances_index = positive_index, negative_index
        return empirical_probability, instances_index

    def _train_ge_criteria(self, X, y, new_weights, weak_signals_probas):
        """
        This internal function returns the objective value of ge criteria

        :param new_weights: weights to use for computing multinomial logistic regression
        :type new_weights: ndarray
        :return: tuple containing (objective, gradient)
        :rtype: (float, array)
        """

        obj = 0
        score = X.dot(new_weights)
        #probs, grad = logistic(score)
        probs = 1 / (1 + np.exp(-score))
        grad = probs * (1 - probs)
        gradient = 0
        num_weak_signals = weak_signals_probas.shape[0]
        # Code to compute the objective function
        for i in range(num_weak_signals):
            weak_signal_proba = weak_signals_probas[i]
            reference_probs = self._compute_reference_distribution(y, weak_signal_proba)
            empirical_probs, index = self._compute_empirical_distribution(probs, weak_signal_proba)

            # empirical computations
            pos_empirical_probs, neg_empirical_probs = empirical_probs
            pos_index, neg_index = index

            # reference computations
            pos_reference_probs, neg_reference_probs = reference_probs

            try:
                with np.errstate(all='ignore'):
                    # compute objective for positive probabilities
                    obj += pos_reference_probs * np.log(pos_reference_probs / pos_empirical_probs)
                    gradient += (pos_reference_probs / pos_empirical_probs) * X[pos_index].T.dot(grad[pos_index]) / grad[pos_index].size

                    # compute objective for negative probabilities
                    obj += neg_reference_probs * np.log(neg_reference_probs / neg_empirical_probs)
                    gradient += (neg_reference_probs / neg_empirical_probs) * X[neg_index].T.dot(grad[neg_index]) / grad[neg_index].size
            except:
                # compute objective for positive probabilities
                obj += np.nan_to_num(pos_reference_probs * np.log(pos_reference_probs / pos_empirical_probs))
                gradient += np.nan_to_num((pos_reference_probs / pos_empirical_probs) * X[pos_index].T.dot(grad[pos_index]) / grad[pos_index].size)

                # compute objective for negative probabilities
                obj += np.nan_to_num(neg_reference_probs * np.log(neg_reference_probs / neg_empirical_probs))
                gradient += np.nan_to_num((neg_reference_probs / neg_empirical_probs) * X[neg_index].T.dot(grad[neg_index]) / grad[neg_index].size)

        objective = obj + (0.5 * np.sum(new_weights**2))
        gradient = new_weights - gradient

        return objective, gradient


    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, y):
        """
        X is passed in as (d, n), to match the other fits.
        X is thus transposed until this is fixed
        Removed the check gradient param

        error_bounds param is not used, but included for consistency
        """

        if y is None:
            sys.exit("GE Criterion requires labels")

        X = X.T 
        n, d = X.shape
        self.weights = np.random.rand(d)

        # optimizer
        res = minimize(lambda w: self._train_ge_criteria(X, y, w, weak_signals_probas)[0], jac=lambda w: self._train_ge_criteria(X, y, w, weak_signals_probas)[1].ravel(), x0=self.weights) 
        self.weights = res.x

        return self