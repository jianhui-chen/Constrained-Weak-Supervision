import sys
import numpy as np

from BaseClassifier import BaseClassifier
from ConstraintEstimator import ConstraintEstimator
from utilities import convert_to_ovr_signals
from log import Logger


class ALL(BaseClassifier):
    """
    Adversarial Label Learning Classifier

    This class implements ALL training on a set of data

    Parameters
    ----------
    max_iter : int, default=10000
        Maximum number of iterations taken for solvers to converge.

    log_name : string, default=None
        Specifies directory name for a logger object.

    rho : float
            Scalar tuning hyperparameter

    """

    def __init__(self, max_iter=10000, log_name=None, rho=2.5):

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            
            """
            Logging is done via TensorBoard. 
            The suggested storage format for each run is by the date/time the expirment was started, 
            and then by dataset, and then by algorithm.
            """

            self.logger = Logger("logs/" + log_name)      # this can be modified to include date and time in file name
        else:
            sys.exit("Not of string type")

        self.weights = None
        self.rho = rho


    def _bound_loss(self, y, a_matrix, bounds):
        """
        Computes the gradient of Lagrangian inequality penalty parameters

        Parameters
        ----------
        :param y: estimated labels for the data
        :type y: ndarray
        :param a_matrix: size num_weak, num_data of a constraint matrix
        :type a_matrix: ndarray
        :param bounds: size num_weak of the bounds for the constraint
        :type bounds: ndarray

        Returns
        -------
        :return: loss of the constraint (num_weak, num_class)
        :rtype: ndarray
        """

        AY = a_matrix.dot(y)
        return AY - bounds


    def _weight_gradient(self, X):
        """
        Computes the gradient the probabilities wrt to the weights

        Parameters
        ----------
        :param X: size (num_features, num_data) of estimated labels for the data
        :type X: ndarray

        Returns
        -------
        ndarray of size (num_features, n) gradients for probability wrt to weight

        """


        y = X.dot(self.weights)

        # replacing logistic func for now
        y_squish = 1 / (1 + np.exp(-y))
        grad = y_squish * (1 - y_squish)

        grad = X.T * grad

        return grad


    def _y_gradient(self, y, error_constraints,
                    learnable_probabilities, rho, gamma):
        """
        Computes gradient of y

        Parameters
        ----------
        :param y: estimated Labels
        :type  y: ndarray
        :param error_constraints: error constraints of the weak signals
        :type  error_constraints: dictionary containing both a_matrix and bounds
        :param learnable_probabilities: estimated probabilities for the classifier
        :type learnable_probabilities: ndarray
        :param rho: Scalar tuning hyperparameter
        :type rho: float
        :param gamma: vector of lagrangian inequality penalty parameters
        :type gamma: array

        Returns
        -------
        ndarray gradient of y
        """

        gradient = 0
        a_matrix = np.squeeze(error_constraints['A'], axis=-1)
        bounds = np.squeeze(error_constraints['b'], axis=-1)

        n = learnable_probabilities.size
        learnable_term = 1 - (2 * learnable_probabilities)
        learnable_term = learnable_term / n

        gamma_term = np.dot(a_matrix.T, gamma)

        augmented_term = self._bound_loss(y, a_matrix, bounds)
        augmented_term = augmented_term.clip(min=0)
        augmented_term = rho * np.dot(a_matrix.T, augmented_term)

        return learnable_term + gamma_term - augmented_term



    def fit(self, X, weak_signals, weak_signals_error_bounds=None):
        """
        Optimizes model according to given training data (X)

        Parameters
        ----------
        :param X: training data
        :type X: ndarray
        :param weak_signals: weak signals for the data
        :type  weak_signals_probas: ndarray
        :param weak_signals_error_bounds: error bounds of the weak signals
        :type weak_signals_error_bounds: ndarray

        Returns
        -------
        self
            Fitted and optimized estimator

        """
        t = 0
        n, d = X.shape
        weights = np.zeros(d)
        self.weights = weights
        learnable_proba = self.predict_proba(X)

        # initialize algorithm variables
        weak_signals = convert_to_ovr_signals(weak_signals)
        m,n,k = weak_signals.shape
        y = 0.5 * np.ones(n)
        gamma = np.zeros(m)
        one_vec = np.ones(n)
        lr = 0.0001
        converged = False

        cons = ConstraintEstimator()
        self.constraints = cons.error_constraint(weak_signals, weak_signals_error_bounds)

        a_matrix = np.squeeze(self.constraints['A'], axis=-1)
        bounds = np.squeeze(self.constraints['b'], axis=-1)

        while not converged and t < self.max_iter:

            rate = 1 / (1 + t)

            # update y
            old_y = y
            y_grad = self._y_gradient(y, self.constraints, learnable_proba, self.rho, gamma)
            y = y + rate * y_grad
            # projection step: clip y to [0, 1]
            y = y.clip(min=0, max=1)

            # compute gradient of probabilities
            dl_dp = (1 / n) * (1 - 2 * old_y)

            # update gamma
            old_gamma = gamma
            gamma_grad = self._bound_loss(y, a_matrix, bounds)
            gamma = gamma - self.rho * gamma_grad
            gamma = gamma.clip(max=0)

            # compute gradient of probabilities wrt weights
            dp_dw = self._weight_gradient(X)
            # update weights
            old_weights = self.weights.copy()
            weights_gradient = dp_dw.dot(dl_dp)

            # update weights of the learnable functions
            weights = weights - lr * weights_gradient

            conv_weights = np.linalg.norm(weights - old_weights)
            conv_y = np.linalg.norm(y - old_y)

            # check that inequality constraints are satisfied
            ineq_constraint = gamma_grad
            ineq_infeas = np.linalg.norm(ineq_constraint.clip(min=0))

            converged = np.isclose(0, conv_y, atol=1e-6) and \
                        np.isclose(0, ineq_infeas, atol=1e-6) and \
                        np.isclose(0, conv_weights, atol=1e-5)


            learnable_proba = self.predict_proba(X)
            t += 1


            self.weights = weights

    def predict_proba(self, X):
        """
        Computes probabilistic labels for the training data

        Parameters
        ----------
        X : ndarray of training data


        Returns
        -------
        probas : ndarray of label probabilities

        """

        y = X.dot(self.weights)

        # logistic function
        y_proba = 1 / (1 + np.exp(-y))

        return y_proba


    def predict(self, X):
        """
        Computes predicted classes for the training data.

        Parameters
        ----------
        X : ndarray of training data

        Returns
        -------
        predicted classes : ndarray array of predicted classes
        """
        proba = self.predict_proba(X)
        return np.round(proba)
