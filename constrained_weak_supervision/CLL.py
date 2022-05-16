import sys
import numpy as np

from BaseClassifier import BaseClassifier
from ConstraintEstimator import ConstraintEstimator
from utilities import convert_to_ovr_signals
from log import Logger


class CLL(BaseClassifier):
    """
    Constrained Label Learning

    This class implements CLL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations taken for solvers to converge.

    num_trials : int, default=3
        number of time's labels are estimated before the mean is taken
    
    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=300, num_trials=3, log_name=None,):

        self.max_iter = max_iter
        self.num_trials = num_trials

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name) 
        else:
            sys.exit("Not of string type")
        self.constraints = None


    def _bound_loss(self, y, a_matrix, bounds):
        """
        Computes the gradient of lagrangian inequality penalty parameters

        Parameters
        ----------
        :param y: size (num_data, num_class) of estimated labels for the data
        :type y: ndarray
        :param a_matrix: size (num_weak, num_data, num_class) of a constraint matrix
        :type a_matrix: ndarray
        :param bounds: size (num_weak, num_class) of the bounds for the constraint
        :type bounds: ndarray

        Returns
        -------
        :return: loss of the constraint (num_weak, num_class)
        :rtype: ndarray
        """
        constraint = np.zeros(bounds.shape)

        for i, current_a in enumerate(a_matrix):
            constraint[i] = np.sum(current_a * y, axis=0)
        return constraint - bounds


    def _y_gradient(self, y, error_constraints):
        """
        Computes y gradient

        Parameters
        ----------
        :param y: estimated Labels
        :type  y: ndarray
        :param error_constraints: error constraints of the weak signals
        :type  error_constraints: dictionary containing both a_matrix and bounds

        Returns
        -------
        :return: computed gradient
        :rtype: float
        """
        gradient = 0
        a_matrix = error_constraints['A']
        bound_loss = error_constraints['bound_loss']

        for i, current_a in enumerate(a_matrix):
            constraint = a_matrix[i]
            gradient += 2*constraint * bound_loss[i]

        return gradient


    def _run_constraints(self, y, error_constraints):
        """
        Run constraints from CLL

        :param y: Random starting values for labels
        :type  y: ndarray 
        :param error_constraints: error constraints (a_matrix and bounds) of the weak signals 
        :type  error_constraints: dictionary

        :return: estimated learned labels
        :rtype: ndarray
        """
        grad_sum = 0
        a_matrix = error_constraints['A']
        bounds = error_constraints['b']

        for iter in range(self.max_iter):

            # get bound loss for constraint
            loss = self._bound_loss(y, a_matrix, bounds)

            # update constraint values
            error_constraints['bound_loss'] = loss
            violation = np.linalg.norm(loss.clip(min=0))

            # Update y˜ with its gradient
            y_grad = self._y_gradient(y, error_constraints)
            grad_sum += y_grad**2
            y = y - y_grad / np.sqrt(grad_sum + 1e-8)
            y = np.clip(y, a_min=0, a_max=1)

            # log current data 
            if self.logger is not None and iter % 10 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("y", np.average(y), iter)
                    self.logger.log_scalar("y_grad", np.average(y_grad), iter)
                    self.logger.log_scalar("loss", np.average(loss), iter)
                    self.logger.log_scalar("violation", np.average(violation), iter)
        return y
    

    def fit(self, weak_signals, weak_signals_error_bounds=None):
        """
        Finds estimated labels

        Parameters
        ----------
        :param weak_signals: weak signals for the data
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error bounds of the weak signals
        :type  error_bounds: ndarray

        """
        cons = ConstraintEstimator(error_threshold=0)
        self.constraints = cons.error_constraint(weak_signals, weak_signals_error_bounds)


    def get_score(self, true_labels, predicted_probas, metric='accuracy'):
        """
        Calculate accuracy of the model

        Parameters
        ----------
        :param true_labels: true labels of data set
        :type  true_labels: ndarray
        :param predicted_probas: Estimated labels that where trained on
        :type  predicted_probas: ndarray

        Returns
        -------
        :return: percent accuary of Estimated labels given the true labels
        :rtype: float
        """
        score = super().get_score(true_labels, predicted_probas, metric)
        return score


    def predict_proba(self, weak_signals):
        """
        Computes probability estimates for given class
        Should be able to be extendable for multi-class implementation

        Parameters
        ----------
        weak_signals : ndarray of weak signals


        Returns
        -------
        probas : ndarray of label probabilities

        """

        weak_signals = convert_to_ovr_signals(weak_signals)
        m, n, num_classes = weak_signals.shape

        # initialize y and lists
        y = np.random.rand(n, num_classes)
        ys = []

        for i in range(self.num_trials):
            ys.append(self._run_constraints(y, self.constraints))
        return np.squeeze(np.mean(ys, axis=0))


    def predict(self, predicted_labels):
        """
        Computes predicted classes for the weak signals.

        Parameters
        ----------
        predicted_labels : predicted_labels

        Returns
        -------
        predicted classes : ndarray array of predicted classes
        """
        predicted_labels = np.squeeze(predicted_labels)
        if len(predicted_labels.shape)==1:
            return np.round(predicted_labels)
        return np.argmax(predicted_labels, axis=-1)

