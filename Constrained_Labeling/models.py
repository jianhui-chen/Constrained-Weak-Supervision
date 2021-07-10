import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod

from log import Logger


"""
Contains abstract base class BaseClassifier

Also contains CLL
"""

class BaseClassifier(ABC):
    """
    Abstract Base Class for learning classifiers

    Constructors are all defined in subclasses

    Current purely abstract methods are:
    - fit
    """

    def predict(self, predicted_probas):
        """
        Computes predicted labels based on probability predictions.
        
        NOTE: It may be good to have a version that takes in data X, instead
        of precomputed probabilities. 

        Parameters
        ----------
        predicted_probas : ndarray of shape (n_examples,)
            Precomputed probabilities

        Returns
        -------
        predicted_labels : ndarray of shape (n_examples,)
            Binary labels
        """
  
        predicted_labels = np.zeros(predicted_probas.size)

        # could also implement by rounding
        predicted_labels[predicted_probas > 0.5] =1    
        return predicted_labels
    
    def get_accuracy(self, true_labels, predicted_probas):
        """
        Computes accuracy of predicted labels based on the true labels.
        This may be good to move out of the class, also make it take in 
        predicted labels, not probas.

        Parameters
        ----------
        true_labels : ndarray of shape (n_examples,)

        predicted_probas : ndarray of shape (n_examples,)
            I don't know why I pass in probas instead of labels

        Returns
        -------
        score : float
            Value between 0 to 1.00

        """
        score = accuracy_score(true_labels, self.predict(predicted_probas))
        return score

 
    def predict_proba(self, X):   
        """
        Computes probability estimates for given class
        Should be able to be extendable for multi-class implementation

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)

        """
        if self.weights is None:
            sys.exit("No Data fit")
        
        try: 
            y = self.weights.dot(X)
        except:
            y = X.dot(self.weights)

        # first line of logistic from orig code, squishes y values
        probas = 1 / (1 + np.exp(-y))    
        
        return probas.ravel()

    @abstractmethod 
    def fit(self, X):
        """
        Abstract method to fit models

        Parameters
        ----------
        X : ndarry 
        """
        pass



class CLL(BaseClassifier):
    """
    Constrained label learning Classifier

    This class implements CLL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations taken for solvers to converge.

    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=300, log_name=None):

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/CLL/" + log_name)      # this can be modified to include date and time in file name
        else:
            sys.exit("Not of string type")

        self.model = None

    def get_accuracy(self, true_labels, predicted_probas): 
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
        try:
            n, k = true_labels.shape
            if k > 1:
                assert true_labels.shape == predicted_probas.shape
                return np.mean(np.equal(np.argmax(true_labels, axis=-1),
                                        np.argmax(predicted_probas, axis=-1)))
        except:
            if len(true_labels.shape) == 1:
                y_pred = np.round(predicted_probas.ravel())
    
        assert true_labels.shape == y_pred.shape
        return np.mean(np.equal(true_labels, np.round(y_pred)))

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
        # n, k = y.shape

        for i, current_a in enumerate(a_matrix):
            constraint[i] = np.sum(current_a * y, axis=0)
        return constraint - bounds


    def _y_gradient(self, y, error_bounds):
        """
        Computes y gradient

        Parameters
        ----------
        :param y: estimated Labels
        :type  y: ndarray
        :param weak_signals_error_bounds: error constraints of the weak signals
        :type  weak_signals_error_bounds: dictionary containing both a_matrix and bounds

        Returns
        -------
        :return: computed gradient
        :rtype: float
        """
        gradient = 0
        a_matrix = error_bounds['A']
        bound_loss = error_bounds['bound_loss']

        for i, current_a in enumerate(a_matrix):
            constraint = a_matrix[i]
            gradient += 2*constraint * bound_loss[i]

        return gradient


    def _run_constraints(self, y, rho, error_constraints):
        """
        Run constraints from CLL

        :param y: Random starting values for labels
        :type  y: ndarray 
        :param rho: ????????????
        :type  rho: 0.1 (for some reason) 
        :param error_constraints: error constraints (a_matrix and bounds) of the weak signals 
        :type  error_constraints: dictionary

        :return: estimated learned labels
        :rtype: ndarray
        """
        n, k = y.shape
        rho = n
        grad_sum = 0
        lamdas_sum = 0

        """
            should this loop be for while not converged??? 
            Maybe when violation and loss reach/nears 0
        """
        for iter in range(self.max_iter):
            
            current_constraint = error_constraints
            a_matrix = current_constraint['A']
            bounds = current_constraint['b']

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

            # logg current data 
            if self.logger is not None and iter % 10 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("y", np.average(y), iter)
                    self.logger.log_scalar("y_grad", np.average(y_grad), iter)
                    # might not need both violation and loss
                    self.logger.log_scalar("loss", np.average(loss), iter)
                    self.logger.log_scalar("violation", np.average(violation), iter)
        return y


    def fit(self, weak_signals_probas, error_bounds):
        """
        Finds estimated labels

        Parameters
        ----------
        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for each example
        :type  weak_signals_probas: ndarray 
        :param error_bounds: error constraints (a_matrix and bounds) of the weak signals. Contains both 
                             left (a_matrix) and right (bounds) hand matrix of the inequality 
        :type  error_bounds: dictionary 

        Returns
        -------
        :return: average of learned labels over several trials
        :rtype: ndarray
        """
        assert len(weak_signals_probas.shape) == 3, "Reshape weak signals to num_weak x num_data x num_class"
        m, n, k = weak_signals_probas.shape

        # initialize y and hyperparameters
        y = np.random.rand(n, k)
        
        rho = 0.1  #not sure what rho is for 

        # t = 3  # number of random trials
        # ys = []
        # for i in range(t):
        #     ys.append( self._run_constraints(y, rho, error_bounds) )
        # return np.mean(ys, axis=0)
        
        return self._run_constraints(y, rho, error_bounds)

    

