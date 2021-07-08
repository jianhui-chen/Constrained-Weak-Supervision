
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from log import Logger


"""
Contains abstract base class BaseClassifier

Also contains ALL, LabelEstimator, and GECriterion

All code takes in data as (n_features, n_examples), which is BAD and should
be changed, along with the weak signals.


To work on:
    Silent try/catches should be removed.
    Add logging to console.
    Switch data forms.
    Modify predict() and get_accuracy() as needed.
    Add more algos.

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
    max_iter : int, default=10000
        Maximum number of iterations taken for solvers to converge.

    log_name : string, default=None
        Specifies directory name for a logger object.

    """

    def __init__(self, max_iter=None, log_name=None):

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            
            self.logger = Logger("logs/CLL/" + log_name)      # this can be modified to include date and time in file name
            
        else:
            sys.exit("Not of string type")

        # self.weights = None
        # not based on args bc based on feature number
        self.model = None

    def predict_proba(self, X):
        """
        Computes probability estimates for given class

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)
        """
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict_proba(X.T)[:,1]

        return probabilities


    def _estimate_labels(self, weak_signals_probas, weak_signals_error_bounds):
        """
        Estimates labels by averaging weak signals

        Parameters
        ----------
        weak_signals_proba : ndarray of shape (n_weak_signals, n_examples)
            A set of soft or hard weak estimates for data examples.
            This may later be changed to accept just the weak signals, and these 
            probabilities will be calculated within the ALL class. 

        weak_signals_error_bounds : ndarray of shape (n_weak_signals,)
            Stores upper bounds of error rates for each weak signal.


        Returns
        -------
        Estimated labels

        """
        labels=np.zeros(weak_signals_probas.shape[1]) # no of examples
        average_weak_labels = np.mean(weak_signals_probas, axis=0)
        labels[average_weak_labels > 0.5] = 1
        labels[average_weak_labels <= 0.5] = 0


        return labels























    ############
    # Fit code # 
    ############

    def bound_loss(self, y, a_matrix, bounds):
        """
        Computes the gradient of lagrangian inequality penalty parameters

        :param y: size (num_data, num_class) of estimated labels for the data
        :type y: ndarray
        :param a_matrix: size (num_weak, num_data, num_class) of a constraint matrix
        :type a_matrix: ndarray
        :param bounds: size (num_weak, num_class) of the bounds for the constraint
        :type bounds: ndarray

        :return: loss of the constraint (num_weak, num_class)
        :rtype: ndarray
        """
        constraint = np.zeros(bounds.shape)
        # n, k = y.shape

        for i, current_a in enumerate(a_matrix):
            constraint[i] = np.sum(current_a * y, axis=0)
        return constraint - bounds


    def y_gradient(self, y, error_bounds):
        """
        Computes y gradient

        :param weak_signals_error_bounds: error constraints of the weak signals
        :type  weak_signals_error_bounds: dictionary containing both a_matrix and bounds
        """
        gradient = 0
        a_matrix = error_bounds['A']
        bound_loss = error_bounds['bound_loss']

        for i, current_a in enumerate(a_matrix):
            constraint = a_matrix[i]
            gradient += 2*constraint * bound_loss[i]

        return gradient


    def run_constraints(self, y, rho, error_constraints, iters=300, enable_print=True):
        """
        Trains CLL algorithm

        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for weak signals
        :type  weak_signals_probas: ndarray 
        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for weak signals
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error constraints (a_matrix and bounds) of the weak signals 
        :type  weak_signals_error_bounds: dictionary 

        :return: average of learned labels over several trials
        :rtype: ndarray
        """
        # Run constraints from CLL

        # constraint_keys = constraint_set['constraints']
        n, k = y.shape
        rho = n
        grad_sum = 0
        lamdas_sum = 0

        """
            should this loop be for while not converged??? 
        """
        for iter in range(iters):
            
            current_constraint = error_constraints
            a_matrix = current_constraint['A']
            bounds = current_constraint['b']

            # get bound loss for constraint
            loss = self.bound_loss(y, a_matrix, bounds)

            # update constraint values
            error_constraints['bound_loss'] = loss
            violation = np.linalg.norm(loss.clip(min=0))

            # Update y˜ with its gradient
            y_grad = self.y_gradient(y, error_constraints)
            grad_sum += y_grad**2
            y = y - y_grad / np.sqrt(grad_sum + 1e-8)
            y = np.clip(y, a_min=0, a_max=1)

            # logg current data 
            if self.logger is not None and iter % 10 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("y", np.average(y), iter)
                    self.logger.log_scalar("y_grad", np.average(y_grad), iter)
                    self.logger.log_scalar("loss", np.average(loss), iter)
                    # might not need both violation and loss
                    self.logger.log_scalar("violation", np.average(violation), iter)
        return y


    def fit(self, weak_signals_probas, weak_signals_error_bounds):
        """
        Trains CLL algorithm

        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for weak signals
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error constraints (a_matrix and bounds) of the weak signals 
        :type  weak_signals_error_bounds: dictionary 

        :return: average of learned labels over several trials
        :rtype: ndarray
        """

        # weak_signals = constraint_set['weak_signals']
        assert len(weak_signals_probas.shape) == 3, "Reshape weak signals to num_weak x num_data x num_class"
        m, n, k = weak_signals_probas.shape
        # initialize y
        y = np.random.rand(n, k)
        # initialize hyperparameters
        rho = 0.1

        t = 3  # number of random trials
        ys = []
        for i in range(t):
            ys.append( self.run_constraints(y, rho, weak_signals_error_bounds) )
        return np.mean(ys, axis=0)

    

