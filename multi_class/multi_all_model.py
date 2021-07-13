import sys
sys.path.append('../')

from ALL_code.BaseClassifier import BaseClassifier

""" Multi-ALL class """

class MultiALL(BaseClassifier):
    """
    Multi Class Adversarial Label Learning Classifier

    This class implements Multi ALL training on a set of data

    Parameters
    ----------
    max_iter : int, default=10000
        Maximum number of iterations taken for solvers to converge.

    log_name : Can be added, need to deal with some issues with imports

    """

    def __init__(self, max_iter=10000):
    
        self.max_iter = max_iter

        self.weights = None


    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, weak_signals_precision):
        """
        Fits MultiAll model

        Parameters
        ----------
        X : ndarray of shape (n_examples, n_features)      
            Training matrix, where n_examples is the number of examples and 
            n_features is the number of features for each example


        weak_signals_proba : ndarray of shape (n_weak_signals, n_examples, n_classes)
            A set of soft or hard weak estimates for data examples.
            This may later be changed to accept just the weak signals, and these 
            probabilities will be calculated within the ALL class. 

        weak_signals_error_bounds : ndarray of shape (n_weak_signals, n_classes)
            Stores upper bounds of error rates for each weak signal.

        weak_signals_precision : ndarray of shape (n_weak_signals, n_class)

        Returns
        -------
        self
            Fitted estimator

        """
