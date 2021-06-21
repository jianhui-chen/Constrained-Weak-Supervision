# Note for ALL model class –– inherit from sklean base.py?

# Need to consider if we want to create an abstract base class

from log import Logger
import sys
import numpy as np

class ALL():
    """
    Adversarial Label Learning Classifier

    This class implements ALL training on a set of data
    Comments to be modified

    Parameters
    ----------
    weak_signals_proba : ndarray of shape (n_weak_signals, n_examples)
        A set of soft or hard weak estimates for data examples.
        This may later be changed to accept just the weak signals, and these 
        probabilities will be calculated within the ALL class. 
            __init__ would then store the models, and probabilities would have 
            to be calculated in fit() according to the training data.

    weak_signals_error_bounds : ndarray of shape (n_weak_signals,)
        Stores upper bounds of error rates for each weak signal.

    max_iter : int, default=10000
        Maximum number of iterations taken for solvers to converge.

    logging : bool, default=False
        Specifies if a Logger object should be initialized for logging to 
        Tensorboard.

    """

    def __init__(self, weak_signals_proba, weak_signals_error_bounds, 
                 max_iter=10000, log_name=None):
    
        # based on args
        self.weak_signals_proba = weak_signals_proba
        self.weak_signals_error_bounds = weak_signals_error_bounds
        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name + "/" + 
                                 str(weak_signals_proba.shape[0]) + 
                                 "_weak_signals/")      # this can be modified to include date and time in file name
        else:
            sys.exit("Not of string type")

        # not based on args bc based on feature number
        self.weights = None
    
    def fit(self, X, y=None):
        """
        Fits the model according to given training data (X)

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)       NOTE: Usually this is transposed, might want to change for consistency with other models
            Training matrix, where n_examples is the number of examples and 
            n_features is the number of features for each example

        y : Not to be used for this function, would be used with GE computations

        Returns
        -------
        self
            Fitted estimator

        """
        self.weights = np.zeros(X.shape[0]) # this should be length of n_features
  


       

      


    def predict_proba(self, X):     # Note to self: this should replace "probablity" function in train_classifier
        """
        Computes probability estimates for given class
        Should be able to be extendable for multi-class implementation

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        P : ndarray of shape (n_examples,)

        """
        if self.weights is None:
            sys.exit("No Data fit")
        
        try: 
            y = self.weights.dot(X)
        except:
            y = X.dot(self.weights)

        probas = 1 / (1 + np.exp(-y))    # first line of logistic, squishes y values
        
        return probas
        

class Baseline():
    """
    Baseline Classifier
    Need to add more on its functionality. 
    """

    def __init__(self, logging=False):
        self.logging = logging