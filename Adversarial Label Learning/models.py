# Note for ALL model class –– inherit from sklean base.py?

# Need to consider if we want to create an abstract base class

from log import Logger

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
                 max_iter=10000, logging=False):
    
        self.weak_signals_proba = weak_signals_proba
        self.weak_signals_error_bounds = weak_signals_error_bounds
        self.max_iter = max_iter
        self.logging = logging      # might want to rename this , can initialize logger object here

    
    def fit(self, X, y=None):
        """
        Fits the model according to given training data (X)

        Parameters
        ----------
        X : {}


        Returns
        -------
        model

        """
        print(X) #placeholder code

        # Would need to do if statement with self.logging to implement logger


    def predict_proba(self, X):     # Note to self: this should replace "probablity" function in train_classifier
        """
        Computes probability estimates for given class
        Should be able to be extendable for multi-class implementation

        Parameters
        ----------
        X : 


        Returns
        -------
        label predictions

        """
        print(X) # placeholder code
        

class Baseline():
    """
    Baseline Classifier
    Need to add more on its functionality. 
    """

    def __init__(self, logging=False):
        self.logging = logging