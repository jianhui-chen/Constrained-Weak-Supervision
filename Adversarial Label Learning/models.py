# Note for ALL model class –– inherit from sklean base.py?

# Need to consider if we want to create an abstract base class


class ALL():
    """
    Adversarial Label Learning Classifier

    This class implements ALL training on a set of data
    Comments to be modified

    """

    def __init__(self, max_iter=10000, logging=False):
        self.max_iter = max_iter
        self.logging = logging      # might want to rename this 

    
    def fit(self, X):
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