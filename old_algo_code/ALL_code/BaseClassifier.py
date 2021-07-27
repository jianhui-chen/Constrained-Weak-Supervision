import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from scipy.optimize import minimize


"Contains abstract base class BaseClassifier"

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