import sys
import numpy as np


from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense

from BaseClassifier import BaseClassifier
from log import Logger



"""
    Includes LabelEstimator and CLL
    CLL should inherit from LabelEstimator – needs to be fixed
"""

class LabelEstimator(BaseClassifier):   # Might want to change the name of Base Classifier?
    """
    Label Estimator + Classifier
    Subclasses can redefine _estimate_labels process

    Parameters
    ----------
    max_iter : int, default=None
        Maximum number of iterations taken for solvers to converge.

    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=None, log_name=None):
    
   
        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        # elif type(log_name) is str:
        #     self.logger = Logger("logs/Baseline/" + log_name + "/" + 
        #                          str(weak_signals_proba.shape[0]) + 
        #                          "_weak_signals/")      # this can be modified to include date and time in file name
        # else:
        #     sys.exit("Not of string type")

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

        probabilities = self.model.predict_proba(X)

        return probabilities
    
    def predict(self, X):
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

        probabilities = self.model.predict(X)

        return probabilities

    
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
    


    def _estimate_labels(self, X, weak_signals_probas, weak_signals_error_bounds):
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
    

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, train_model=None):
        """
        Finds estimated labels

        Parameters
        ----------
        X: ndarray of shape (num_examples, num_features)
            training data

        weak_signals_probas: ndarray of shape (num_weak, num_examples, num _class)
            weak signal probabilites containing -1 for abstaining signals, and between 
            0 to 1 for non-abstaining

        weak_signals_error_bounds: dictionary
            error constraints (a_matrix and bounds) of the weak signals. Contains both 
            left (a_matrix) and right (bounds) hand matrix of the inequality 

        Returns
        -------
        self: DataConsistency class object
            predicted labels by majority vote algorithm
        """

        # Estimates labels
        labels = self._estimate_labels(X, weak_signals_probas, weak_signals_error_bounds)

        # Fit based on labels generated above
        if train_model is None:
            m, n, k = weak_signals_probas.shape
            self.model = self._mlp_model(X.shape[1], k)
            self.model.fit(X, labels, batch_size=32, epochs=20, verbose=1)
        else:
            self.model = train_model
            try:
                self.model.fit(X.T, labels)
            except:
                print("The mean of the baseline labels is %f" %np.mean(labels))
                sys.exit(1)

        return self

    #################################################
    # Maybe put in utilities ########################
    #################################################
    def _mlp_model(self, dimension, output):
        """ 
            Builds Simple MLP model

            Parameters
            ----------
            :param dimension: amount of input
            :type  dimension: int
            :param output: amount of final states
            :type  output: int

            Returns
            -------
            :returns: Simple MLP 
            :return type: Sequential tensor model
        """

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(dimension,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adagrad', metrics=['accuracy'])

        return model