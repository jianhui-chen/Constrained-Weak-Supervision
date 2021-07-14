import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from scipy.optimize import minimize

from log import Logger
from setup_model import set_up_constraint
from train_stochgall import *


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







# # # # # # # # # # # #
#      Our models     # 
# # # # # # # # # # # #

class ALL(BaseClassifier):
    """
    Adversarial Label Learning Classifier

    This class implements ALL training on a set of data

        Potentially the following methods can be moved out of this class:
            _objective_function
            _gamma_gradient
            _y_gradient

    Parameters
    ----------
    max_iter : int, default=10000
        Maximum number of iterations taken for solvers to converge.

    log_name : string, default=None
        Specifies directory name for a logger object.

    """

    def __init__(self, max_iter=10000, log_name=None):

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            """
            self.logger = Logger("logs/ALL/" + log_name + "/" + 
                                 str(weak_signals_proba.shape[0]) + 
                                 "_weak_signals/")      # this can be modified to include date and time in file name
            """
        else:
            sys.exit("Not of string type")

        self.weights = None

       

    # Following functions beginning with _ may be moved out of class

    def _objective_function(self, y, weak_signals_probas, 
                            weak_signals_error_bounds, learnable_probabilities, 
                            rho, gamma):
        """
        Computes the value of the objective function

        :param y: vector of estimated labels for the data
        :type y: array, size n
        :param learnable_probabilities: estimated probabilities for the classifier
        :type learnable_probabilities: array, size n
        :param rho: Scalar tuning hyperparameter
        :type rho: float
        :param gamma: vector of lagrangian inequality penalty parameters
        :type gamma: array
        :return: scalar value of objective function
        :rtype: float
        """

        n = learnable_probabilities.size
        objective = np.dot(learnable_probabilities, 1 - y) + \
                    np.dot(1 - learnable_probabilities, y)
        objective = np.sum(objective) / n

        weak_term = np.dot(1 - weak_signals_probas, y) + \
                    np.dot(weak_signals_probas, 1 - y)
        ineq_constraint = (weak_term / n) - weak_signals_error_bounds
        gamma_term = np.dot(gamma.T, ineq_constraint)

        ineq_constraint = ineq_constraint.clip(min=0)
        ineq_augmented_term = (rho/2) * ineq_constraint.T.dot(ineq_constraint)

        return objective + gamma_term - ineq_augmented_term

    def _weight_gradient(self, X):
        """
        Computes the gradient the probabilities wrt to the weights

        See description in fit() for the Parameters

        Returns
        -------
        ndarray of size (n_of_features, n) gradients for probability wrt to weight
  
        """

        try:
            y = self.weights.dot(X)
        except:
            y = X.dot(self.weights)

        # replacing logistic func for now
        y_squish = 1 / (1 + np.exp(-y))
        grad = y_squish * (1 - y_squish)

        grad = X * grad
        return grad


    def _gamma_gradient(self, y, weak_signals_probas, weak_signals_error_bounds):
        """
        Computes the gradient of lagrangian inequality penalty parameters

        See description in fit() for the Parameters

        Returns
        -------
        vector of length gamma containing the gradient of gamma

        """
        _, n = weak_signals_probas.shape

        weak_term = np.dot(1 - weak_signals_probas, y) + \
                    np.dot(weak_signals_probas, 1 - y)
        ineq_constraint = (weak_term / n) - weak_signals_error_bounds

        return ineq_constraint


    def _y_gradient(self, y, weak_signals_probas, weak_signals_error_bounds, 
                    learnable_probabilities, rho, gamma):
        """
        Computes the gradient y

        See description in objective function for the Parameters

        Returns
        -------
        Gradient of y
        """
        n = learnable_probabilities.size
        learnable_term = 1 - (2 * learnable_probabilities)
        learnable_term = np.sum(learnable_term, axis=0) / n

        ls_term = 1 - (2 * weak_signals_probas)
        gamma_term = np.dot(gamma.T, ls_term) / n

        weak_term = np.dot(1 - weak_signals_probas, y) + \
                    np.dot(weak_signals_probas, 1 - y)
        ineq_constraint = (weak_term / n) - weak_signals_error_bounds
        ineq_constraint = ineq_constraint.clip(min=0)
        ineq_augmented_term = rho * np.dot(ineq_constraint.T, ls_term)

        return learnable_term + gamma_term - ineq_augmented_term



    def _optimize(self, X, weak_signals_probas, weak_signals_error_bounds, 
                  learnable_probas, y, rho, gamma, n_examples, lr):
        """
        Optimizes model according to given training data (X)

        Parameters
        ----------
        X : view comments for fit()

        weak_signals_proba : view comments for fit()

        weak_signals_error_bounds : view comments for fit()

        learnable_probas : ndarray of size (n_examples,)
            Estimated probabilities for data

        y : ndarray of size (n_examples,)
            Estimated labels for data

        rho : float
            Scalar tuning hyperparameter

        gamma : ndarray of size (n_weak_signals,)
            Lagrangian inequality penalty parameters

        n_examples : int
            Denotes number of examples

        lr : float


        Returns
        -------
        self
            Fitted and optimized estimator

        """
        t = 0
        converged = False
        while not converged and t < self.max_iter:
            rate = 1 / (1 + t)

            # update y
            old_y = y
            y_grad = self._y_gradient(y, weak_signals_probas, 
                                      weak_signals_error_bounds, 
                                      learnable_probas, rho, gamma)
            y = y + rate * y_grad
            # projection step: clip y to [0, 1]
            y = y.clip(min=0, max=1)

            # compute gradient of probabilities
            dl_dp = (1 / n_examples) * (1 - 2 * old_y)

            # update gamma
            old_gamma = gamma
            gamma_grad = self._gamma_gradient(old_y, weak_signals_probas, 
                                              weak_signals_error_bounds)
            gamma = gamma - rho * gamma_grad
            gamma = gamma.clip(max=0)

            weights_gradient = []
            # compute gradient of probabilities wrt weights
            dp_dw = self._weight_gradient(X)
            # update weights
            old_weights = self.weights.copy()
            weights_gradient.append(dp_dw.dot(dl_dp))

            # update weights of the learnable functions
            self.weights = self.weights - lr * np.array(weights_gradient)
            conv_weights = np.linalg.norm(self.weights - old_weights)
            conv_y = np.linalg.norm(y - old_y)

            # check that inequality constraints are satisfied
            ineq_constraint = self._gamma_gradient(y, weak_signals_probas, 
                                                   weak_signals_error_bounds)
            ineq_infeas = np.linalg.norm(ineq_constraint.clip(min=0))

            converged = np.isclose(0, conv_y, atol=1e-6) and \
                        np.isclose(0, ineq_infeas, atol=1e-6) and \
                        np.isclose(0, conv_weights, atol=1e-5)

            if t % 1000 == 0:
                lagrangian_obj = self._objective_function(y, weak_signals_probas, weak_signals_error_bounds, learnable_probas, rho, gamma) # might be slow
                primal_objective = np.dot(learnable_probas, 1 - y) + \
                                   np.dot(1 - learnable_probas, y)
                primal_objective = np.sum(primal_objective) / n_examples
                # print("Iter %d. Weights Infeas: %f, Y_Infeas: %f, Ineq Infeas: %f, lagrangian: %f, obj: %f" % (t, np.sum(conv_weights), conv_y,
                # 									ineq_infeas, lagrangian_obj, primal_objective))
                
                if self.logger is not None:
                    self.logger.log_scalar("Primal Objective", primal_objective, t)
                    self.logger.log_scalar("lagrangian", lagrangian_obj, t)
                    self.logger.log_scalar("Change in y", conv_y, t)
                    self.logger.log_scalar("Change in Weights", conv_weights, t)

                    # So small, does not even register????
                    self.logger.log_scalar("Ineq Infeas", ineq_infeas, t)



            learnable_probas = self.predict_proba(X)

            t += 1
        return self

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds):
        """
        Fits the model according to given training data (X)

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)      
            Training matrix, where n_examples is the number of examples and 
            n_features is the number of features for each example

            NOTE: Usually this is transposed, might want to change for 
            consistency with other models

        weak_signals_proba : ndarray of shape (n_weak_signals, n_examples)
            A set of soft or hard weak estimates for data examples.
            This may later be changed to accept just the weak signals, and these 
            probabilities will be calculated within the ALL class. 

        weak_signals_error_bounds : ndarray of shape (n_weak_signals,)
            Stores upper bounds of error rates for each weak signal.

        Returns
        -------
        self
            Fitted estimator

        """
        self.weights = np.zeros(X.shape[0]) # this should be length of n_features
   
        self.train_data = X
        n_examples = X.shape[1]

        # initializing algo vars
        y = 0.5 * np.ones(n_examples)
        gamma = np.zeros(weak_signals_probas.shape[0])

        gamma = np.zeros(weak_signals_probas.shape[0])
        one_vec = np.ones(n_examples)
        rho = 2.5
        lr = 0.0001

        learnable_probas = self.predict_proba(X)

        if self.logger is None:
            return self._optimize(X, weak_signals_probas, 
                                  weak_signals_error_bounds, learnable_probas, 
                                  y, rho, gamma, n_examples, lr)
        else:
            with self.logger.writer.as_default():
                return self._optimize(X, weak_signals_probas, 
                                      weak_signals_error_bounds, 
                                      learnable_probas, y, rho, gamma, 
                                      n_examples, lr)
            
        return self







""" Multi-ALL class """

class MultiALL(BaseClassifier):
    """
    Multi Class Adversarial Label Learning Classifier

    This class implements Multi ALL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        For run_constraints

    log_name : Can be added, need to deal with some issues with imports

    """

    def __init__(self, max_iter=10, max_epoch=20, rho=0.1, loss='multilabel', batch_size=32):
    
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.model = None
        self.rho = rho
        self.loss = loss
        self.batch_size = batch_size


    def predict_proba(self, X):
        if self.model is None:
            sys.exit("no model")

        return self.model.predict(X)

    

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, weak_signals_precision, active_signals):
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

        # original variables
        constraint_keys = ["error"]
        num_weak_signals = weak_signals_probas.shape[0]

        constraint_set = set_up_constraint(weak_signals_probas[:num_weak_signals, :, :],
                                           weak_signals_precision[:num_weak_signals, :],
                                           weak_signals_error_bounds[:num_weak_signals, :])
        
        constraint_set['constraints'] = constraint_keys
        constraint_set['weak_signals'] = weak_signals_probas[:num_weak_signals, :, :] * active_signals[:num_weak_signals, :, :]
        constraint_set['num_weak_signals'] = num_weak_signals
        constraint_set['loss'] = self.loss

        # Code for fitting algo
        results = dict()

        m, n, k = constraint_set['weak_signals'].shape

        m = 2 if k == 1 else k

        # initialize final values
        learnable_probabilities = np.ones((n,k)) * 1/m
        y = np.ones((n,k)) * 0.1
        assert y.shape[0] == X.shape[0]

        # initialize hyperparams -- CAN THIS BE IN INIT
        rho = 0.1
        loss = 'multilabel'
        batch_size = 32

        # This is to prevent the learning algo from wasting effort fitting a model to arbitrary y values.
        y, constraint_set = run_constraints(y, learnable_probabilities, rho, constraint_set, optim='max')

        self.model = mlp_model(X.shape[1], k)

        grad_sum = 0
        epoch = 0
        while epoch < self.max_epoch:
            indices = list(range(n))
            random.shuffle(indices)
            batches = np.array_split(indices, int(n / batch_size))

            rate = 1.0
            old_y = y.copy()

            # Currently leaving out print statements

            if epoch % 1 == 0:
                for batch in batches:
                    self.model.train_on_batch(X[batch], y[batch])
                learnable_probabilities = self.model.predict(X)
            
            if epoch % 2 == 0:
                y, constraint_set = run_constraints(y, learnable_probabilities, rho, constraint_set, iters=10, enable_print=False)

                #Leaving out print stuff for now
            
            epoch += 1

        return self





class CLL(BaseClassifier):
    """
    Constrained label learning Classifier

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

            # log current data 
            if self.logger is not None and iter % 10 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("y", np.average(y), iter)
                    self.logger.log_scalar("y_grad", np.average(y_grad), iter)
                    self.logger.log_scalar("loss", np.average(loss), iter)
                    self.logger.log_scalar("violation", np.average(violation), iter)
        return y
    

    def _estimate_labels(self, weak_signals_probas, weak_signals_error_bounds):
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

        # initialize y and lists
        y = np.random.rand(n, k)
        ys = []

        for i in range(self.num_trials):
            ys.append( self._run_constraints(y, weak_signals_error_bounds) )
        return np.mean(ys, axis=0)    


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



    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, train_model=None):
        """
        Finds estimated labels

        Parameters
        ----------
        :param X: current data examples to fit model with
        :type  X: ndarray 
        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for each example
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error constraints (a_matrix and bounds) of the weak signals. Contains both 
                                          left (a_matrix) and right (bounds) hand matrix of the inequality 
        :type  weak_signals_error_bounds: dictionary 

        Returns
        -------
        :return: average of learned labels over several trials
        :rtype: ndarray
        """

        # Estimates labels
        labels = self._estimate_labels(weak_signals_probas, weak_signals_error_bounds)

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

    
















# # # # # # # # # # # #
# Compartitive modesl #
# # # # # # # # # # # #


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
        elif type(log_name) is str:
            self.logger = Logger("logs/Baseline/" + log_name + "/" + 
                                 str(weak_signals_proba.shape[0]) + 
                                 "_weak_signals/")      # this can be modified to include date and time in file name
        else:
            sys.exit("Not of string type")

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


    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, 
            train_model=None): 
        """
        Option: we can make it so the labels are generated outside of method
            i.e. passed in as y, or change to pass in algo to generate within
            this method
        error_bounds param is not used, but included for consistency
        """

        # Estimates labels
        labels = self._estimate_labels(weak_signals_probas, weak_signals_error_bounds)


        # Fit based on labels generated above
        if train_model is None:
            self.model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        else:
            self.model = train_model

        try:
            self.model.fit(X.T, labels)
        except:
            print("The mean of the baseline labels is %f" %np.mean(labels))
            sys.exit(1)
        return self


class GECriterion(BaseClassifier):
    """
    GE Criterion class for implementation
    """

    def __init__(self, max_iter=None, log_name=None):
        # based on args

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            """
            self.logger = Logger("logs/Baseline/" + log_name + "/" + 
                                 str(weak_signals_proba.shape[0]) + 
                                 "_weak_signals/")      # this can be modified to include date and time in file name
            """
        else:
            sys.exit("Not of string type")

        # not based on args bc based on feature number
        self.weights = None
    
    # Uses prediction stuff in parent class

    def _compute_reference_distribution(self, y, weak_signal_proba):
        """
        Computes the score value of the reference expectation

        :param labels: size n labels for each instance in the dataset
        :type labels: array
        :param weak_signal: weak signal trained using one dimensional feature
        :type  weak_signal: array
        :return: tuple containing scalar values of positive and negative reference probability distribution
        :rtype: float
        """
        threshold = 0.5
        positive_index = np.where(weak_signal_proba >= threshold)
        negative_index = np.where(weak_signal_proba < threshold)
        pos_feature_labels = y[positive_index]
        neg_feature_labels = y[negative_index]

        try:
            with np.errstate(all='ignore'):
                reference_pos_probability = np.sum(pos_feature_labels) / pos_feature_labels.size
                reference_neg_probability = np.sum(neg_feature_labels) / neg_feature_labels.size
        except:
            reference_pos_probability = np.nan_to_num(np.sum(pos_feature_labels) / pos_feature_labels.size) + 0
            reference_neg_probability = np.nan_to_num(np.sum(neg_feature_labels) / neg_feature_labels.size) + 0

        return reference_pos_probability, reference_neg_probability

    def _compute_empirical_distribution(self, est_probability, weak_signal_proba):
        """
        Computes the score value of the emperical distribution

        :param est_probability: size n estimated probabtilities for the instances
        :type labels: array
        :param weak_signal_proba: weak signal trained using one dimensional feature
        :type  weak_signal: array
        :return: (tuple of scalar values of the empirical distribution, tuple of index of instances)
        :rtype: tuple
        """
        threshold = 0.5
        positive_index = np.where(weak_signal_proba >= threshold)
        negative_index = np.where(weak_signal_proba < threshold)
        pos_feature_labels = est_probability[positive_index]
        neg_feature_labels = est_probability[negative_index]

        try:
            with np.errstate(all='ignore'):
                empirical_pos_probability = np.sum(pos_feature_labels) / pos_feature_labels.size
                empirical_neg_probability = np.sum(neg_feature_labels) / neg_feature_labels.size
        except:
            empirical_pos_probability = np.nan_to_num(np.sum(pos_feature_labels) / pos_feature_labels.size) + 0
            empirical_neg_probability = np.nan_to_num(np.sum(neg_feature_labels) / neg_feature_labels.size) + 0

        empirical_probability = empirical_pos_probability, empirical_neg_probability
        instances_index = positive_index, negative_index
        return empirical_probability, instances_index

    def _train_ge_criteria(self, X, y, new_weights, weak_signals_probas):
        """
        This internal function returns the objective value of ge criteria

        :param new_weights: weights to use for computing multinomial logistic regression
        :type new_weights: ndarray
        :return: tuple containing (objective, gradient)
        :rtype: (float, array)
        """

        obj = 0
        score = X.dot(new_weights)
        #probs, grad = logistic(score)
        probs = 1 / (1 + np.exp(-score))
        grad = probs * (1 - probs)
        gradient = 0
        num_weak_signals = weak_signals_probas.shape[0]
        # Code to compute the objective function
        for i in range(num_weak_signals):
            weak_signal_proba = weak_signals_probas[i]
            reference_probs = self._compute_reference_distribution(y, weak_signal_proba)
            empirical_probs, index = self._compute_empirical_distribution(probs, weak_signal_proba)

            # empirical computations
            pos_empirical_probs, neg_empirical_probs = empirical_probs
            pos_index, neg_index = index

            # reference computations
            pos_reference_probs, neg_reference_probs = reference_probs

            try:
                with np.errstate(all='ignore'):
                    # compute objective for positive probabilities
                    obj += pos_reference_probs * np.log(pos_reference_probs / pos_empirical_probs)
                    gradient += (pos_reference_probs / pos_empirical_probs) * X[pos_index].T.dot(grad[pos_index]) / grad[pos_index].size

                    # compute objective for negative probabilities
                    obj += neg_reference_probs * np.log(neg_reference_probs / neg_empirical_probs)
                    gradient += (neg_reference_probs / neg_empirical_probs) * X[neg_index].T.dot(grad[neg_index]) / grad[neg_index].size
            except:
                # compute objective for positive probabilities
                obj += np.nan_to_num(pos_reference_probs * np.log(pos_reference_probs / pos_empirical_probs))
                gradient += np.nan_to_num((pos_reference_probs / pos_empirical_probs) * X[pos_index].T.dot(grad[pos_index]) / grad[pos_index].size)

                # compute objective for negative probabilities
                obj += np.nan_to_num(neg_reference_probs * np.log(neg_reference_probs / neg_empirical_probs))
                gradient += np.nan_to_num((neg_reference_probs / neg_empirical_probs) * X[neg_index].T.dot(grad[neg_index]) / grad[neg_index].size)

        objective = obj + (0.5 * np.sum(new_weights**2))
        gradient = new_weights - gradient

        return objective, gradient


    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, y):
        """
        X is passed in as (d, n), to match the other fits.
        X is thus transposed until this is fixed
        Removed the check gradient param

        error_bounds param is not used, but included for consistency
        """

        if y is None:
            sys.exit("GE Criterion requires labels")

        X = X.T 
        n, d = X.shape
        self.weights = np.random.rand(d)

        # optimizer
        res = minimize(lambda w: self._train_ge_criteria(X, y, w, weak_signals_probas)[0], jac=lambda w: self._train_ge_criteria(X, y, w, weak_signals_probas)[1].ravel(), x0=self.weights) 
        self.weights = res.x

        return self
        