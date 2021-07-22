import sys # base class
import random
import numpy as np # baseclass

from BaseClassifier import BaseClassifier
from LabelEstimators import LabelEstimator, CLL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score # base class
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from scipy.optimize import minimize

from log import Logger
from setup_model import set_up_constraint, mlp_model
from train_stochgall import run_constraints


"""


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

            self.logger = Logger("logs/results/" + log_name)      # this can be modified to include date and time in file name
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

        # # FIX LATTER
        # try:
        #     y = self.weights * X
        # except:
        #     y = X * self.weights

        # FIX LATTER
        try:
            y = self.weights.dot(X)
        except:
            y = X.dot(self.weights)
        

        # replacing logistic func for now
        y_squish = 1 / (1 + np.exp(-y))
        grad = y_squish * (1 - y_squish)

        # # FIX THIS LATTER
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


            # FIX THIS LATTER
            weights_gradient = dp_dw.T.dot(dl_dp)

            # print("\n\nHere:")
            # # print("\n\nweights_gradient:", weights_gradient)
            # print("weights_gradient shape:",weights_gradient.shape)
            # print("self.weights shape:",self.weights.shape)
            # print("self.weights shape:", (self.weights * weights_gradient[:, None]).shape )
            


            # update weights of the learnable functions
            self.weights = self.weights - lr * weights_gradient[:, None]

            
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
                
                if self.logger is not None:
                    self.logger.log_scalar("Primal Objective", primal_objective, t)
                    self.logger.log_scalar("lagrangian", lagrangian_obj, t)
                    self.logger.log_scalar("Change in y", conv_y, t)
                    self.logger.log_scalar("Change in Weights", conv_weights, t)
                    self.logger.log_scalar("Ineq Infeas", ineq_infeas, t)



            learnable_probas = self.predict_proba(X)


            # print("here", learnable_probas.shape)
            # print("\n\n")
            # exit()

            t += 1


        return self

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds):
        """
        Fits the model according to given training data (X)

        Parameters
        ----------
        X : ndarray of shape (n_examples, n_features)      
            Training matrix, where n_examples is the number of examples and 
            n_features is the number of features for each example

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

        # reshap multi class datasets into singular one
        if len(weak_signals_probas.shape) == 3:

            m, n, k = weak_signals_probas.shape

            assert k == 1, "Can't handel multi class datasets"
            weak_signals_probas = weak_signals_probas.reshape(m, n)
        
        # reshap weak_signals_error_bounds into singular one
        if len(weak_signals_error_bounds.shape) == 2:

            m, n = weak_signals_error_bounds.shape
            weak_signals_error_bounds = weak_signals_error_bounds.reshape(m)


        self.weights = np.zeros([X.shape[1], 1]) # this should be length of n_features 
   
        self.train_data = X
        n_examples = X.shape[0]

        # initializing algo vars
        y = 0.5 * np.ones(n_examples)
        gamma = np.zeros(weak_signals_probas.shape[0])

        gamma = np.zeros(weak_signals_probas.shape[0])
        one_vec = np.ones(n_examples)
        rho = 2.5
        lr = 0.0001

        learnable_probas = self.predict_proba(X)


        # Check for abstaining signals 
        # active_signal = weak_signals_probas >= 0
        # for i in weak_signals_probas:
        #     if weak_signals_probas[i] == -1:
        #         print("found -1" )

        # weak_signals_probas = weak_signals_probas * active_signal
        # constraint_set['weak_signals'] = weak_signals_probas[:num_weak_signals, :, :] * active_signals[:num_weak_signals, :, :]


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
            
        # return self







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

        to_return = self.model.predict(X)
        
        return to_return.flatten()

    

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds):
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



        " to make up for active_signals"
        active_signals = weak_signals_probas >= 0
        # active_signals = weak_signals_probas[:num_weak_signals, :] >= 0

        " to make up for weak_signals_precision, need to make optional or fix later"
        # FIX LATTER
        # n, m = weak_signals_error_bounds.shape
        weak_signals_precision = np.zeros(weak_signals_error_bounds.shape)

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

            if epoch % 1 == 0:
                for batch in batches:
                    self.model.train_on_batch(X[batch], y[batch])
                learnable_probabilities = self.model.predict(X)
            
            if epoch % 2 == 0:
                y, constraint_set = run_constraints(y, learnable_probabilities, rho, constraint_set, iters=10, enable_print=False)
            
            epoch += 1

        return self






















# # # # # # # # # # # #
# Compartitive modesl #
# # # # # # # # # # # #






        