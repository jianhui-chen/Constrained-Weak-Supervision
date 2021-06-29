# Note for ALL model class –– inherit from sklean base.py?

# Need to consider if we want to create an abstract base class

from log import Logger
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from scipy.optimize import minimize






class BaseClassifier(ABC):
    """
    Abstract Base Class for learning classifiers
    """

    def predict(self, predicted_probas):
        """

        """
        #proba_predictions = self.predict_proba(X)   # the subclass predict_proba will throw an error if not trained

        predictions = np.zeros(predicted_probas.size)
        predictions[predicted_probas > 0.5] =1    # could also implement by rounding
        return predictions
    
    def get_accuracy(self, true_labels, predicted_probas):
        """

        """
        score = accuracy_score(true_labels, self.predict(predicted_probas))
        return score

    #not abstract
    #def predict_proba
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
        
        return probas.ravel()

    @abstractmethod 
    def fit(self, X):
        """

        """
        pass

class ALL(BaseClassifier):
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

    log_name : string, default=None
        Specifies directory name for a logger object.

    """

    def __init__(self, max_iter=10000, log_name=None):
    
        # based on args

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

        # not based on args bc based on feature number
        self.weights = None
        self.weak_signals_probas = None
        self.weak_signals_error_bounds = None
       


    # Following functions beginning with _ may be moved out of class

    def _objective_function(self, y, learnable_probabilities, rho, gamma):
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
        objective = np.dot(learnable_probabilities, 1 - y) + np.dot(1 - learnable_probabilities, y)
        objective = np.sum(objective) / n

        weak_term = np.dot(1 - self.weak_signals_probas, y) + np.dot(self.weak_signals_probas, 1 - y)
        ineq_constraint = (weak_term / n) - self.weak_signals_error_bounds
        gamma_term = np.dot(gamma.T, ineq_constraint)

        ineq_constraint = ineq_constraint.clip(min=0)
        ineq_augmented_term = (rho/2) * ineq_constraint.T.dot(ineq_constraint)

        return objective + gamma_term - ineq_augmented_term

    def _weight_gradient(self, X):
        """
        Computes the gradient the probabilities wrt to the weights

    
        :return: ndarray of size (n_of_features, n) gradients for probability wrt to weight
        :rtype: ndarray
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


    def _gamma_gradient(self, y):
        """
        Computes the gradient of lagrangian inequality penalty parameters

        :param y: vector of estimated labels for the data
        :type y: array
  
        :return: vector of length gamma containing the gradient of gamma
        :rtype: array
        """
        _, n = self.weak_signals_probas.shape

        weak_term = np.dot(1 - self.weak_signals_probas, y) + np.dot(self.weak_signals_probas, 1 - y)
        ineq_constraint = (weak_term / n) - self.weak_signals_error_bounds

        return ineq_constraint


    def _y_gradient(self, y, learnable_probabilities, rho, gamma):
        """
        Computes the gradient y

        See description in objective function for the variables
        :return: vector of length y containing the gradient of y
        :rtype: array
        """
        n = learnable_probabilities.size
        learnable_term = 1 - (2 * learnable_probabilities)
        learnable_term = np.sum(learnable_term, axis=0) / n

        ls_term = 1 - (2 * self.weak_signals_probas)
        gamma_term = np.dot(gamma.T, ls_term) / n

        weak_term = np.dot(1 - self.weak_signals_probas, y) + np.dot(self.weak_signals_probas, 1 - y)
        ineq_constraint = (weak_term / n) - self.weak_signals_error_bounds
        ineq_constraint = ineq_constraint.clip(min=0)
        ineq_augmented_term = rho * np.dot(ineq_constraint.T, ls_term)

        return learnable_term + gamma_term - ineq_augmented_term



    def predict_proba(self, X):     # Note to self: this should replace "probablity" function in train_classifier
        
        if self.weights is None:
            sys.exit("No Data fit")
   
        try: 
            y = self.weights.dot(X)
        except:
            y = X.dot(self.weights)

        probas = 1 / (1 + np.exp(-y))    # first line of logistic, squishes y values
        
        #exit()
        return probas.ravel()
   

    def _optimize(self, X, learnable_probas, y, rho, gamma, n_examples, lr):
        t = 0
        converged = False
        while not converged and t < self.max_iter:
            rate = 1 / (1 + t)

            # update y
            old_y = y
            y_grad = self._y_gradient(y, learnable_probas, rho, gamma)
            y = y + rate * y_grad
            # projection step: clip y to [0, 1]
            y = y.clip(min=0, max=1)

            # compute gradient of probabilities
            dl_dp = (1 / n_examples) * (1 - 2 * old_y)

            # update gamma
            old_gamma = gamma
            gamma_grad = self._gamma_gradient(old_y)
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
            ineq_constraint = self._gamma_gradient(y)
            ineq_infeas = np.linalg.norm(ineq_constraint.clip(min=0))

            converged = np.isclose(0, conv_y, atol=1e-6) and np.isclose(0, ineq_infeas, atol=1e-6) and np.isclose(0, conv_weights, atol=1e-5)

            if t % 1000 == 0:
                lagrangian_obj = self._objective_function(y, learnable_probas, rho, gamma) # might be slow
                primal_objective = np.dot(learnable_probas, 1 - y) + np.dot(1 - learnable_probas, y)
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
        self.weak_signals_probas = weak_signals_probas
        self.weak_signals_error_bounds = weak_signals_error_bounds
        self.train_data = X
        n_examples = X.shape[1]

        # initializing algo vars
        y = 0.5 * np.ones(n_examples)
        gamma = np.zeros(weak_signals_probas.shape[0])
        one_vec = np.ones(n_examples)
        rho = 2.5
        lr = 0.0001

        learnable_probas = self.predict_proba(X)

        if self.logger is None:
            return self._optimize(X, learnable_probas, y, rho, gamma, n_examples, lr)
        else:
            with self.logger.writer.as_default():
                return self._optimize(X, learnable_probas, y, rho, gamma, n_examples, lr)
            
        return self
    

       


class LabelEstimator(BaseClassifier):   # Might want to change the name of Base Classifier?
    """
    Label Estimator + Classifier
    Need to add more on its functionality. 
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
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict_proba(X.T)[:,1]

        

        return probabilities


    #def _estimate_labels(self, )

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, train_model=None, label_model=None): # can change to get these in init
        """
        Option: we can make it so the labels are generated outside of method
            i.e. passed in as y, or change to pass in algo to generate within
            this method
        error_bounds param is not used, but included for consistency
        """
        labels=np.zeros(weak_signals_probas.shape[1]) # no of examples
        # Generate labels
        if label_model is None:
            average_weak_labels = np.mean(weak_signals_probas, axis=0)
            labels[average_weak_labels > 0.5] = 1
            labels[average_weak_labels <= 0.5] = 0
        else:
            labels = label_model.fit(X).predict(X) #this can be changed


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
        