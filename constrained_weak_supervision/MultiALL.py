import numpy as np
import matplotlib.pyplot as plt
#from setup_supervision import accuracy_score, writeToFile
from setup_model import accuracy_score, writeToFile, mlp_model
from scipy.optimize import check_grad
import random, json, sys, gc
#from utilities import projection_simplex
from data_utilities import projection_simplex
from tensorflow.python.keras import backend as K
#from setup_model import convnet_model


def multiclass_loss(y, learnable_probabilities):
    objective = y * (1 - learnable_probabilities)
    gradient = 1 - learnable_probabilities
    return np.sum(objective) / y.shape[0], gradient


def multilabel_loss(y, learnable_probabilities):
    objective = y * (1 - learnable_probabilities) + learnable_probabilities * (1 - y)
    n,k = y.shape
    gradient = (1 - 2 * learnable_probabilities) /(n*k)
    return np.mean(objective), gradient


def quadratic_loss(y, learnable_probabilities):
    objective = (y - learnable_probabilities)**2
    n,k = y.shape
    gradient = 0.5*(y - learnable_probabilities)/(n*k)
    return np.mean(objective), gradient


def crossentropy_loss(y, learnable_probabilities):
    objective = -y * np.log(learnable_probabilities+1e-8)
    n,k = y.shape
    gradient = -np.log(learnable_probabilities+1e-8)/(n*k)
    return np.mean(objective), gradient


def loss_functions(y, learnable_probabilities, loss='crossentropy'):

    if loss == 'multiclass':
        return multiclass_loss(y, learnable_probabilities)

    if loss == 'multilabel':
        return multilabel_loss(y, learnable_probabilities)

    if loss == 'quadratic':
        return multilabel_loss(y, learnable_probabilities)

    return crossentropy_loss(y, learnable_probabilities)


def objective_function(y, learnable_probabilities, constraint_set, rho):
    """
    Computes the value of the objective function
    One weak signal contains k num of weak signals, one for each class k=num_class

    Parameters
    ----------
    y : ndarray o fsize (n_data_points, num_class) 
        adversarial labels for the data

    learnable_probabilities : ndarray size (n_data_points, num_class)
        stimated probabilities for the learnable classifier

    constraint_set : dict
        dictionary containing constraints specified in the constraint_keys

    rho : float
         scalar value of objective function
        
    Returns
    -------
    return : float
        scalar value of objective function
    """

    gamma_constraint = 0
    augmented_term = 0
    # constraint_keys = constraint_set['constraints']
    key='error'
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']

    objective = loss_functions(y, learnable_probabilities, loss)

    # for key in constraint_keys:
    current_constraint = constraint_set[key]
    a_matrix = current_constraint['A']
    bounds = current_constraint['b']
    constant = current_constraint['c']
    gamma = current_constraint['gamma']
    constraint = np.zeros(bounds.shape)

    for i, current_a in enumerate(a_matrix):
        # constraint[i] = np.sum((active_mask[i]*current_a) * y + (constant[i]*active_mask[i]), axis=0)
        constraint[i] = np.sum(current_a * y + constant[i], axis=0)

    constraint = constraint - bounds

    gamma_constraint += np.sum(gamma * constraint)
    augmented_term += (rho / 2) * np.sum(
        constraint.clip(min=0) * constraint.clip(min=0))
    
    # for loop ended here

    return objective + gamma_constraint - augmented_term


#def bound_loss(y, a_matrix, active_mask, constant, bounds):
def gamma_gradient(y, a_matrix, constant, bounds):
    """
    Computes the gradient of lagrangian inequality penalty parameters

    Parameters
    ----------
    y : 
        estimated labels for the data

    a_matrix : ndarray of size (num_weak, n, num_class) 
        constraint matrix
    
    constant : ndarray of size size (num_weak, n, num_class) 
        the constant

    bounds : 


        
    Returns
    -------

    return: ndarray
        loss of the constraint set wrt adversarial ys


    :param y: size (n_data_points, num_class) of estimated labels for the data
    :type y: ndarray
    :param a_matrix: size (num_weak, n, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param constant: size (num_weak, n, num_class) of the constant
    :type constant: ndarray
    :param bounds: size (num_weak, num_class) of the bounds for the constraint
    :type bounds: ndarray
]
    """
    constraint = np.zeros(bounds.shape)
    # n, k = y.shape

    for i, current_a in enumerate(a_matrix):
        #constraint[i] = np.sum((active_mask[i]*current_a) * y + (constant[i]*active_mask[i]), axis=0)
        constraint[i] = np.sum(current_a * y + constant[i], axis=0)
    return constraint - bounds


def y_gradient(learnable_probabilities, constraint_set, rho, y, quadratic=False):
    """
    Computes the gradient y

    Parameters
    ----------
    learnable_probabilities : of size (n_data_points, num_class)
        probabilities for the learnable classifier

    a_matrix : ndarray of size (num_weak, n, num_class) 
        constraint matrix

    gamma : array of size number of weak signala
        penalty parameters corresponding to the number of weak signala

    rho : float
        penalty parameter for the augmented lagrangian
        
    Returns
    -------
    return : ndarray of size (n_data_points, num_class) 
        y gradient
    """

    n, k = learnable_probabilities.shape
    augmented_term = 0
    upper_bound_term = 0
    # constraint_keys = constraint_set['constraints']
    key='error'
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']

    # obj_grad = 1 - learnable_probabilities \
    #                 if loss == 'multiclass' else 1 - 2*learnable_probabilities

    # obj_grad = obj_grad / n if loss == 'multiclass' else obj_grad / (n*k)

    if loss == 'multiclass':
        obj_grad = 1 - learnable_probabilities
        obj_grad = obj_grad / n
    else:
        obj_grad = 1 - 2 * learnable_probabilities
        obj_grad = obj_grad / (n * k)

    # for key in constraint_keys:
    current_constraint = constraint_set[key]
    a_matrix = current_constraint['A']
    bound_loss = current_constraint['bound_loss']
    gamma = current_constraint['gamma']

    for i, current_a in enumerate(a_matrix):
        #constraint = a_matrix[i] * active_mask[i]
        constraint = a_matrix[i]
        upper_bound_term += gamma[i] * constraint
        augmented_term += bound_loss[i].clip(min=0) * constraint
    
    # for loop ended here

    return obj_grad + upper_bound_term - rho * augmented_term


def optimize(label, predicted_probs, rho, constraint_set, iters=300, enable_print=True, optim='min'):
    # First find a feasible label with adagrad, initialization step

    # constraint_keys = constraint_set['constraints']
    key = 'error'
    weak_signals = constraint_set['weak_signals']
    num_weak_signal = constraint_set['num_weak_signals']
    # true_bounds = constraint_set['true_bounds'] # boolean value
    # true_bounds = False
    # true_bounds = True
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']
   
    grad_sum = 0
    y = label.copy()
    n,k = y.shape

    #weak_signals = weak_signals * active_mask

    # get the min weak_signal vector
    min_vector = np.min(weak_signals[:num_weak_signal, :, :], axis=0)
    max_vector = np.max(weak_signals[:num_weak_signal, :, :], axis=0)
    assert y.shape == min_vector.shape

    for iter in range(iters):
        print_constraints = [iter]
        print_builder = "Iteration %d, "
        constraint_viol = []
        viol_text = ''

        # for key in constraint_keys:

        current_constraint = constraint_set[key]

        a_matrix = current_constraint['A']
        bounds = current_constraint['b']
        constant = current_constraint['c']
        gamma = current_constraint['gamma']

        
        gamma_grad = gamma_gradient(y, a_matrix, constant, bounds)

        # gamma_grad = full_loss
        

        if optim == 'max':
            gamma = gamma - rho * gamma_grad
            gamma = gamma.clip(max=0)

        else:
            gamma = gamma + rho * gamma_grad
            gamma = gamma.clip(min=0)
        

        # update constraint values
        constraint_set[key]['gamma'] = gamma
        constraint_set[key]['bound_loss'] = gamma_grad

        # violation = np.linalg.norm(gamma_grad.clip(min=0))
        # print_builder += key + "_viol: %.4e "
        # print_constraints.append(violation)

        # viol_text += key + "_viol: %.4e "
        # constraint_viol.append(violation)

        # this was where for loop ended

        y_grad = y_gradient(predicted_probs, constraint_set, rho, y, quadratic=True)
        grad_sum += y_grad**2


        if optim == 'max':
            y = y + y_grad / np.sqrt(grad_sum + 1e-8)
        else:
            y = y - y_grad / np.sqrt(grad_sum + 1e-8)

        
        # y = np.clip(y, a_min=min_vector, a_max=max_vector)  if not true_bounds \
        #                         else (y if loss == 'multiclass' else np.clip(y, a_min=0, a_max=1))

        # if not true_bounds:
        #     y = np.clip(y, a_min=min_vector, a_max=max_vector)
        # else:
        #     if loss == 'multiclass':    
        #         y = y
        #     else:
        #         y = np.clip(y, a_min=0, a_max=1)        # not multiclass


        # y = projection_simplex(y, axis=1) if loss == 'multiclass' else y

        if loss == 'multiclass':
            y = projection_simplex(y, axis=1)
        else:       #not multiclass
            y = np.clip(y, a_min=0, a_max=1) 

     
        constraint_set['violation'] = [viol_text, constraint_viol]
 
    return y, constraint_set

class ALL(BaseClassifier):
    """
    Adversarial Label Learning Classifier

    This class implements ALL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        For optimize

    """

    def __init__(self, max_iter=300, max_epoch=20, rho=0.1, loss='multilabel', batch_size=32):

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
        X: ndarray of shape (n_examples, n_features)
            Training matrix, where n_examples is the number of examples and
            n_features is the number of features for each example

        weak_signals_probas: ndarray of shape (n_weak_signals, n_examples, n_classes)
            A set of soft or hard weak estimates for data examples.
            This may later be changed to accept just the weak signals, and these
            probabilities will be calculated within the ALL class.

        weak_signals_error_bounds: dictionary
            error constraints (a_matrix and bounds) of the weak signals. Contains both
            left (a_matrix) and right (bounds) hand matrix of the inequality

        Returns
        -------
        self
            Fitted estimator

        """

        # original variables
        # constraint_keys = ["error"]
        num_weak_signals = weak_signals_probas.shape[0]

        active_signals = weak_signals_probas >= 0
        # active_signals = weak_signals_probas[:num_weak_signals, :] >= 0

        # todo: fix this to allow precision
        weak_signals_precision = np.zeros(weak_signals_probas.shape)

        constraint_set = {}
        constraint_set['error'] = weak_signals_error_bounds
        # constraint_set['constraints'] = constraint_keys
        constraint_set['weak_signals'] = weak_signals_probas[:num_weak_signals, :, :] * active_signals[:num_weak_signals, :, :]
        constraint_set['num_weak_signals'] = num_weak_signals
        constraint_set['loss'] = self.loss

        # Code for fitting algo
        results = dict()

        m, n, k = constraint_set['weak_signals'].shape

        m = 2 if k == 1 else k

        # initialize final values
        learnable_probabilities = np.ones((n, k)) * 1/m
        y = np.ones((n, k)) * 0.1
        assert y.shape[0] == X.shape[0]

        # initialize hyperparams todo: can this be in init?
        rho = 0.1
        loss = 'multilabel'
        batch_size = 32

        # This is to prevent the learning algo from wasting effort fitting a model to arbitrary y values.
        y, constraint_set = optimize(y, learnable_probabilities, rho, constraint_set, optim='max')

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
                y, constraint_set = optimize(y, learnable_probabilities, rho, constraint_set, iters=10, enable_print=False)

            epoch += 1

        return self


    def get_score(self, true_labels, predicted_probas, metric='accuracy'):
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
        score = super.get_score(true_labels, predicted_probas, metric)
        return score


    def predict_proba(self, X):
        """
        Computes probabilistic labels for the training data

        Parameters
        ----------
        X : ndarray of training data


        Returns
        -------
        probas : ndarray of label probabilities

        """

        return self.model(X)


    def predict(self, X):
        """
        Computes predicted classes for the training data.

        Parameters
        ----------
        X : ndarray of training data

        Returns
        -------
        predicted classes : ndarray array of predicted classes
        """
        proba = self.predict_proba(X)
        return np.argmax(predicted_probas, axis=-1)


