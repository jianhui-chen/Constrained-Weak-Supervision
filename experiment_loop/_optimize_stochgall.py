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

    :param y: size (n_data_points, num_class) adversarial labels for the data
    :type y: ndarray
    :param learnable_probabilities: size (n_data_points, num_class) of estimated probabilities for the learnable classifier
    :type learnable_probabilities: ndarray
    :param constraint_set: dictionary containing constraints specified in the constraint_keys
    :type constraint_set: dict
    :param rho: penalty parameter for the augmented lagrangian
    :type rho: float
    :return: scalar value of objective function
    :rtype: float
    """

    gamma_constraint = 0
    augmented_term = 0
    constraint_keys = constraint_set['constraints']
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']

    objective = loss_functions(y, learnable_probabilities, loss)

    for key in constraint_keys:
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

    return objective + gamma_constraint - augmented_term


#def bound_loss(y, a_matrix, active_mask, constant, bounds):
def gamma_gradient(y, a_matrix, constant, bounds):
    """
    Computes the gradient of lagrangian inequality penalty parameters

    :param y: size (n_data_points, num_class) of estimated labels for the data
    :type y: ndarray
    :param a_matrix: size (num_weak, n, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param constant: size (num_weak, n, num_class) of the constant
    :type constant: ndarray
    :param bounds: size (num_weak, num_class) of the bounds for the constraint
    :type bounds: ndarray
    :return: loss of the constraint set wrt adversarial ys
    :rtype: ndarray
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

    :param learnable_probabilities: size (n_data_points, num_class) of probabilities for the learnable classifier
    :type learnable_probabilities: ndarray
    :param a_matrix: size (num_weak, n, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param gamma: vector of lagrangian inequality(upper bound) penalty parameters corresponding to the number of weak signals
    :type gamma: array
    :param rho: penalty parameter for the augmented lagrangian
    :type rho: float
    :return: ndarray of size (n_data_points, num_class) for y gradient
    :rtype: ndarray
    """

    n, k = learnable_probabilities.shape
    augmented_term = 0
    upper_bound_term = 0
    constraint_keys = constraint_set['constraints']
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

    for key in constraint_keys:
        current_constraint = constraint_set[key]
        a_matrix = current_constraint['A']
        bound_loss = current_constraint['bound_loss']
        gamma = current_constraint['gamma']

        for i, current_a in enumerate(a_matrix):
            #constraint = a_matrix[i] * active_mask[i]
            constraint = a_matrix[i]
            upper_bound_term += gamma[i] * constraint
            augmented_term += bound_loss[i].clip(min=0) * constraint

    return obj_grad + upper_bound_term - rho * augmented_term


def optimize(label, predicted_probs, rho, constraint_set, iters=300, enable_print=True, optim='min'):
    # First find a feasible label with adagrad, initialization step

    constraint_keys = constraint_set['constraints']
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

        for key in constraint_keys:

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

            violation = np.linalg.norm(gamma_grad.clip(min=0))
            print_builder += key + "_viol: %.4e "
            print_constraints.append(violation)

            viol_text += key + "_viol: %.4e "
            constraint_viol.append(violation)

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

