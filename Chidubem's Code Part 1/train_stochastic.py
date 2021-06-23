import random
import numpy as np
import pandas as pd
import codecs, datetime, glob, itertools, os
import re, sklearn, string, sys, tensorflow, time
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras import backend as K, regularizers, optimizers
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.layers import  Activation, Dropout, Flatten, Dense, InputLayer
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import optimizers
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


def objective_function(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma):
    """
    Computes the value of the objective function

    :param y: vector of estimated labels for the data
    :type y: array
    :param learnable_probabilities: size n_data_points of probabilities for the learnable classifiers
    :type learnable_probabilities: ndarray
    :param weak_signal_probabilities: size (n_weak_signals, n_data_points) of probabilities for weak signal classifiers
    :type weak_signal_probabilities: ndarray
    :param weak_signal_ub: vector of upper bound error rate for the weak signals
    :type weak_signal_ub: array
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

    weak_term = np.dot(1 - weak_signal_probabilities, y) + np.dot(weak_signal_probabilities, 1 - y)
    ineq_constraint = (weak_term / n) - weak_signal_ub
    gamma_term = np.dot(gamma.T, ineq_constraint)

    ineq_constraint = ineq_constraint.clip(min=0)
    ineq_augmented_term = (rho/2) * ineq_constraint.T.dot(ineq_constraint)

    return objective + gamma_term - ineq_augmented_term


def logistic(x):
    """
    Squashing function that returns the squashed value and a vector of the
                derivatives of the squashing operation.

    :param x: ndarray of inputs to be squashed
    :type x: ndarray
    :return: tuple of (1) squashed inputs and (2) the gradients of the
                squashing function, both ndarrays of the same shape as x
    :rtype: tuple
    """
    y = 1 / (1 + np.exp(-x))
    grad = y * (1 - y)

    return y, grad


def probability(data, weights):
    """
    Computes the probabilities of the data for the learnable functions

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param weights: size (n_learnable_functions, d) containing vectors of weights for each function
    :type weights: ndarray
    :return: size n prababilities for the learnable function
    :rtype: array
    """

    try:
        y = weights.dot(data)
    except:
        y = data.dot(weights)

    probs, _ = logistic(y)
    return probs


def dp_dw_gradient(data, weights):
    """
    Computes the gradient the probabilities wrt to the weights

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param weights: size  d containing vector of weights for each learnable function
    :type weights: array
    :return: ndarray of size (n_of_features, n) gradients for probability wrt to weight
    :rtype: ndarray
    """

    try:
        y = weights.dot(data)
    except:
        y = data.dot(weights)

    _, grad = logistic(y)
    grad = data * grad
    return grad


def loss_y_and_q(y, weak_signal_probabilities, weak_signal_ub, n):
    """
    Computes the gradient of lagrangian inequality, same as the loss between l(y, q)

    :param y: vector of estimated labels for the data
    :type y: array
    :param weak_signal_probabilities: size (n_weak_signals, n_data_points) of probabilities for weak signal classifiers
    :type weak_signal_probabilities: ndarray
    :param weak_signal_ub: vector of upper bound error rate for the weak signals
    :type weak_signal_ub: array
    param n: size of the full dataset
    :type n: int
    :return: vector of length gamma containing the gradient of gamma
    :rtype: array
    """

    weak_term = np.dot(1 - weak_signal_probabilities, y) + np.dot(weak_signal_probabilities, 1 - y)
    ineq_constraint = (weak_term / n) - weak_signal_ub

    return ineq_constraint


def y_gradient(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma, loss_y_and_q = None):
    """
    Computes the gradient y

    :param y: vector of estimated labels for the data
    :type y: array
    :param learnable_probabilities: size (n_learnable, n_data_points) of probabilities for the learnable classifiers
    :type learnable_probabilities: ndarray
    :param weak_signal_probabilities: size (n_weak_signals, n_data_points) of probabilities for weak signal classifiers
    :type weak_signal_probabilities: ndarray
    :param weak_signal_ub: vector of upper bound error rate for the weak signals
    :type weak_signal_ub: array
    :param rho: Scalar tuning hyperparameter
    :type rho: float
    :param gamma: vector of lagrangian inequality penalty parameters
    :type gamma: array
    :param loss_y_and_q: scalar loss between y and q
    :type loss_y_and_q: float
    :return: vector of length y containing the gradient of y
    :rtype: array
    """
    n = learnable_probabilities.size
    learnable_term = 1 - (2 * learnable_probabilities)
    learnable_term = np.sum(learnable_term, axis=0) / n

    ls_term = 1 - (2 * weak_signal_probabilities)
    gamma_term = np.dot(gamma.T, ls_term) / n

    if loss_y_and_q is None:
        weak_term = np.dot(1 - weak_signal_probabilities, y) + np.dot(weak_signal_probabilities, 1 - y)
        ineq_constraint = (weak_term / n) - weak_signal_ub
    else:
        ineq_constraint = loss_y_and_q

    ineq_constraint = ineq_constraint.clip(min=0)
    ineq_augmented_term = rho * np.dot(ineq_constraint.T, ls_term/n)

    return learnable_term + gamma_term - ineq_augmented_term


def custom_loss(y_true, y_pred):
    '''Implement custom objective loss'''
    # scale preds so that the class probas of each sample sum to 1

    return K.mean(y_pred * (1 - y_true) + (1 - y_pred) * y_true)


def create_logistic_model(dimension):
    '''
    Creates a simple logistic regression function
    '''
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=dimension))
    model.add(Dense(1,  # output dim is 1
                    activation='sigmoid'))  # input dimension = number of features your data has
    model.compile(optimizer='adagrad',
                  loss=custom_loss,
                  metrics=['binary_accuracy'])

    return model


def train_stochastic_all(data, weak_signal_probabilities, weak_signal_ub, max_epochs=50):

    """
    Minimize the weights of the learnable functions

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param weights: size (n_learnable_functions, d) containing vectors of weights for each function
    :type weights: ndarray
    :param weak_signal_probabilities: size (n_weak_signals, n_data_points) of probabilities for weak signal classifiers
    :type weak_signal_probabilities: ndarray
    :param weak_signal_ub: vector of upper bound error rate for the weak signals
    :type weak_signal_ub: array
    :return: tuple of ndarray containing optimized weights for the learnable functions and vector of inequality constraints
    :rtype: tuple
    """

    def check_tolerance(vec_t, vec_t2, dim=None):
        """
        :param vec_t, vec_t: vectors at different timesteps
        :type vector: array
        :param dim: integer indicating which dimension to sum along
        :type dim: int
        :return: boolean value, True if the vectors are equal within a tolerance level
        :rtype: boolean
        """
        tol = 1e-8
        diff = np.linalg.norm(vec_t - vec_t2)
        return diff < tol

    # Initialize variables
    #learnable_probabilities = probability(data, weights)
    n, dimension = data.shape
    model = create_logistic_model(dimension)
    learnable_probabilities = model.predict(data).ravel()
    n = learnable_probabilities.size

    # y = 0.5 * np.ones(n)
    y = np.mean(weak_signal_probabilities, axis=0)
    rho = 2.5
    gamma = np.zeros(weak_signal_probabilities.shape[0])

    one_vec = np.ones(n)
    batch_size = 200
    old_y = y

    full_loss = loss_y_and_q(y, weak_signal_probabilities, weak_signal_ub, n)

    # update gamma
    gamma_grad = full_loss
    gamma = gamma - rho * gamma_grad
    gamma = gamma.clip(max=0)

    epoch = 0
    t = 1
    converged = False
    while not converged and epoch < max_epochs:

        all_inds = list(range(n))
        random.shuffle(all_inds)
        batches = []

        while len(all_inds) > 0:
            batch = []
            while len(all_inds) > 0 and len(batch) < batch_size:
                batch.append(all_inds.pop())
            batches.append(batch)

        # train on minibatches
        for batch in batches:
            rate = 1 / np.sqrt(1 + t)

            old_batch_loss = loss_y_and_q(y[batch], weak_signal_probabilities[:, batch], weak_signal_ub, n)

            old_y = y
            old_gamma = gamma

            # update y
            y_grad = y_gradient(y[batch], learnable_probabilities[batch],
                        weak_signal_probabilities[:, batch], weak_signal_ub, rho, gamma, full_loss)
            y[batch] = y[batch] + rate * y_grad
            # clip y to [0, 1]
            y = y.clip(min=0, max=1)
            # compute gradient of probabilities
            # dl_dp = (1 / len(batch)) * (1 - 2 * y[batch])

            new_batch_loss = loss_y_and_q(y[batch], weak_signal_probabilities[:, batch], weak_signal_ub, n)

            full_loss = full_loss - old_batch_loss + new_batch_loss
            # update gamma
            gamma_grad = full_loss
            gamma = gamma - rho * gamma_grad
            gamma = gamma.clip(max=0)

            #learnable_probabilities[batch] = probability(data[:, batch], weights)
            model.train_on_batch(data[batch], y[batch])
            learnable_probabilities[batch] = model.predict(data[batch]).ravel()
            t += 1

        epoch += 1
        conv_y = np.linalg.norm(y - old_y)
        converged = False

        if epoch % 1 == 0:
            # For debugging
            keras_obj, _ = model.evaluate(data, y, verbose=0)
            lagrangian_obj = objective_function(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma) # might be slow
            objective = np.dot(learnable_probabilities, 1 - y) + np.dot(1 - learnable_probabilities, y)
            objective = np.sum(objective) / n
            print("Epoch %d. Y_Infeas: %f, Ineq Infeas: %f, lagrangian: %f, obj: %f, keras: %f" % (epoch, conv_y,
                                                np.linalg.norm(full_loss), lagrangian_obj, objective, keras_obj))

    print("Converged", converged)
    ineq_constraint = loss_y_and_q(y, weak_signal_probabilities, weak_signal_ub, n)
    print("Inequality constraints", ineq_constraint)
    # print("Weak signal upper bounds: ", weak_signal_ub)
    learnable_probabilities = model.predict(data).ravel()

    return learnable_probabilities, model
