import numpy as np


def build_constraints(a_matrix, bounds):
    # a_matrix left hand matrix of the inequality size: m x n x k type: ndarray
    # bounds right hand vectors of the inequality size: m x n type: ndarray
    # dictionary containing constraint vectors

    m, n, k = a_matrix.shape
    assert (m,k) == bounds.shape, \
    "The constraint matrix shapes don't match"

    constraints = dict()
    constraints['A'] = a_matrix
    constraints['b'] = bounds
    constraints['gamma'] = np.zeros(bounds.shape)

    # temp for now
    constraints['c'] = np.zeros(a_matrix.shape)

    return constraints


def set_up_constraint(weak_probabilities, precision, error_bounds):
    # set up constraints
    # new modifications to account for abstaining weak_signals
    constraint_set = dict()
    m, n, k = weak_probabilities.shape
    precision_amatrix = np.zeros((m, n, k))
    error_amatrix = np.zeros((m, n, k))
    constants = []

    for i, weak_signal in enumerate(weak_probabilities):
        active_signal = weak_signal >=0
        precision_amatrix[i] = -1 * weak_signal * active_signal / (np.sum(active_signal*weak_signal, axis=0) + 1e-8)
        error_amatrix[i] = (1 - 2 * weak_signal) * active_signal

        # error denom to check abstain signals
        error_denom = np.sum(active_signal, axis=0)
        error_amatrix[i] /= error_denom

        # constants for error constraints
        constant = (weak_signal*active_signal) / error_denom
        constants.append(constant)




    # set up error upper bounds constraints
    constants = np.sum(constants, axis=1)
    assert len(constants.shape) == len(error_bounds.shape)
    bounds = error_bounds - constants
    error_set = build_constraints(error_amatrix, bounds)
    constraint_set['error'] = error_set


    # set up precision upper bounds constraints
    bounds = -1 * precision
    precision_set = build_constraints(precision_amatrix, bounds)
    constraint_set['precision'] = precision_set



    return constraint_set

