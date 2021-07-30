import numpy as np # baseclass




"""
in...
    - _optimize_stochgall.py
    - ALL_modle.py
    - maybe old_ALL????
    
decide a name for it... bound_loss or gamma_gradient
"""
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