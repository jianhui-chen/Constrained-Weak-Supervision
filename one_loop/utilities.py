import numpy as np

def set_up_constraint(weak_signals, error_bounds):
    """ 
        Set up error constraints for A and b matrices

        Parameters
        ----------
        :param weak_signals: weak signals of data set
        :type  weak_signals: ndarray
        :param error_bounds: weak signals of data set
        :type  error_bounds: ndarray

        Returns
        -------
        :returns: error set with both both left (a_matrix) 
                  and right (bounds) hand matrix of the inequality 
        :return type: dictionary
    """
    constraint_set = dict()
    m, n, k = weak_signals.shape
    # precision_amatrix = np.zeros((m, n, k))
    error_amatrix = np.zeros((m, n, k))
    constants = []

    for i, weak_signal in enumerate(weak_signals):
        active_signal = weak_signal >= 0

        # # optional
        # precision_amatrix[i] = -1 * weak_signal * active_signal / \
        #     (np.sum(active_signal*weak_signal, axis=0) + 1e-8)
        
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

    m, n, k = error_amatrix.shape
    assert (m, k) == bounds.shape, \
        "The constraint matrix shapes don't match"
   
    error_set = dict()
    error_set['A'] = error_amatrix
    error_set['b'] = bounds

    return error_set