import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense


def calculate_bounds(true_labels, predicted_labels, mask=None):
    """ 
        Calculate error rate on data points the weak signals label 

        Parameters
        ----------
        :param true_labels: correct labeling of the data set
        :type  true_labels: ndarray
        :param predicted_labels: labels estimated by some algorithm
        :type  predicted_labels: ndarray
        :param mask: hyper parameter
        :type  mask: none or passed in ndarray

        Returns
        -------
        :returns: calculated error bounds
        :return type: list of size num_weak x num_classes
    """

    if len(true_labels.shape) == 1:
        predicted_labels = predicted_labels.ravel()
    assert predicted_labels.shape == true_labels.shape

    if mask is None:
        mask = np.ones(predicted_labels.shape)
    if len(true_labels.shape) == 1:
        mask = mask.ravel()

    error_rate = true_labels*(1-predicted_labels) + \
        predicted_labels*(1-true_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.sum(error_rate*mask, axis=0) / np.sum(mask, axis=0)
        error_rate = np.nan_to_num(error_rate)

    # check results are scalars
    if np.isscalar(error_rate):
        error_rate = np.asarray([error_rate])
    return error_rate

def get_error_bounds(true_labels, weak_signals):
    """ 
        Get error bounds of the weaks signals

        Parameters
        ----------
        :param true_labels: correct labeling of the data set
        :type  true_labels: ndarray
        :param error_bounds: weak signals corresponding to the data set
        :type  error_bounds: ndarray

        Returns
        -------
        :returns: calculated error bounds
        :return type: list of size num_weak x num_classes
    """
    error_bounds = []
    mask = weak_signals >= 0

    for i, weak_probs in enumerate(weak_signals):
        active_mask = mask[i]
        error_rate = calculate_bounds(true_labels, weak_probs, active_mask)
        error_bounds.append(error_rate)
    return error_bounds


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

def majority_vote_signal(weak_signals):
    """ 
        Calculate majority vote labels for the weak_signals

        Parameters
        ----------
        :param datapath: weak signals of data set
        :type  datapath: ndarray

        Returns
        -------
        :returns: estimated labels
        :return type: ndarray
    """

    baseline_weak_labels = np.rint(weak_signals)
    mv_weak_labels = np.ones(baseline_weak_labels.shape)
    mv_weak_labels[baseline_weak_labels == -1] = 0
    mv_weak_labels[baseline_weak_labels == 0] = -1
    mv_weak_labels = np.sign(np.sum(mv_weak_labels, axis=0))
    break_ties = np.random.randint(2, size=int(np.sum(mv_weak_labels == 0)))
    mv_weak_labels[mv_weak_labels == 0] = break_ties
    mv_weak_labels[mv_weak_labels == -1] = 0
    return mv_weak_labels


# def mlp_model(dimension, output):
#     """ 
#         Builds Simple MLP model

#         Parameters
#         ----------
#         :param dimension: amount of input
#         :type  dimension: int
#         :param output: amount of final states
#         :type  output: int

#         Returns
#         -------
#         :returns: Simple MLP 
#         :return type: Sequential tensor model
#     """

#     model = Sequential()
#     model.add(Dense(512, activation='relu', input_shape=(dimension,)))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(output, activation='sigmoid'))

#     model.compile(loss='binary_crossentropy',
#                   optimizer='adagrad', metrics=['accuracy'])

#     return model




""" bellow are NOT used currently """

def prepare_mmce(weak_signals, labels):
    """ 
        Convert weak_signals to format for mmce 
    """

    crowd_labels = np.zeros(weak_signals.shape)
    true_labels = labels.copy()
    try:
        n, k = true_labels.shape
    except:
        k = 1
    crowd_labels[weak_signals == 1] = 2
    crowd_labels[weak_signals == 0] = 1
    if k > 1:
        true_labels = np.argmax(true_labels, axis=1)
    true_labels += 1

    if len(crowd_labels.shape) > 2:
        assert crowd_labels.any() != 0
        m, n, k = crowd_labels.shape
        if k > 1:
            for i in range(k):
                crowd_labels[:, :, i] = i+1
            crowd_labels[weak_signals == -1] = 0
        crowd_labels = crowd_labels.transpose((1, 0, 2))
        crowd_labels = crowd_labels.reshape(n, m*k)
    return crowd_labels.astype(int), true_labels.ravel().astype(int)
