import numpy as np 
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense



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

    Parameters
    ----------
    y: ndarray of size (n_data_points, num_class)
        estimated labels for the data

    a_matrix: ndarray of  size (num_weak, n, num_class)
        constraint matrix

    constant: ndarray of size (num_weak, n, num_class)
        constant

    bounds: ndarray of size (num_weak, num_class)
         bounds for the constraint

    Returns
    -------


    Computes the gradient of lagrangian inequality penalty parameters


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






###########################################################################
######################### OTHER MODELS ####################################
###########################################################################

"""
    Maybe move this here from label estimators
"""
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


"""
    Maybe move this here from data_consistency
"""
def _simple_nn(self, dimension, output):
        """ 
        Data consistent model

        Parameters
        ----------
        dimension: list with two valuse [num_examples, num_features]
            first value is number of training examples, second is 
            number of features for each example


        output: int
            number of classes

        Returns
        -------
        mv_weak_labels: ndarray of shape (num_examples, num_class)
            fitted  Data consistancy algorithm
        """

        actv = 'softmax' if output > 1 else 'sigmoid'
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(dimension,)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(output, activation=actv))
        return model