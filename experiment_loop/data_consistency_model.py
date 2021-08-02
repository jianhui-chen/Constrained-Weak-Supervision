import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

from LabelEstimator import LabelEstimator
from log import Logger



class DataConsistency(LabelEstimator):
    """
    Data Consistency learning Classifier

    This class implements Data Consistency training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
       max number of iterations to train model

    max_stagnation : int, default=100
        number of epochs without improvement to tolerate
    
    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=1000, max_stagnation=100, log_name=None,):

        self.max_iter = max_iter 
        self.max_stagnation = max_stagnation

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name)
        else:
            sys.exit("Not of string type")        

        self.model = None
    
    #################################################
    # Maybe put in utilities ########################
    #################################################

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

    
    #################################################
    # Maybe put in utilities ########################
    #################################################
    
    def _majority_vote_signal(self, weak_signals_probas):
        """ 
        Calculate majority vote labels for the weak_signals

        Parameters
        ----------
        weak_signals: ndarray of shape (num_weak, num_examples, num _class)
            weak signal probabilites containing -1 for abstaining signals, and between 
            0 to 1 for non-abstaining

        Returns
        -------
        mv_weak_labels: ndarray of shape (num_examples, num_class)
            fitted  Data consistancy algorithm
        """

        baseline_weak_labels = np.rint(weak_signals_probas)
        mv_weak_labels = np.ones(baseline_weak_labels.shape)

        # Why is it flipping these??? 
        mv_weak_labels[baseline_weak_labels == -1] = 0
        mv_weak_labels[baseline_weak_labels == 0] = -1

        mv_weak_labels = np.sign(np.sum(mv_weak_labels, axis=0))
        break_ties = np.random.randint(2, size=int(np.sum(mv_weak_labels == 0)))
        mv_weak_labels[mv_weak_labels == 0] = break_ties
        mv_weak_labels[mv_weak_labels == -1] = 0
        return mv_weak_labels


    def _consistency_loss(self, model, X, mv_labels, a_matrix, bounds, slack, gamma, C):

        """
        Lagragian loss function

        Parameters
        ----------
        model: Sequential model
         
        X: tensor of shape (num_examples, num_features)
            training data
         
        mv_labels: ndarray of shape (num_examples, num_class)
            predicted labels by majority vote algorithm
         
        a_matrix: ndarray of shape (num_weak_signals, num_examples,  num_class)
            left bounds on the data
        
        bounds: ndarray of shape (num_weak_signals,  num_class)
            right bounds on the data

        slack: tensor of shape (num_weak_signals, num_class)
            linear slack to adaptively relax the constraints

        gamma: tensor of shape (num_weak_signals, num_class)
            gamma constant
        
        C: tf.Tensor(10.0, shape=(), dtype=float32)
            tensor constant

        Returns
        -------
        lagragian_objective: a tensor float32
            the loss value

        constraint_violation: a tensor float32 of shape (1,)
            How much the constraints are currently being violated 
        """

        m, n, k = a_matrix.shape
        lagragian_objective = tf.zeros(k)
        constraint_violation = tf.zeros(k)
        i = 0
        Y = model(X, training=True)

        primal = tf.divide(tf.nn.l2_loss(Y - mv_labels), n)
        primal = tf.add(primal, tf.multiply(C, tf.reduce_mean(slack)))
        for A in a_matrix:
            AY = tf.reduce_sum(tf.multiply(A, Y), axis=0)
            violation = tf.add(bounds[i], slack[i])
            violation = tf.subtract(AY, violation)
            value = tf.multiply(gamma[i], violation)
            lagragian_objective = tf.add(lagragian_objective, value)
            constraint_violation = tf.add(constraint_violation, violation)
            i += 1
        lagragian_objective = tf.add(primal, tf.reduce_sum(lagragian_objective))

        return lagragian_objective, constraint_violation


    # def _estimate_labels(self, model, X, mv_labels, a_matrix, bounds):
    def _estimate_labels(self, X, weak_signals_probas, weak_signals_error_bounds):
        """
        Train DCWS algorithm

        Parameters
        ----------
        model: Sequential model
         
        X: tensor of shape (num_examples, num_features)
            training data
         
        mv_labels: ndarray of shape (num_examples, num_class)
            predicted labels by majority vote algorithm using weak signals
         
        a_matrix: ndarray of shape (num_weak_signals, num_examples,  num_class)
            left bounds on the data
        
        bounds: ndarray of shape (num_weak_signals,  num_class)
            right bounds on the data

        Returns
        -------
        pred_y: tensor of shape (num_examples,  num_class)
            predicted labels for data set
        
        """ 

        # set up variables
        m, n, k = weak_signals_probas.shape
        nn_data = tf.cast(X, dtype=tf.float32)
        model = self._simple_nn(nn_data.shape[1], k)
        a_matrix = weak_signals_error_bounds['A']
        bounds = weak_signals_error_bounds['b']
        mv_labels = self._majority_vote_signal(weak_signals_probas)

        # Set up more variables
        adam_optimizer = tf.keras.optimizers.Adam()
        grad_optimizer = tf.keras.optimizers.SGD()
        best_viol, best_iter = np.inf, self.max_iter
        early_stop = False
        C = tf.constant(10, dtype=tf.float32)

        # Set up tensors
        m, k = bounds.shape
        gamma = np.random.rand(m, k)
        gamma = tf.Variable(gamma.astype(np.float32), trainable=True,
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        slack = np.zeros(bounds.shape, dtype="float32")
        slack = tf.Variable(slack, trainable=True,
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        mv_labels = tf.constant(mv_labels, dtype=tf.float32)

        for iters in range(self.max_iter):
            if early_stop:
                break

            with tf.GradientTape() as tape:
                loss_value, constraint_viol = self._consistency_loss(
                    model, nn_data, mv_labels, a_matrix, bounds, slack, gamma, C)
                model_grad, gamma_grad, slack_grad = tape.gradient(
                    loss_value, [model.trainable_variables, gamma, slack])

            adam_optimizer.apply_gradients(zip(model_grad, model.trainable_variables))
            grad_optimizer.apply_gradients(zip([slack_grad], [slack]))
            grad_optimizer.apply_gradients(zip([-1 * gamma_grad], [gamma]))

            # check primal feasibility
            constraint_viol = tf.reduce_sum(
                constraint_viol[constraint_viol > 0]).numpy()  

            #log values
            if self.logger is not None and iters % 50 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("Loss", loss_value, iters)
                    self.logger.log_scalar("Violation", constraint_viol, iters)

            # check if nothing is improving for a while, or save last improvment 
            if best_iter < iters - self.max_stagnation and best_viol < 1e-8:
                early_stop = True
            if constraint_viol < best_viol:
                best_viol, best_iter = constraint_viol, iters

        pred_y = model(nn_data)
        return pred_y