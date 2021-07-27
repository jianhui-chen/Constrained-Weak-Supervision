import sys
import numpy as np
import tensorflow as tf


from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.keras import layers, losses
from sklearn.cluster import MiniBatchKMeans

from BaseClassifier import BaseClassifier
from log import Logger



"""
    Includes LabelEstimator and CLL
    CLL should inherit from LabelEstimator – needs to be fixed
"""

class LabelEstimator(BaseClassifier):   # Might want to change the name of Base Classifier?
    """
    Label Estimator + Classifier
    Subclasses can redefine _estimate_labels process

    Parameters
    ----------
    max_iter : int, default=None
        Maximum number of iterations taken for solvers to converge.

    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=None, log_name=None):
    
   
        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        # elif type(log_name) is str:
        #     self.logger = Logger("logs/Baseline/" + log_name + "/" + 
        #                          str(weak_signals_proba.shape[0]) + 
        #                          "_weak_signals/")      # this can be modified to include date and time in file name
        # else:
        #     sys.exit("Not of string type")

        # not based on args bc based on feature number
        self.model = None

 

    def predict_proba(self, X):
        """
        Computes probability estimates for given class

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)
        """
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict_proba(X.T)[:,1]

        return probabilities


    def _estimate_labels(self, weak_signals_probas, weak_signals_error_bounds):
        """
        Estimates labels by averaging weak signals

        Parameters
        ----------
        weak_signals_proba : ndarray of shape (n_weak_signals, n_examples)
            A set of soft or hard weak estimates for data examples.
            This may later be changed to accept just the weak signals, and these 
            probabilities will be calculated within the ALL class. 

        weak_signals_error_bounds : ndarray of shape (n_weak_signals,)
            Stores upper bounds of error rates for each weak signal.


        Returns
        -------
        Estimated labels

        """
        labels=np.zeros(weak_signals_probas.shape[1]) # no of examples
        average_weak_labels = np.mean(weak_signals_probas, axis=0)
        labels[average_weak_labels > 0.5] = 1
        labels[average_weak_labels <= 0.5] = 0


        return labels


    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, 
            train_model=None): 
        """
        Option: we can make it so the labels are generated outside of method
            i.e. passed in as y, or change to pass in algo to generate within
            this method
        error_bounds param is not used, but included for consistency
        """

        # Estimates labels
        labels = self._estimate_labels(weak_signals_probas, weak_signals_error_bounds)


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


"""


####################### CLL #########################################


"""

class CLL(BaseClassifier):
    """
    Constrained label learning Classifier

    This class implements CLL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations taken for solvers to converge.

    num_trials : int, default=3
        number of time's labels are estimated before the mean is taken
    
    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=300, num_trials=3, log_name=None,):

        self.max_iter = max_iter
        self.num_trials = num_trials

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name)      # this can be modified to include date and time in file name
        else:
            sys.exit("Not of string type")

        # Might nee weights for predict proba
        

        self.model = None

    def get_accuracy(self, true_labels, predicted_probas): 
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
        try:
            n, k = true_labels.shape
            if k > 1:
                assert true_labels.shape == predicted_probas.shape
                return np.mean(np.equal(np.argmax(true_labels, axis=-1),
                                        np.argmax(predicted_probas, axis=-1)))
        except:
            if len(true_labels.shape) == 1:
                y_pred = np.round(predicted_probas.ravel())
    
        assert true_labels.shape == y_pred.shape
        return np.mean(np.equal(true_labels, np.round(y_pred)))

    def predict(self, X):
        """
        Computes probability estimates for given class

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)
        """
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict(X)

        return probabilities
    

    def predict_proba(self, X):
        """
        Computes probability estimates for given class

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)
        """
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict_proba(X)

        return probabilities


    def _bound_loss(self, y, a_matrix, bounds):
        """
        Computes the gradient of lagrangian inequality penalty parameters

        Parameters
        ----------
        :param y: size (num_data, num_class) of estimated labels for the data
        :type y: ndarray
        :param a_matrix: size (num_weak, num_data, num_class) of a constraint matrix
        :type a_matrix: ndarray
        :param bounds: size (num_weak, num_class) of the bounds for the constraint
        :type bounds: ndarray

        Returns
        -------
        :return: loss of the constraint (num_weak, num_class)
        :rtype: ndarray
        """
        constraint = np.zeros(bounds.shape)

        for i, current_a in enumerate(a_matrix):
            constraint[i] = np.sum(current_a * y, axis=0)
        return constraint - bounds


    def _y_gradient(self, y, error_bounds):
        """
        Computes y gradient

        Parameters
        ----------
        :param y: estimated Labels
        :type  y: ndarray
        :param weak_signals_error_bounds: error constraints of the weak signals
        :type  weak_signals_error_bounds: dictionary containing both a_matrix and bounds

        Returns
        -------
        :return: computed gradient
        :rtype: float
        """
        gradient = 0
        a_matrix = error_bounds['A']
        bound_loss = error_bounds['bound_loss']

        for i, current_a in enumerate(a_matrix):
            constraint = a_matrix[i]
            gradient += 2*constraint * bound_loss[i]

        return gradient


    def _run_constraints(self, y, error_constraints):
        """
        Run constraints from CLL

        :param y: Random starting values for labels
        :type  y: ndarray 
        :param error_constraints: error constraints (a_matrix and bounds) of the weak signals 
        :type  error_constraints: dictionary

        :return: estimated learned labels
        :rtype: ndarray
        """
        grad_sum = 0

        for iter in range(self.max_iter):
            
            current_constraint = error_constraints
            a_matrix = current_constraint['A']
            bounds = current_constraint['b']

            # get bound loss for constraint
            loss = self._bound_loss(y, a_matrix, bounds)

            # update constraint values
            error_constraints['bound_loss'] = loss
            violation = np.linalg.norm(loss.clip(min=0))

            # Update y˜ with its gradient
            y_grad = self._y_gradient(y, error_constraints)
            grad_sum += y_grad**2
            y = y - y_grad / np.sqrt(grad_sum + 1e-8)
            y = np.clip(y, a_min=0, a_max=1)

            # log current data 
            if self.logger is not None and iter % 10 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("y", np.average(y), iter)
                    self.logger.log_scalar("y_grad", np.average(y_grad), iter)
                    self.logger.log_scalar("loss", np.average(loss), iter)
                    self.logger.log_scalar("violation", np.average(violation), iter)
        return y
    

    def _estimate_labels(self, weak_signals_probas, weak_signals_error_bounds):
        """
        Finds estimated labels

        Parameters
        ----------
        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for each example
        :type  weak_signals_probas: ndarray 
        :param error_bounds: error constraints (a_matrix and bounds) of the weak signals. Contains both 
                             left (a_matrix) and right (bounds) hand matrix of the inequality 
        :type  error_bounds: dictionary 

        Returns
        -------
        :return: average of learned labels over several trials
        :rtype: ndarray
        """
        assert len(weak_signals_probas.shape) == 3, "Reshape weak signals to num_weak x num_data x num_class"
        m, n, k = weak_signals_probas.shape

        # initialize y and lists
        y = np.random.rand(n, k)
        ys = []

        for i in range(self.num_trials):
            ys.append( self._run_constraints(y, weak_signals_error_bounds) )
        return np.mean(ys, axis=0)    


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



    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, train_model=None):
        """
        Finds estimated labels

        Parameters
        ----------
        :param X: current data examples to fit model with
        :type  X: ndarray 
        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for each example
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error constraints (a_matrix and bounds) of the weak signals. Contains both 
                                          left (a_matrix) and right (bounds) hand matrix of the inequality 
        :type  weak_signals_error_bounds: dictionary 

        Returns
        -------
        :return: average of learned labels over several trials
        :rtype: ndarray
        """

        # Estimates labels
        labels = self._estimate_labels(weak_signals_probas, weak_signals_error_bounds)

        # Fit based on labels generated above
        if train_model is None:
            m, n, k = weak_signals_probas.shape
            self.model = self._mlp_model(X.shape[1], k)
            self.model.fit(X, labels, batch_size=32, epochs=20, verbose=1)
        else:
            self.model = train_model
            try:
                self.model.fit(X.T, labels)
            except:
                print("The mean of the baseline labels is %f" %np.mean(labels))
                sys.exit(1)

        return self



class DataConsistency(BaseClassifier):
    # FIX THIS
    """
    Data Consistency learning Classifier

    This class implements CLL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations taken for solvers to converge.

    num_trials : int, default=3
        number of time's labels are estimated before the mean is taken
    
    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=300, num_trials=3, log_name=None,):

        self.max_iter = max_iter
        self.num_trials = num_trials

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name)      # this can be modified to include date and time in file name
        else:
            sys.exit("Not of string type")

        # Might nee weights for predict proba
        

        self.model = None

    def get_accuracy(self, true_labels, predicted_probas): 
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
        try:
            n, k = true_labels.shape
            if k > 1:
                assert true_labels.shape == predicted_probas.shape
                return np.mean(np.equal(np.argmax(true_labels, axis=-1),
                                        np.argmax(predicted_probas, axis=-1)))
        except:
            if len(true_labels.shape) == 1:
                y_pred = np.round(predicted_probas.ravel())
    
        assert true_labels.shape == y_pred.shape
        return np.mean(np.equal(true_labels, np.round(y_pred)))

    def predict(self, X):
        """
        Computes probability estimates for given class

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)
        """
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict(X)

        return probabilities
    

    def predict_proba(self, X):
        """
        Computes probability estimates for given class

        Parameters
        ----------
        X : ndarray of shape (n_features, n_examples)
            Examples to be assigned a probability (binary)


        Returns
        -------
        probas : ndarray of shape (n_examples,)
        """
        if self.model is None:
            sys.exit("No Data fit")

        probabilities = self.model.predict_proba(X)

        return probabilities

    # def _batch_clustering(self, data, no_clusters, batch_size=50):
    #     cluster = MiniBatchKMeans(init='k-means++', n_clusters=no_clusters, batch_size=batch_size,
    #                             n_init=10, max_no_improvement=10, verbose=0)
    #     cluster.fit(data)
    #     cluster_labels = cluster.labels_
    #     return tf.one_hot(cluster_labels, no_clusters)


    # @tf.function
    def _consistency_loss(self, model, X, mv_labels, a_matrix, b, slack, gamma, C):
        # Lagragian loss function

        m, n, k = a_matrix.shape
        lagragian_objective = tf.zeros(k)
        constraint_violation = tf.zeros(k)
        i = 0
        Y = model(X, training=True)
        # mv_labels = tf.ones((n, k), dtype=tf.float32) * 0.5 #### uniform
        # regularization

        primal = tf.divide(tf.nn.l2_loss(Y - mv_labels), n)
        primal = tf.add(primal, tf.multiply(C, tf.reduce_mean(slack)))
        for A in a_matrix:
            AY = tf.reduce_sum(tf.multiply(A, Y), axis=0)
            violation = tf.add(b[i], slack[i])
            violation = tf.subtract(AY, violation)
            value = tf.multiply(gamma[i], violation)
            lagragian_objective = tf.add(lagragian_objective, value)
            constraint_violation = tf.add(constraint_violation, violation)
            i += 1
        lagragian_objective = tf.add(primal, tf.reduce_sum(lagragian_objective))
        return lagragian_objective, constraint_violation


    def _estimate_labels(self, model, X, mv_labels, a_matrix, b):
        # Train DCWS algorithm

        train_loss_results = []
        train_accuracy_results = []
        optimizer = tf.keras.optimizers.Adam()
        opt = tf.keras.optimizers.SGD()
        iterations = 1000  # max number of iterations to train model
        max_stagnation = 100  # number of epochs without improvement to tolerate
        best_viol, best_iter = np.inf, iterations
        early_stop = False
        C = tf.constant(10, dtype=tf.float32)

        m, k = b.shape
        gamma = np.random.rand(m, k)
        gamma = tf.Variable(gamma.astype(np.float32), trainable=True,
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        slack = np.zeros(b.shape, dtype="float32")
        slack = tf.Variable(slack, trainable=True,
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        mv_labels = tf.constant(mv_labels, dtype=tf.float32)

        for iters in range(iterations):
            if early_stop:
                break
            with tf.GradientTape() as tape:
                loss_value, constraint_viol = self._consistency_loss(
                    model, X, mv_labels, a_matrix, b, slack, gamma, C)
                model_grad, gamma_grad, slack_grad = tape.gradient(
                    loss_value, [model.trainable_variables, gamma, slack])

            optimizer.apply_gradients(zip(model_grad, model.trainable_variables))
            opt.apply_gradients(zip([slack_grad], [slack]))
            opt.apply_gradients(zip([-1 * gamma_grad], [gamma]))

            constraint_viol = tf.reduce_sum(
                constraint_viol[constraint_viol > 0]).numpy()  # check primal feasibility
            if iters % 50 == 0:
                print("Iter {:03d}:, Loss: {:.3}, Violation: {:.3}".format(
                    iters, loss_value, constraint_viol))

            if best_iter < iters - max_stagnation and best_viol < 1e-8:
                # nothing is improving for a while
                early_stop = True
            if constraint_viol < best_viol:
                best_viol, best_iter = constraint_viol, iters

        pred_y = model(X)
        return pred_y

    def _simple_nn(self, dimension, output):
        # Data consistent model

        actv = 'softmax' if output > 1 else 'sigmoid'
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(dimension,)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(output, activation=actv))
        return model
        


    def fit(self, X, weak_signals_probas, weak_signals_error_bounds, train_model=None):
        """
        Finds estimated labels

        Parameters
        ----------
        :param X: current data examples to fit model with
        :type  X: ndarray 
        :param weak_signals_probas: weak signal probabilites containing -1, 0, 1 for each example
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error constraints (a_matrix and bounds) of the weak signals. Contains both 
                                          left (a_matrix) and right (bounds) hand matrix of the inequality 
        :type  weak_signals_error_bounds: dictionary 

        Returns
        -------
        :return: average of learned labels over several trials
        :rtype: ndarray
        """

        model = self.simple_nn(nn_data.shape[1], k)

        

        # Estimates labels
        labels = self._estimate_labels(weak_signals_probas, weak_signals_error_bounds)

        # Fit based on labels generated above
        if train_model is None:
            m, n, k = weak_signals_probas.shape
            self.model = self._mlp_model(X.shape[1], k)
            self.model.fit(X, labels, batch_size=32, epochs=20, verbose=1)
        else:
            self.model = train_model
            try:
                self.model.fit(X.T, labels)
            except:
                print("The mean of the baseline labels is %f" %np.mean(labels))
                sys.exit(1)

        return self
        