import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.cluster import MiniBatchKMeans


def simple_nn(dimension, output):
    # Data consistent model

    actv = 'softmax' if output > 1 else 'sigmoid'
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(dimension,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output, activation=actv))
    return model


def batch_clustering(data, no_clusters, batch_size=50):
    cluster = MiniBatchKMeans(init='k-means++', n_clusters=no_clusters, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
    cluster.fit(data)
    cluster_labels = cluster.labels_
    return tf.one_hot(cluster_labels, no_clusters)


@tf.function
def consistency_loss(model, X, mv_labels, a_matrix, b, slack, gamma, C):
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


def train_dcws(model, X, mv_labels, a_matrix, b):
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
            loss_value, constraint_viol = consistency_loss(
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


if __name__ == '__main__':
    pass
