import os
import json
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras import backend as K
from data_generator import *
from models import *
from image_generator import load_image_signals

ROOT_DIR = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(ROOT_DIR))

from train_classifier import train_all
from mmce import MinimaxEntropy_crowd_model
from train_CLL import run_constraints
from model_utilities import *


def consistency_data(dataset, form='data', encoding_dim=32, no_clusters=10):
    """ Select form of data consistency"""

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']

    if form == 'data':
        return tf.cast(train_data, dtype=tf.float32)

    if form == 'data_cluster':
        return batch_clustering(train_data, no_clusters)

    embedding_model = build_autoencoder(train_data.shape[1],
            encoding_dim=encoding_dim, lr=1e-3)
    embedding = train_autoender(embedding_model, train_data, test_data,
            epochs=30, batch_size=32)
    embedding = embedding.encoder(train_data)
    if form == 'embedding_cluster':
        return batch_clustering(embedding, no_clusters)
    return tf.cast(embedding, dtype=tf.float32)


def run_cll(constraints):
    """ Run CLL experiments"""
    constraints['constraints'] = ['error']
    m, n, k = constraints['error']['A'].shape
    rho = 0.1
    y = run_constraints(np.random.rand(n, k), rho,
                        constraints, enable_print=False)
    return y


def run_mmce(crowd_labels, true_labels):
    """ Run MMCE experiments"""

    mmce_crowd_labels, transformed_labels = prepare_mmce(
        crowd_labels, true_labels)
    mmce_labels, mmce_error_rate = MinimaxEntropy_crowd_model(
        mmce_crowd_labels, transformed_labels)
    return mmce_labels - 1, 1 - mmce_error_rate


def print_signal_stats(weak_signals):
    """ Print statistics about weak signals"""

    m, n, k = weak_signals.shape
    mask = weak_signals >= 0
    denom = n * k
    # no of labelers per example
    redundancy = np.mean(np.sum(mask, axis=(0, 2)))
    average = np.sum(weak_signals * mask, axis=0) / np.sum(mask, axis=0)
    conflict = denom - np.sum(average == 0) - np.sum(average == 1)

    print("Number of labelers per example: ", redundancy)
    print("Weak signal conflict is: %f \n" % (conflict / denom))


def run_experiment(dataset, savename, datatype='data', true_bound=False):
    """ Run the main experiments"""

    batch_size = 32
    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape
    num_trials = 3
    consistency_accuracy = []
    consistency_test = []
    results = {}
    weak_errors = np.zeros((m, k))

    if true_bound:
        weak_errors = get_error_bounds(train_labels, weak_signals)
        weak_errors = np.asarray(weak_errors)
    nn_data = consistency_data(dataset, datatype)

    # Define the variables
    constraints = set_up_constraint(weak_signals, weak_errors)
    mv_labels = majority_vote_signal(weak_signals)

    a_matrix = tf.constant(constraints['error']['A'], dtype=tf.float32)
    b = tf.constant(constraints['error']['b'], dtype=tf.float32)
    model = simple_nn(nn_data.shape[1], k)

    for _ in range(num_trials):
        model = simple_nn(nn_data.shape[1], k)
        pred_y = train_dcws(model, nn_data, mv_labels, a_matrix, b)
        pred_y = pred_y.numpy()
        label_accuracy = accuracy_score(train_labels, pred_y)
        print("Label accuracy: ", label_accuracy)
        consistency_accuracy.append(label_accuracy)
        model = mlp_model(nn_data.shape[1], k)
        model.fit(nn_data, pred_y, batch_size=batch_size, epochs=20, verbose=1)
        test_predictions = model.predict(test_data)
        test_score = accuracy_score(test_labels, test_predictions)
        print("Test accuracy: ", test_score)
        consistency_test.append(test_score)

        K.clear_session()
        del model
        gc.collect()

    # #################################################################
    # for regular experiments

    # cll experiment
    cll_y = run_cll(constraints)
    cll_accuracy = accuracy_score(train_labels, cll_y)
    model = mlp_model(nn_data.shape[1], k)
    model.fit(nn_data, cll_y, batch_size=batch_size, epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    cll_test_score = accuracy_score(test_labels, test_predictions)

    # majority vote experiment
    mv_accuracy = accuracy_score(train_labels, mv_labels)
    model = mlp_model(nn_data.shape[1], k)
    model.fit(nn_data, mv_labels, batch_size=batch_size, epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    mv_test_score = accuracy_score(test_labels, test_predictions)

    # mmce experiment
    mmce_labels = weak_signals
    mmce_labels, mmce_accuracy = run_mmce(mmce_labels, train_labels)
    if k > 1:
        mmce_labels = tf.one_hot(mmce_labels, k)
    model = mlp_model(nn_data.shape[1], k)
    model.fit(nn_data, mmce_labels, batch_size=batch_size,
            epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    mmce_test_score = accuracy_score(test_labels, test_predictions)

    mmce = {}
    filename = 'results/mmce_results.json'
    mmce['label_accuracy'] = mmce_accuracy
    mmce['test_score'] = mmce_test_score
    mmce['experiment'] = savename
    with open(filename, 'a') as file:
        json.dump(mmce, file, indent=4, separators=(',', ':'))
    file.close()

    # supervised learning
    model = mlp_model(nn_data.shape[1], k)
    model.fit(nn_data, train_labels, batch_size=batch_size,
            epochs=30, verbose=1)
    test_predictions = model.predict(test_data)
    spv_test_score = accuracy_score(test_labels, test_predictions)

    # print results
    print('SPV Test accuracy: %f' % spv_test_score)
    print("CLL accuracy: ", cll_accuracy)
    print("CLL Test accuracy: ", cll_test_score)
    print("")
    print("MV accuracy: ", mv_accuracy)
    print("MV Test accuracy: ", mv_test_score)
    print("")
    print('MMCE accuracy: %f' % mmce_accuracy)
    print('MMCE Test accuracy: %f' % mmce_test_score)

    results['consistency_label_accuracies'] = consistency_accuracy
    results['consistency_label_stats'] = [np.mean(consistency_accuracy), np.std(consistency_accuracy)]
    results['consistency_test_accuracies'] = consistency_test
    results['consistency_test_stats'] = [np.mean(consistency_test), np.std(consistency_test)]
    results['cll_label_acc'] = cll_accuracy
    results['cll_test_acc'] = cll_test_score
    results['mv_label_acc'] = mv_accuracy
    results['mv_test_acc'] = mv_test_score
    results['spv_test_acc'] = spv_test_score
    filename = 'results/' + savename + '.json'
    writeToFile(results, filename)

    print_signal_stats(weak_signals)
    #########################################################################
    # for ablation tests
    # filename = 'results/'+savename+'.json'
    # results['consistency_accuracies'] = consistency_accuracy
    # results['consistency_stats'] = [np.mean(consistency_accuracy), np.std(consistency_accuracy)]
    # results['experiment'] = 'uniform_regularization'
    # with open(filename, 'a') as file:
    #     json.dump(results, file, indent=4, separators=(',', ':'))
    # file.close()
    ########################################################################


def run_ALL_experiments(dataset, savename, true_bound=False):
    """ Run ALL experiments"""

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']

    bound, precisions = get_error_bounds(train_labels, weak_signals)
    bound = np.squeeze(bound, axis=-1)

    all_results = {}
    m, n, k = weak_signals.shape
    if k == 1:
        all_signals = np.squeeze(weak_signals, axis=-1)
    else:
        print("ALL does not run multi-class data")
        sys.exit(0)

    if not true_bound:
        bound = np.ones(m * k) * 0  # constant bounds
    learned_labels, optimized_weights = train_all(
        train_data.T, all_signals, bound, max_iter=4000)
    all_accuracy = accuracy_score(train_labels, np.round(learned_labels))
    print("ALL accuracy is: ", all_accuracy)
    all_results['label_accuracy'] = all_accuracy

    model = mlp_model(train_data.shape[1], 1)
    model.fit(train_data, np.round(learned_labels).T, batch_size=32,
            epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    test_score = accuracy_score(test_labels, test_predictions)
    all_results['test_score'] = test_score

    ###################################################################
    filename = 'results/all_results.json'
    all_results['experiment'] = savename
    with open(filename, 'a') as file:
        json.dump(all_results, file, indent=4, separators=(',', ':'))
    file.close()


def run_2D_experiment(dataset):
    X, y = dataset['train']
    X_test, y_test = dataset['train']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape
    results = {}

    weak_signal_errors = np.ones((m, k)) * 0
    # Define the variables
    weak_errors = np.asarray(weak_signal_errors)
    constraints = set_up_constraint(weak_signals, np.ones((m, k)), weak_errors)
    mv_labels = majority_vote_signal(weak_signals, m)

    a_matrix = tf.constant(constraints['error']['A'])
    b = tf.constant(constraints['error']['b'])
    dcws_model = simple_nn(X.shape[1], 1)

    pred_y = train_consistent_cll(dcws_model, X, mv_labels, a_matrix, b)
    pred_y = pred_y.numpy()
    label_accuracy = accuracy_score(y, pred_y)
    print("Label accuracy: ", label_accuracy)
    print("MV accuracy: ", accuracy_score(y, mv_labels))

    def plot_boundary(labels, savename):
        print("Weak signal accuracy:", accuracy_score(y, labels))
        labels = np.ravel(labels)
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X, np.round(labels))
        parameters = model.coef_[0]
        # Intercept (a.k.a. bias) added to the decision function. (theta 0)
        parameter0 = model.intercept_
        x_values = np.array([np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)])
        # calculate y values
        y_values = (-1. / parameters[1]) * \
            (parameters[0] * x_values + parameter0)

        plt.plot(X_test[:, 0][labels == 0], X_test[:, 1][labels == 0], 'r.')
        plt.plot(X_test[:, 0][labels == 1], X_test[:, 1][labels == 1], 'b.')
        plt.plot(x_values, y_values, label='Decision Boundary')
        plt.ylim(ymin=-14.95, ymax=5.68)
        plt.savefig('results/' + savename)
        plt.show()

    # plot_boundary(pred_y, 'dcws.png')
    plot_boundary(weak_signals[0], 'signal1.png')
    plot_boundary(weak_signals[1], 'signal2.png')
    plot_boundary(weak_signals[2], 'signal3.png')
    plot_boundary(mv_labels, 'majority.png')


if __name__ == '__main__':
    print("Running experiments...")
    # run_2D_experiment(synthetic_2D(1250, 3, form='sep'))

    run_experiment(synthetic_data(20000, 10), 'synthetic')
    # run_experiment(read_text_data('../../datasets/imbd/'), 'imbd')
    # run_experiment(read_text_data('../../datasets/yelp/'),'yelp')
    # run_experiment(read_text_data('../../datasets/sst-2/'), 'sst-2')
    # run_experiment(load_image_signals('../../datasets/fashion-mnist'), 'fashion-mnist')
    # run_experiment(read_text_data('../../datasets/sst-2/'), 'sst-2_ablation_test', true_bound=False)
    # run_experiment(read_text_data('../../datasets/yelp/'),'yelp_ablation_test', true_bound=False)

    # run_ALL_experiments(read_text_data('../../datasets/sst-2/'), 'sst-2', true_bound=True)
    # run_ALL_experiments(read_text_data('../../datasets/imbd/'), 'imbd', true_bound=True)
    # run_ALL_experiments(read_text_data('../../datasets/yelp/'),'yelp', true_bound=True)
    # run_ALL_experiments(load_image_signals('../../datasets/fashion-mnist'), 'fashion-mnist', true_bound=True)
    # run_ALL_experiments(load_image_signals('../../datasets/svhn'), 'svhn', true_bound=True)
    # run_ALL_experiments(synthetic_data2(20000, 10), 'synthetic', true_bound=True)
