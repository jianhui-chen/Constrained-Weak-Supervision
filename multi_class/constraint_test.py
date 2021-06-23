import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, json
from utilities import *
from setup_supervision import get_supervision_data, accuracy_score, writeToFile
from setup_model import build_constraints
from train_lagrangian import run_constraints, multiclass_loss, multilabel_loss



def run_algo(data, optim):
    """
    Trains the cnn model for image classification

    :param labels: True labels for the data, only used for debugging
    :type labels: ndarray
    :param constraint_set: dictionary containing constraints specified in the constraint_keys
    :type constraint_set: dict
    :return: error of the objective function
    :rtype: scalar
    """
    rho = 0.1
    labels = data['labels']
    constraint_set = data['constraints']
    loss = data['constraints']['loss']
    n,k = labels.shape
    # y = np.ones(n,k) * 0.1
    y = np.random.rand(n,k)

    y, constraint_set = run_constraints(y, labels, rho, constraint_set, optim=optim)
    return 1 - accuracy_score(labels, y)



def run_min_max(data_set):

    constraint_keys = ["error"]

    true_bounds = False
    loss = 'multilabel'
    data = get_supervision_data(data_set, weak_signals='pseudolabels', true_bounds=true_bounds)
    weak_model = data['weak_model']
    weak_signal_probabilities = weak_model['weak_probabilities']

    constraint_set = data['constraint_set']
    train_data, train_labels = data['train_data']

    # build up data_info for the algorithm
    data_info = dict()
    data_info['labels']= train_labels

    start = 1
    min_optimised = []
    max_optimised = []

    max_weak_signals = weak_signal_probabilities.shape[0]

    for num_weak_signals in range(start, max_weak_signals + 1):

        new_constraint_set = dict()

        for key in constraint_keys:
            current_constraint = constraint_set[key]
            a_matrix = current_constraint['A']
            bounds = current_constraint['b']
            constant = current_constraint['c']

            # use only listed number of signals
            a_matrix = a_matrix[:num_weak_signals, :, :]
            bounds = bounds[:num_weak_signals, :]
            constant = constant[:num_weak_signals, :, :]

            new_set = build_constraints(a_matrix, bounds, constant)
            new_constraint_set[key] = new_set
            new_constraint_set['constraints'] = constraint_keys
            new_constraint_set['loss'] = loss

        print("weak_signal_probabilities", a_matrix.shape)
        new_constraint_set['weak_signals'] = weak_signal_probabilities
        new_constraint_set['num_weak_signals'] = num_weak_signals
        new_constraint_set['true_bounds'] = true_bounds

        data_info['constraints'] = new_constraint_set

        print("Running minimization test...")
        result = run_algo(data_info, optim='min')
        print("Y accuracy wrt true label", 1 - result)
        min_optimised.append(result)
        print("Error of Y wrt true label", result)
        print("")

        print("Running maximization test...")
        result = run_algo(data_info, optim='max')
        print("Y accuracy wrt true label", 1 - result)
        print("Error of Y wrt true label", result)
        max_optimised.append(result)
        print("")

    output = {}
    output['min_errors'] = min_optimised
    output['max_errors'] = max_optimised

    filename = 'results/constraints.json'

    print("Saving results to file")
    writeToFile(output, filename)

    weak_signals = np.arange(start, max_weak_signals + 1)
    plt.plot(weak_signals, min_optimised, color='g', label='min')
    plt.plot(weak_signals, max_optimised, color='orange', label='max')
    plt.xlabel('No. of weak signals')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.title('Error of y wrt True Labels')
    plt.savefig('results/error_test.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_min_max(load_svhn())
