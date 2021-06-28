import pickle
import sys
import os
import gc
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.keras import backend as K

ROOT_DIR = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIR)

from model_utilities import *
from models import *


def convert_labels(l, m):
    # Convert labels to DCWS & snorkel format
    labels = l*m + -1*(1-m)
    return labels.astype(np.int)


def read_pickle_data(filepath, filename):
    # Read pickle files for the dataset

    U_file = open(os.path.join(filepath, filename), "rb")
    data = pickle.load(U_file)
    weak_labels = pickle.load(U_file)
    coverage_mask = pickle.load(U_file)
    data_labels = pickle.load(U_file)
    label_mask = pickle.load(U_file)

    weak_labels = convert_labels(weak_labels, coverage_mask)
    return data, data_labels, weak_labels, coverage_mask


def get_binary_data(filepath, multi=False):
    # Join labeled data with unlabeled data, also return test data

    l_data, l_labels, l_weak_labels, l_coverage_mask = read_pickle_data(
        filepath, "d_processed.p")
    u_data, u_labels, u_weak_labels, u_coverage_mask = read_pickle_data(
        filepath, "U_processed.p")
    t_data, t_labels, t_weak_labels, t_coverage_mask = read_pickle_data(
        filepath, "test_processed.p")

    # indices of instances where atleast one rule fired
    fired_idx = [i for i, item in enumerate(t_coverage_mask) if sum(item) > 0]
    test_data, test_labels = t_data[fired_idx], t_labels[fired_idx]

    # indices of instances where atleast one rule fired
    fired_idx = [i for i, item in enumerate(u_coverage_mask) if sum(item) > 0]
    u_data, u_labels = u_data[fired_idx], u_labels[fired_idx]
    u_weak_labels = u_weak_labels[fired_idx]
    # join unlabeled and labeled data
    train_data, train_labels = np.vstack(
        [l_data, u_data]), np.concatenate([l_labels, u_labels])

    # get weak signals
    if multi:
        weak_signals = u_weak_labels
    else:
        l_weak_labels = np.ones(l_weak_labels.shape) * l_labels.reshape(-1, 1)
        weak_signals = np.vstack([l_weak_labels, u_weak_labels])

    dataset = {}
    dataset['train'] = train_data, train_labels
    dataset['test'] = test_data, test_labels
    dataset['weak_signals'] = np.expand_dims(weak_signals.T, axis=-1)
    dataset['fired_idx'] = fired_idx

    return dataset


def transform_signals(weak_signals, num_classes, weak_signal_dict=None):
    ''' transform multiclass (num_weak x num_data) weak signals into
        num_weak x num_data x num_class weak signal form for DCWS
        Weak signal shape should be num_signals x num_data
    '''
    m, n = weak_signals.shape

    if weak_signal_dict is None:
        max_array = np.zeros(num_classes, dtype=int)
        dict_index = defaultdict(list)

        for i, signal in enumerate(weak_signals):
            # if the weak signal labels the example
            signal = signal[signal != -1]
            if len(signal) != 0:
                label = signal[0]
                dict_index[str(label)].append(i)  # append indices of label
                max_array[label] += 1
        max_index = np.max(max_array)
    else:
        dict_index = weak_signal_dict
        max_index = 0
        for k, v in dict_index.items():
            max_index = np.max([max_index, len(v)])

    reshaped_signals = np.ones((max_index, n, num_classes)) * -1
    for k, v in dict_index.items():
        for i, signal in enumerate(v):
            reshaped_signals[i, :, int(k)] = weak_signals[signal]
    reshaped_signals[reshaped_signals != -1] = 1

    return reshaped_signals, dict_index


def get_multiclass_data(filepath):
    # process the multiclass dataset

    dataset = get_binary_data(filepath, multi=True)
    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    fired_idx = dataset['fired_idx']
    num_classes = np.unique(test_labels).size
    weak_signals = np.squeeze(weak_signals, axis=-1)
    weak_signals, dict_index = transform_signals(weak_signals, num_classes)

    _, labels, l_weak_labels, _ = read_pickle_data(filepath, "d_processed.p")
    labeled_signals = tf.one_hot(labels, num_classes).numpy()
    weak_signals = np.array([np.vstack([labeled_signals, signal])
                            for signal in weak_signals])
    train_labels = tf.one_hot(train_labels, num_classes).numpy()

    # get validation data to calculate weak signal errors
    v_data, v_labels, v_weak_labels, v_coverage_mask = read_pickle_data(
        filepath, "validation_processed.p")
    fired_idx = [i for i, item in enumerate(v_coverage_mask) if sum(item) > 0]
    v_weak_labels, v_labels = v_weak_labels[fired_idx], v_labels[fired_idx]
    v_signals, _ = transform_signals(v_weak_labels.T, num_classes, dict_index)
    weak_signal_errors = get_error_bounds(
        tf.one_hot(v_labels, num_classes).numpy(), v_signals)

    dataset['weak_signal_errors'] = np.asarray(weak_signal_errors)
    dataset['train'] = train_data, train_labels
    dataset['test'] = test_data, tf.one_hot(test_labels, num_classes).numpy()
    dataset['weak_signals'] = weak_signals
    return dataset


def run_experiment(dataset, savename, datatype='data'):

    batch_size = 32
    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape
    num_trials = 3
    consistency_accuracy = []
    consistency_test = []
    results = {}
    metric_labels = np.argmax(test_labels, axis=-1)

    weak_errors = np.ones((m, k))*0
    if k > 2:
        weak_errors = dataset['weak_signal_errors']

    # Define the variables
    constraints = set_up_constraint(weak_signals, weak_errors)
    mv_labels = majority_vote_signal(weak_signals)

    a_matrix = tf.constant(constraints['error']['A'], dtype=tf.float32)
    b = tf.constant(constraints['error']['b'], dtype=tf.float32)
    model = simple_nn(train_data.shape[1], k)

    for _ in range(num_trials):
        model = simple_nn(train_data.shape[1], k)
        pred_y = train_dcws(model, train_data, mv_labels, a_matrix, b)
        pred_y = pred_y.numpy()
        label_accuracy = accuracy_score(train_labels, pred_y)
        print("Label accuracy: ", label_accuracy)
        consistency_accuracy.append(label_accuracy)
        model = mlp_model(train_data.shape[1], k)
        model.fit(train_data, pred_y, batch_size=batch_size,
                  epochs=20, verbose=1)
        test_predictions = model.predict(test_data)
        test_score = accuracy_score(test_labels, test_predictions)
        print("Test accuracy: ", test_score)
        if 'mitr' in savename or 'sms' in savename:
            if k > 1:
                precision, recall, test_score, support = precision_recall_fscore_support(
                    metric_labels, np.argmax(test_predictions, axis=-1), average='macro')
            else:
                precision, recall, test_score, support = precision_recall_fscore_support(
                    test_labels, np.round(test_predictions).ravel(), average='binary')
            print("F-score: ", test_score)  # save the f-score instead
        consistency_test.append(test_score)
        K.clear_session()
        del model
        gc.collect()

    results['consistency_label_accuracies'] = consistency_accuracy
    results['consistency_label_stats'] = [
        np.mean(consistency_accuracy), np.std(consistency_accuracy)]
    results['consistency_test_accuracies'] = consistency_test
    results['consistency_test_stats'] = [
        np.mean(consistency_test), np.std(consistency_test)]
    filename = 'results/'+savename+'.json'
    writeToFile(results, filename)


if __name__ == '__main__':
    print("Running experiments...")

    # run_experiment(get_binary_data('../../datasets/CENSUS/'),'census')
    run_experiment(get_binary_data('../../datasets/SMS/'), 'sms')
    # run_experiment(get_binary_data('../../datasets/YOUTUBE/'),'youtube')
    # run_experiment(get_multiclass_data('../../datasets/MITR/'),'mitr')
    # run_experiment(get_multiclass_data('../../datasets/TREC/'),'trec') # trec train labels are not provided
