import os
import sys
import numpy as np
import pandas as pd
import codecs
import json
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel

from data_generator import read_text_data, synthetic_data

ROOT_DIR = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIR)
from model_utilities import accuracy_score, mlp_model


def get_metal_labels(weak_signals, no_of_classes=2):
    label_model = LabelModel(cardinality=no_of_classes, verbose=True)
    label_model.fit(L_train=weak_signals)
    learned_labels = label_model.predict(weak_signals)
    return learned_labels


def run_experiments(savename):
    print("Running experiment on %s dataset" % savename)

    if 'fashion-mnist' or 'svhn' in savename:
        datapath = '../../datasets/'+savename
        dataset = np.load(os.path.join(
            datapath, 'binary_data.npy'), allow_pickle=True)[()]
    else:
        dataset = read_text_data('../../datasets/'+savename+'/')

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape
    weak_signals = np.squeeze(weak_signals, axis=-1)

    learned_labels = get_metal_labels(weak_signals.T)
    label_accuracy = accuracy_score(train_labels, learned_labels)
    print('Label accuracy: %f' % label_accuracy)

    model = mlp_model(train_data.shape[1], k)
    model.fit(train_data, learned_labels, batch_size=32, epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print('Test accuracy: %f' % test_accuracy)

    results = {}
    filename = 'results/metal_results.json'
    results['label_accuracy'] = label_accuracy
    results['test_accuracy'] = test_accuracy
    results['experiment'] = savename

    with open(filename, 'a') as file:
        json.dump(results, file, indent=4, separators=(',', ':'))
    file.close()


def run_crowd_experiments(datapath):
    train_data = np.load(datapath+'data_features.npy', allow_pickle=True)[()]
    crowd_labels = np.load(datapath+'crowd_labels.npy',
                           allow_pickle=True)[()].astype(int)
    train_labels = np.load(datapath+'true_labels.npy', allow_pickle=True)[()]

    learned_labels = get_metal_labels(crowd_labels)
    print("Metal label accuracy is:", accuracy_score(
        train_labels, np.round(learned_labels)))

    majority_model = MajorityLabelVoter()
    majority_vote_labels = majority_model.predict(
        L=crowd_labels,  tie_break_policy='random')
    print("Majority vote accuracy is:", accuracy_score(
        train_labels, majority_vote_labels))


def run_synthetic_experiments(data, savename):
    weak_signals = data['weak_signals'].astype(int)
    train_data, train_labels = data['train']
    test_data, test_labels = data['test']
    weak_signals = np.squeeze(weak_signals, axis=-1).T

    learned_labels = get_metal_labels(weak_signals)
    label_accuracy = accuracy_score(train_labels, learned_labels)
    print("Metal label accuracy is:", label_accuracy)

    model = mlp_model(train_data.shape[1], 1)
    model.fit(train_data, learned_labels, batch_size=32, epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print('Test accuracy: %f' % test_accuracy)

    results = {}
    filename = 'results/metal_results.json'
    results['label_accuracy'] = label_accuracy
    results['test_accuracy'] = test_accuracy
    results['experiment'] = savename

    with open(filename, 'a') as file:
        json.dump(results, file, indent=4, separators=(',', ':'))
    file.close()


if __name__ == '__main__':
    run_synthetic_experiments(synthetic_data(20000, 10), 'synthetic')
    run_experiments('yelp')
    run_experiments('sst-2')
    run_experiments('imbd')
    run_experiments('fashion-mnist')
    # run_experiments('svhn')
    # run_multi_experiments('TREC')

    # print("Running experiments on rte datasets:")
    # run_crowd_experiments("../datasets/rte/")
    # print("Running experiments on wordsim datasets:")
    # run_crowd_experiments("../datasets/wordsim/")
    # print("Running experiments on bluebirds datasets:")
    # run_crowd_experiments("../datasets/bluebirds/")
    # run_crowd_experiments("../../datasets/medical-relations/treats/")
    # run_crowd_experiments("../../datasets/medical-relations/causes/")
