from setup_model import *
from sklearn.linear_model import LogisticRegression
import numpy as np
import codecs, json, time, sys
from sklearn import preprocessing
import random, gc, time, sys
from tensorflow.python.keras import backend as K
from sklearn.base import clone
from data_utilities import *
from sklearn.model_selection import train_test_split
from setup_model import mlp_model
from constraints import build_constraints


def get_text_supervision_data(dataset, supervision='manual', true_bounds=False):

    data_set = dataset.copy()
    train_data = data_set['data_features']
    weak_signals = data_set['weak_signals']
    train_labels = data_set['labels']
    test_data = data_set['test_features']
    test_labels = data_set['test_labels']
    has_test_data = data_set['has_test_data']
    has_labels = data_set['has_labels']

    num_weak_signals = weak_signals.shape[0]
    num_classes = weak_signals.shape[2]

    if not has_test_data:
        # get indices for test data
        n = train_data.shape[0]
        indexes = np.arange(n)
        np.random.seed(2000)
        test_indexes = np.random.choice(n, int(n*0.2), replace=False)
        weak_signals = np.delete(weak_signals, test_indexes, axis=1)

        weak_signals_mask = weak_signals >=0

        if has_labels:
            test_labels = train_labels[test_indexes]
            test_data = train_data[test_indexes]
            train_indexes = np.delete(indexes, test_indexes)
            train_labels = train_labels[train_indexes]
            train_data = train_data[train_indexes]

            # get indexes of labels to train on
            num_unlabeled, indexes = get_labeled_index(train_labels, num_classes)
            np.random.seed(2000)
            np.random.shuffle(indexes)
            labeled_data, labeled_labels = train_data[indexes], train_labels[indexes]
        else:
            test_labels = test_indexes.copy()
            train_labels = np.delete(indexes, test_indexes)
            test_data = train_data[test_indexes]
            train_data = train_data[train_labels]

    else:
        if has_labels:
            # get indexes of labels to train on
            num_unlabeled, indexes = get_labeled_index(train_labels, num_classes)
            np.random.seed(2000)
            np.random.shuffle(indexes)
            labeled_data, labeled_labels = train_data[indexes], train_labels[indexes]
        else:
            train_labels = np.random.choice([0,1],train_data.shape[0])

    # calculate true bounds for the human signals
    true_error_rates, true_precisions = get_validation_bounds(train_labels, weak_signals)

    # define variables
    weak_data = dict()
    model_names = []
    weak_probabilities = []
    error_bounds = []
    precisions = []

    data = ''

    if has_labels:
        if supervision != 'manual':

            for i, model in enumerate(getNewModel()):
                results = dict()
                modelname = str(model).split('(')[0]
                data_set['modelname'] = modelname

                try:
                    data = np.load('results/new_results/'+modelname+'.npy', allow_pickle=True)[()]
                    print("Succesfully loaded data...")
                    predicted_labels = data['predicted_labels']
                    accuracy = data['accuracy']
                except:
                    filename = 'results/new_results/'+modelname+'.npy'
                    output = {}

                    if 'model' in modelname:
                        model = mlp_model(train_data.shape[1], train_labels.shape[1])
                        model.fit(labeled_data,
                                  labeled_labels,
                                  batch_size=32,
                                  epochs=100,
                                  verbose=1)
                        predicted_labels = model.predict(train_data)
                        accuracy = accuracy_score(train_labels, predicted_labels)
                    else:
                        model.fit(labeled_data, np.argmax(labeled_labels,axis=1))
                        predicted_labels = model.predict_proba(train_data)
                        accuracy = accuracy_score(train_labels, predicted_labels)

                    output['predicted_labels'] = predicted_labels
                    output['accuracy'] = accuracy
                    np.save(filename, output)

                weak_probabilities.append(predicted_labels)
                print("%s has accuracy of %f" % (modelname, accuracy))

                # true bounds for the pseudolabels
                error_rates, precision = calculate_bounds(train_labels, predicted_labels)
                error_bounds.append(error_rates)
                precisions.append(precision)
                model_names.append(modelname)

            # concatenate pseudolabels and human weak signals
            weak_probabilities = np.concatenate((weak_probabilities, weak_signals))
            precisions = np.concatenate((precisions, true_precisions))
            error_bounds = np.concatenate((error_bounds, true_error_rates))

        else:
            weak_probabilities = weak_signals
            precisions = np.asarray(true_precisions)
            error_bounds = np.asarray(true_error_rates)

    else:
        m,n,k = weak_signals.shape
        weak_probabilities = weak_signals

    human_model_names = ['human_labels_' + str(i + 1) for i in range(num_weak_signals)]
    model_names.extend(human_model_names)

    num_weak_signals = len(model_names)

    # new modification, comment to use true bounds
    if not true_bounds:
        m,n,k = weak_probabilities.shape
        err = 0.01
        precisions = np.ones((m,k)) * 0.6
        error_bounds = np.ones((m,k)) * err
        # error_bounds, precisions = get_estimated_bounds(weak_probabilities)

  
    weak_signals_mask = weak_probabilities >=0
    weak_data['weak_signals'] = weak_probabilities
    weak_data['active_mask'] = weak_signals_mask
    weak_data['precision'] = np.asarray(precisions)
    weak_data['error_bounds'] = np.asarray(error_bounds)

    data = dict()
    data['model_names'] = model_names
    data['weak_model'] = weak_data
    data['train_data'] = train_data, train_labels
    data['test_data'] = test_data, test_labels
    data['has_labels'] = has_labels, has_test_data

    return data
