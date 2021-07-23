import numpy as np
import codecs, json, time, sys
import random, gc, time, sys
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras import backend as K
from sklearn.base import clone
from data_utilities import *
from setup_model import *
from sklearn.linear_model import LogisticRegression


def get_supervision_data(dataset, weak_signals='manual', prec=0.1, err=0.1, true_bounds=False):

    data_set = dataset.copy()
    num_classes = data_set['num_classes']
    img_rows, img_cols = data_set['img_rows'], data_set['img_cols']
    channels = data_set['channels']
    categories, path = data_set['categories'], data_set['path']
    train_data, train_labels = data_set['train_data']
    test_data, test_labels = data_set['test_data']

    num_weak_signals = 5  # number of human weak signals to read in
    weak_data = dict()
    model_names = []

    train_data, test_data = preprocess_data(train_data, test_data, scheme='default')

    # get human labeled signals
    human_weak_data = read_weak_signals(categories, path, num_weak_signals)
    human_weak_labels = np.round(human_weak_data['weak_probabilities'])

    num_unlabeled, indexes = get_labeled_index(train_labels, num_classes)

    np.random.seed(2000)
    np.random.shuffle(indexes)
    labeled_data, labeled_labels = train_data[indexes], train_labels[indexes]
    labeled_onehot_labels = to_categorical(labeled_labels, num_classes)

    weak_probabilities = []
    error_bounds = []
    precisions = []

    # reshape train_labels
    train_labels = to_categorical(train_labels, num_classes)
    # calculate true bounds for the human signals
    human_error_rates, human_precisions = get_validation_bounds(train_labels, human_weak_labels)
    n,k = train_labels.shape            # inconsistent with other uses of k
    # print(n, k)
    # print(human_weak_labels.shape)
    # print(data_set.keys())
    # print(num_classes)
    # exit()
    rand_labels = np.random.rand(n,k)

    data = ''
    if weak_signals != 'manual':

        for i, model in enumerate(getNewModel()):
            results = dict()
            modelname = str(model).split('(')[0]
            data_set['modelname'] = modelname

            try:
                data = np.load('results/new_results/'+path+'_'+modelname+'.npy', allow_pickle=True)[()]
                print("Succesfully loaded data...")
                predicted_labels = data['predicted_labels']
                accuracy = data['accuracy']
            except:
                filename = 'results/new_results/'+path+'_'+modelname+'.npy'
                output = {}

                if 'model' in modelname:
                    if modelname == 'convnet_model':
                        model = convnet_model(img_rows, img_cols, channels)
                    elif modelname == 'mlp_model': # added this
                        model = mlp_model(train_data.shape[1], k)
                    else:
                        model = build_model((None,img_rows, img_cols, channels))
                    initial_weights = model.get_weights()
                    # reshaped_data = labeled_data.reshape(labeled_data.shape[0],
                    #                                      img_rows, img_cols, channels)
                    
                    # print(labeled_data.shape)
                    # print(reshaped_data.shape)
                    # print(labeled_onehot_labels.shape)
                    # exit()
                    # model.fit(reshaped_data,
                    #           labeled_onehot_labels,
                    #           batch_size=32,
                    #           epochs=200,
                    #           verbose=1)
                    model.fit(labeled_data,
                              labeled_onehot_labels,
                              batch_size=32,
                              epochs=200,
                              verbose=1)
                    # reshaped_data = train_data.reshape(train_data.shape[0], img_rows,
                    #                                    img_cols, channels)
                    # predicted_labels = model.predict(reshaped_data)
                    predicted_labels = model.predict(train_data)
                    accuracy = accuracy_score(train_labels, predicted_labels)
                    model.set_weights(initial_weights)
                else:
                    modelclone = clone(model)
                    model.fit(labeled_data, labeled_labels)
                    predicted_labels = model.predict_proba(train_data)
                    accuracy = accuracy_score(train_labels, predicted_labels)
                    model = modelclone

                output['predicted_labels'] = predicted_labels
                output['accuracy'] = accuracy
                # np.save(filename, output)

            weak_probabilities.append(predicted_labels)
            print("%s has accuracy of %f" % (modelname, accuracy))

            # true bounds for the pseudolabels
            error_rates, precision = calculate_bounds(train_labels, predicted_labels)
            error_bounds.append(error_rates)
            precisions.append(precision)
            model_names.append(modelname)

        # concatenate pseudolabels and human weak signals
        weak_probabilities = np.concatenate((weak_probabilities, human_weak_labels))
        precisions = np.concatenate((precisions, human_precisions))
        error_bounds = np.concatenate((error_bounds, human_error_rates))

    else:
        weak_probabilities = human_weak_labels
        precisions = np.asarray(human_precisions)
        error_bounds = np.asarray(human_error_rates)

    # reshape the rest of the data
    test_labels = to_categorical(test_labels, num_classes)
    # train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, channels)
    # test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, channels)

    human_model_names = ['human_labels_' + str(i + 1) for i in range(num_weak_signals)]
    model_names.extend(human_model_names)

    num_weak_signals = len(model_names)

    # dump snorkel signals to file
    assert num_weak_signals == weak_probabilities.shape[0]
    # convert_snorkel_signals(weak_probabilities, num_weak_signals, change=True)

    # new modification, comment to use true bounds
    if not true_bounds:
        precisions = np.ones(precisions.shape) * prec
        error_bounds = np.ones(error_bounds.shape) * err
        # error_bounds, precisions = get_estimated_bounds(weak_probabilities)

    weak_signals_mask = weak_probabilities >=0

    weak_data['weak_signals'] = weak_probabilities
    weak_data['active_mask'] = weak_signals_mask
    weak_data['precision'] = np.asarray(precisions)
    weak_data['error_bounds'] = np.asarray(error_bounds)
    weak_data['num_unlabeled'] = num_unlabeled

    data = dict()
    data['img_rows'], data['img_cols'] = img_rows, img_cols
    data['channels'] = channels
    data['model_names'] = model_names
    data['random_labels'] = rand_labels
    data['weak_model'] = weak_data
    data['labeled_labels'] = labeled_onehot_labels
    data['train_data'] = (train_data, train_labels)
    data['test_data'] = (test_data, test_labels)

    return data
