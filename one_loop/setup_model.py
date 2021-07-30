import numpy as np
import pandas as pd
import codecs, datetime, glob, itertools, os
import re, sklearn, string, sys, time
import random
import codecs, json
import tensorflow as tf
from tensorflow.python.keras import backend as K, regularizers, optimizers
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Activation, Dropout, Flatten, Dense, InputLayer
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.regularizers import L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from data_utilities import *


def read_weak_signals(categories, path, num_weak_signals):
    """
    Reads in the generated human weak signals

    :param categories: list containing the label categories
    :type data: list
    :param feature_size: path to generated weak_signals
    :type feature_size: string
    :return: dictionary of different factors
    :rtype: dict
    """

    noisy_data = {}
    num_classes = 10  # change this if more than 10 classes

    # file_path = 'weak_image_labeling/weak_signals/'
    file_path = '../datasets/image_data_weak_signals/'
    weak_signals = [[] for _ in range(num_weak_signals)]
    snorkel_signals = [[] for _ in range(num_classes)]
    error_rates = [[] for _ in range(num_weak_signals)]
    precisions = [[] for _ in range(num_weak_signals)]
    k = 0

    for category in categories:
        for i in range(num_weak_signals):
            current_path = path + '/' + category + '_signal_' + str(
                i) + '.json'
            current_path = os.path.join(file_path, current_path)

            weak_signal_dict = codecs.open(current_path, 'r',
                                           encoding='utf-8').read()
            weak_signal_dict = json.loads(weak_signal_dict)

            weak_signal = np.asarray(weak_signal_dict['weak_signal'])
            error_rate = 1 - weak_signal_dict["provide_accuracy"]
            precision = weak_signal_dict["provided_precision"]

            weak_signals[i].append(weak_signal)
            error_rates[i].append(error_rate)
            precisions[i].append(precision)

            snorkel_labels = np.round(weak_signal)
            snorkel_labels[snorkel_labels == 0] = -1
            snorkel_signals[k].append(snorkel_labels.tolist())

        k += 1

    noisy_data['weak_probabilities'] = np.asarray(weak_signals).transpose(
        0, 2, 1)
    noisy_data['error_bounds'] = np.asarray(error_rates)
    noisy_data['precision'] = np.asarray(precisions)
    noisy_data['snorkel_signals'] = snorkel_signals

    return noisy_data


def get_labeled_index(train_labels, num_classes, validation=0.01):

    indexes = []
    num_data_points = int(validation * train_labels.size / num_classes)

    if num_classes == 1:
        num_classes = 2

    # Get datapoints to evaluate bounds on
    for label in range(num_classes):
        index = get_datapoints(train_labels, num_data_points, label)
        indexes = np.append(indexes, index)

    indexes = indexes.astype(int)
    num_unlabeled = int(train_labels.size - indexes.size)

    return num_unlabeled, indexes


def calculate_bounds(true_labels, predicted_labels, mask=None):
    if len(true_labels.shape) == 1:
        predicted_labels = predicted_labels.ravel()
    assert predicted_labels.shape == true_labels.shape

    if mask is None:
        mask = np.ones(predicted_labels.shape)
    if len(true_labels.shape) == 1:
        mask = mask.ravel()
    assert predicted_labels.shape == mask.shape

    error_rate = true_labels*(1-predicted_labels) + predicted_labels*(1-true_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.sum(error_rate*mask, axis=0) / np.sum(mask, axis=0)
        error_rate = np.nan_to_num(error_rate)

    precision = true_labels * predicted_labels
    precision = np.sum(precision*mask, axis=0) / (np.sum(predicted_labels*mask, axis=0)+ 1e-8)

    # check results are scalars
    if np.isscalar(error_rate):
        error_rate = np.asarray([error_rate])
        precision = np.asarray([precision])

    return error_rate, precision


def get_validation_bounds(true_labels, weak_probabilities):
    # calculate stats on evaluation_set
    error_rates = []
    precisions = []
    mask = weak_probabilities >= 0

    for i, weak_probs in enumerate(weak_probabilities):
        active_mask = mask[i]
        error_rate, precision = calculate_bounds(true_labels, weak_probs, active_mask)
        error_rates.append(error_rate)
        precisions.append(precision)

    return error_rates, precisions


def preprocess_data(train_data, test_data, scheme=None):

    if scheme == 'normalize':
        # mean center data with unit variance
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    elif scheme == '0-1_range':
        # scale to a standard range
        scaler = preprocessing.MinMaxScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    elif scheme == 'max_scale':
        # scale to a standard range
        train_data = train_data / np.max(np.abs(train_data))
        test_data = test_data / np.max(np.abs(test_data))
    else:
        normalize = 1 / 255
        train_data = train_data * normalize
        test_data = test_data * normalize

    return train_data, test_data

def build_constraints(a_matrix, bounds):
    # a_matrix left hand matrix of the inequality size: m x n x k type: ndarray
    # bounds right hand vectors of the inequality size: m x n type: ndarray
    # dictionary containing constraint vectors

    m, n, k = a_matrix.shape
    assert (m,k) == bounds.shape, \
    "The constraint matrix shapes don't match"

    constraints = dict()
    constraints['A'] = a_matrix
    constraints['b'] = bounds
    constraints['gamma'] = np.zeros(bounds.shape)

    # temp for now
    constraints['c'] = np.zeros(a_matrix.shape)

    return constraints


def set_up_constraint(weak_probabilities, precision, error_bounds):
    # set up constraints
    # new modifications to account for abstaining weak_signals
    constraint_set = dict()
    m, n, k = weak_probabilities.shape
    precision_amatrix = np.zeros((m, n, k))
    error_amatrix = np.zeros((m, n, k))
    constants = []

    for i, weak_signal in enumerate(weak_probabilities):
        active_signal = weak_signal >=0
        precision_amatrix[i] = -1 * weak_signal * active_signal / (np.sum(active_signal*weak_signal, axis=0) + 1e-8)
        error_amatrix[i] = (1 - 2 * weak_signal) * active_signal

        # error denom to check abstain signals
        error_denom = np.sum(active_signal, axis=0)
        error_amatrix[i] /= error_denom

        # constants for error constraints
        constant = (weak_signal*active_signal) / error_denom
        constants.append(constant)

    # set up precision upper bounds constraints
    bounds = -1 * precision
    precision_set = build_constraints(precision_amatrix, bounds)
    constraint_set['precision'] = precision_set

    # set up error upper bounds constraints
    constants = np.sum(constants, axis=1)
    assert len(constants.shape) == len(error_bounds.shape)
    bounds = error_bounds - constants
    error_set = build_constraints(error_amatrix, bounds)
    constraint_set['error'] = error_set

    # print(constants.shape)
    # print(constant)
    # print(error_amatrix.shape)
    # print(bounds.shape)
    # exit()

    # print(error_set)
    # exit()

    return constraint_set

def getNewModel():
    # yield 'build_model()'
    # yield 'convnet_model()'
    yield 'mlp_model()'
    # yield LogisticRegression(max_iter=1000, solver='lbfgs')


def writeToFile(data, filename):
    json.dump(data,
              codecs.open(filename, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)


def accuracy_score(y_true, y_pred):
    try:
        n,k = y_true.shape
        if k>1:
            assert y_true.shape == y_pred.shape
            return np.mean(np.equal(np.argmax(y_true, axis=-1),
                          np.argmax(y_pred, axis=-1)))
    except:
        if len(y_true.shape)==1:
            y_pred = np.round(y_pred.ravel())

    assert y_true.shape == y_pred.shape
    return np.mean(np.equal(y_true, np.round(y_pred)))



def prepare_mmce(weak_signals, labels):
    ### convert weak_signals to format for mmce ###
    crowd_labels = np.zeros(weak_signals.shape)
    true_labels = labels.copy()
    try:
        n,k = true_labels.shape
    except:
        k = 1
    crowd_labels[weak_signals==1] = 2
    crowd_labels[weak_signals==0] = 1
    if k > 1:
        true_labels = np.argmax(true_labels, axis=1)
    true_labels +=1

    if len(crowd_labels.shape) > 2:
        assert crowd_labels.any() != 0
        m,n,k = crowd_labels.shape
        if k>1:
            for i in range(k):
                crowd_labels[:,:,i] = i+1
            crowd_labels[weak_signals==-1] = 0
        crowd_labels = crowd_labels.transpose((1,0,2))
        crowd_labels = crowd_labels.reshape(n,m*k)
    return crowd_labels.astype(int), true_labels.ravel().astype(int)


def convert_snorkel_signals(weak_signal_probs, num_weak_signals, change=True):

    snorkel_matrix = np.rint(weak_signal_probs.transpose(2, 0, 1))
    weak_signal_probs = np.rint(weak_signal_probs.transpose(2, 0, 1))
    if change:
        snorkel_matrix[weak_signal_probs == 0] = -1
        snorkel_matrix[weak_signal_probs == -1] = 0

    file_path = 'results/snorkel/' + str(num_weak_signals) + '_signal_matrix.json'
    assert snorkel_matrix.shape[1] == num_weak_signals
    writeToFile(snorkel_matrix.tolist(), file_path)

    # # iteratively add signals
    # for i in range(1, num_weak_signals+1):
    #     index = [k for k in range(i, num_weak_signals)]
    #     snorkel_signal = np.delete(snorkel_matrix, index, axis=1)
    #     # Write snorkel matrix to a file
    #     number = int(num_weak_signals - len(index))
    #     file_path = 'results/snorkel/' + str(number) + '_signal_matrix.json'
    #     assert snorkel_signal.shape[1] == number
    #     writeToFile(snorkel_signal.tolist(), file_path)


def mlp_model(dimension, output):

    actv = 'softmax' if output > 1 else 'sigmoid'
    loss = 'categorical_crossentropy' if output > 1 else 'binary_crossentropy'
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dimension,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation=actv))

    model.compile(loss=loss,
                  optimizer='adam', metrics=['accuracy'])

    return model
