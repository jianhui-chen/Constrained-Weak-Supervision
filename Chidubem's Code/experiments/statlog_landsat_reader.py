import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader

def load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path+'sat.trn', header=None, sep=' ')
    df2 = pd.read_csv(path+'sat.tst', header=None, sep=' ')

    #Concat training and test data
    df = pd.concat([df, df2])

    #Use class red soil and very damp grey soil labels only in the data
    mask_1 = df[36] == 1
    mask_2 = df[36] == 7
    df = df[mask_1 | mask_2].copy(deep=True)

    #replace labels 7 with 0
    df[36] = df[36].replace({7:0})
    data_matrix = df.values

    #Split the data into 70% training and 30% test set
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]
    train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)

    #Normalize the features of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    assert train_labels.size == train_data.shape[0]
    assert test_labels.size == test_data.shape[0]

    data = {}

    val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_labels)

    data['training_data'] = weak_supervision_data, weak_supervision_labels
    data['validation_data'] = val_data, val_labels
    data['test_data'] = test_data, test_labels

    return data


def run_experiment(run, save):

    """
    :param run: method that runs real experiment given data
    :type: function
    :param save: method that saves experiment results to JSON file
    :type: function
    :return: none
    """

    #Use 1st, middle and last features as weak signal views
    views = {0:0, 1:17, 2:35}
    datapath = 'datasets/statlog-landsite-satellite/'
    savepath = 'results/json/statlog-satellite.json'
    default_reader.run_experiment(run, save, views, datapath, load_and_process_data, savepath)


def run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """
    views = {0:0, 1:17, 2:35}
    path = 'results/json/statlog_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/statlog-landsite-satellite/', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)