import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader

def load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path+'winequality-white.csv', sep=';')
    df2 = pd.read_csv(path+'winequality-red.csv', sep=';')

    #Concat training and test data
    df = pd.concat([df, df2])

    #Use labels 5 and 6 only in the data
    mask_1 = df['quality'] == 5
    mask_2 = df['quality'] == 6
    df = df[mask_1 | mask_2].copy(deep=True)

    #replace labels 5 with 0 and 6 with 1
    df['quality'] = df['quality'].replace({5:0, 6:1})
    data_matrix = df.values

    #Split the data into 70% training and 30% test set
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]
    train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), 
                                                            test_size=0.3, shuffle=True, random_state=42, stratify=data_labels)
    
    #Normalize the features of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    assert train_labels.size == train_data.shape[0]
    assert test_labels.size == test_data.shape[0]

    data = {}

    val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels.astype('float'),
                                                                                 test_size=0.4285, shuffle=True, random_state=42, stratify=train_labels)

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

    #Use fixed acidity, density, and pH as weak signal views
    views = {0:0, 1:7, 2:8}
    datapath = 'datasets/winequality/'
    savepath = 'results/json/winequality.json'
    default_reader.run_experiment(run, save, views, datapath, load_and_process_data, savepath)
    

def run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """

    #Use fixed acidity, density, and pH as weak signal views
    views = {0:0, 1:7, 2:8}
    path = 'results/json/winequality_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/winequality/', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)