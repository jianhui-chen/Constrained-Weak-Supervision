import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader


def run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """
    #Use AC, MLTV and Median as weak signal views
    views = {0:1, 1:10, 2:18}
    path = 'results/json/cardio_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/cardiotocography/cardio.csv', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)


def run_dep_error_exp(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """
    #Use AC, MLTV and Median as weak signal views
    views = {0:1, 1:18}
    # repeat the bad weak signal 
    for i in range(2,10):
        views[i] = 18
    path = 'results/json/cardio_error.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/cardiotocography/cardio.csv', views, load_and_process_data)
    default_reader.run_dep_error_exp(run, data_and_weak_signal_data, path)