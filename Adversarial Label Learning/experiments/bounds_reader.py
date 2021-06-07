import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader



def breast_cancer_run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """

    #for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
    views = {0:0, 1:10, 2:20}
    path = 'results/json/bc_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/breast-cancer/wdbc.data', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)


def cardio_run_bounds_experiment(run):

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