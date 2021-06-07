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
    #for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals
    views = {0:1, 1:2, 2:20}
    path = 'results/json/obs_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/breast-cancer/wdbc.data', views, default_reader.obs_load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)