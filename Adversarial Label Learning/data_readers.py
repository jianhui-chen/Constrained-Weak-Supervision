import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json


# ------------------------------------------------------------------------- #
# Code for Loading and processing all data sets                             #
# ------------------------------------------------------------------------- #

def breast_cancer_load_and_process_data(path):

    """Loads breast cancer data from `path`"""
    df = pd.read_csv(path, header=None)
    #Remove the first column of the data (the ID)
    df.drop([0], axis=1, inplace=True)
    #replace labels 'B' with 0 and 'M' with 1
    df[1] = df[1].replace({'B': 0, 'M': 1})
    #Convert to array
    data_matrix = df.to_numpy()

    #Seperate Data and labels 
    data_labels = data_matrix[:, :1].ravel() 
    data_matrix = data_matrix[:, 1:]

    return data_matrix, data_labels


def cardio_load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path)
    #Use class 1 and 2 labels only in the data
    mask_1 = df['CLASS'] == 1
    mask_2 = df['CLASS'] == 2
    df = df[mask_1 | mask_2].copy(deep=True)

    #replace labels '1' with 0 and '2' with 1
    df['CLASS'] = df['CLASS'].replace({1: 0, 2: 1})
    #Convert to array
    data_matrix = df.to_numpy()

    #Seperate Data and labels 
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]

    return data_matrix, data_labels

def obs_load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path, header=None)
    #Use 'NB-Wait' and "'NB-No Block'" labels only in the data
    mask_no_block = df[21] == "'NB-No Block'"
    mask_wait = df[21] == 'NB-Wait'
    df = df[mask_no_block | mask_wait].copy(deep=True)

    # remove rows with missing values
    df = df[df[13] != '?']

    #replace node status feature 'NB', 'B' and 'P NB' with 1, 2, 3
    df[19] = df[19].replace(['NB', 'B', "'P NB'"], [1, 2, 3])

    #replace labels "'NB-No Block'" with 0 and 'NB-Wait' with 1
    df[21] = df[21].replace({"'NB-No Block'": 0, 'NB-Wait': 1})

    #Convert to array
    data_matrix = df.to_numpy()

    #Seperate Data and labels 
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]

    return data_matrix, data_labels


