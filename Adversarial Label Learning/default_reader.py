import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json

def run_dep_error_exp(run, data_and_weak_signal_data, path):

	"""
	:param run: method that runs real experiment given data
	:type: function
	:return: none
	:param data_and_weak_signal_data: tuple of data and weak signal data
	:type: tuple
	:param path: relative path to save the bounds experiment results
	:type: string
	"""

	# set up your variables
	num_experiments = 10

	all_accuracy = []
	baseline_accuracy = []
	ge_accuracy = []
	weak_signal_accuracy = []

	data, weak_signal_data = data_and_weak_signal_data

	for num_weak_signal in range(num_experiments):
	    output = run(data, weak_signal_data, num_weak_signal + 1)
	    all_accuracy.append(output['ALL'])
	    baseline_accuracy.append(output['AVG'])
	    ge_accuracy.append(output['GE'])
	    weak_signal_accuracy.append(output['WS'])

	print("Saving results to file...")
	filename = path

	output = {}
	output ['ALL'] = all_accuracy
	output['GE'] = ge_accuracy
	output['AVG'] = baseline_accuracy
	output ['WS'] = weak_signal_accuracy

	with open(filename, 'w') as file:
	    json.dump(output, file, indent=4, separators=(',', ':'))
	file.close()



def run_bounds_experiment(run, data_and_weak_signal_data, path):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    :param data_and_weak_signal_data: tuple of data and weak signal data
    :type: tuple
    :param path: relative path to save the bounds experiment results
    :type: string
    """

    data, weak_signal_data = data_and_weak_signal_data

    # set up your variables
    num_weak_signal = 3
    num_experiments = 100
    errors = []
    accuracies = []
    ineq_constraints = []
    weak_signal_ub = []
    weak_test_accuracy = []

    bounds = np.linspace(0, 1, num_experiments)

    for i in range(num_experiments):
        output = run(data, weak_signal_data, num_weak_signal, bounds[i])
        errors.append(output['error_bound'])
        accuracies.append(output['test_accuracy'])
        ineq_constraints.append(output['ineq_constraint'])
        weak_signal_ub.append(output['weak_signal_ub'])
        weak_test_accuracy.append(output['weak_test_accuracy'])

    print("Saving results to file...")
    filename = path

    output = {}
    output ['Error bound'] = errors
    output['Accuracy'] = accuracies
    output['Ineq constraint'] = ineq_constraints
    output ['Weak_signal_ub'] = weak_signal_ub
    output['Weak_test_accuracy'] = weak_test_accuracy
    with open(filename, 'w') as file:
        json.dump(output, file, indent=4, separators=(',', ':'))
    file.close()



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
    data_matrix = df.values
    # look into using df.to_numpy instead !!!!!!!

    #Seperate Data and labels 
    data_labels = data_matrix[:, :1].ravel() 
    data_matrix = data_matrix[:, 1:]

    print(data_labels)

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
    data_matrix = df.values

   #Split the data into 70% training and 30% test set
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
    data_matrix = df.values

    #Split the data into 70% training and 30% test set
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]

    return data_matrix, data_labels


