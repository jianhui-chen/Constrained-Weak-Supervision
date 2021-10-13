import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime

from data_readers import read_text_data
from constraints import set_up_constraint
from image_utilities import get_image_supervision_data
from load_image_data import load_image_data
from log import Logger, log_results

# Import models
from OldAll import OldAll
from ALL_model import ALL
from cll_model import CLL
from data_consistency_model import DataConsistency

from GEModel import GECriterion 
from PIL import Image


"""
    Binary Datasets:
        1. SST-2
        2. IMDB
        3. Cardio
        4. OBS
        5. Breast Cancer
    
    Multi-Class Datasets:
        1. Fashion

    Algorithms:
        1. Old ALL (Binary labels only)
        2. ALL (Multi label and Abstaining signals supported)
        3. CLL     
        4. Data Consistency
"""


def run_experiments(dataset, set_name, date):
    """ 
        sets up and runs experiments on various algorithms

        Parameters
        ----------
        dataset : dictionary of ndarrays
            contains training set, testing set, and weak signals 
            of read in data
        
        set_name : str
            current name of dataset for logging purposes 

        date : str
            current date and time in format Y_m_d-I:M:S_p

        Returns
        -------
        nothing
    """

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape

    # set up variables
    train_accuracy = []
    test_accuracy = []
    log_name = date + "/" + set_name

    # set up error bounds
    weak_errors = np.ones((m, k)) * 0.01  # default value
    try:
        constraint_set = dataset['weak_errors']
    except KeyError:
        constraint_set = weak_errors

    # cll_setup_weak_errors = multi_all_weak_errors
    constraint_set = set_up_constraint(weak_signals, np.zeros(weak_errors.shape), constraint_set)['error']
    # constraint_set = cll_setup(weak_signals, cll_setup_weak_errors)

    # workaround for an incompatiblity with Binary-Label ALL
    experiment_constraints = [weak_errors, constraint_set, constraint_set, constraint_set]

    # set up algorithms
    experiment_names = ["Binary-Label ALL", "Multi-Label ALL", "CLL", "Data Consistency"]
    
    # instantiate learner objects for all methods
    binary_all = OldAll(max_iter=10000, log_name=log_name + "/BinaryALL")

    if set_name == 'fashion':
        multi_all = ALL(loss='multiclass')
    else:
        multi_all = ALL()
    constrained_labeling = CLL(log_name=log_name+"/CLL")
    data_consistency = DataConsistency(log_name=log_name+"/Const")

    experiment_models = [binary_all, multi_all, constrained_labeling, data_consistency]

    # Loop through each algorithm
    for model_index, model in enumerate(experiment_models):
        print("\n\nRunning experiment using", experiment_names[model_index])

        # skip binary all on multi label or abstaining signal set
        if model_index == 0:
            if set_name == 'sst-2' or set_name == 'imdb' or set_name == 'fashion':
                print("    Skipping binary ALL because data is multi-class or has abstaining signals ")
                train_accuracy.append(0)
                test_accuracy.append(0)
                continue

        model.fit(train_data, weak_signals, experiment_constraints[model_index])

        """Predict_proba"""
        train_probas = model.predict_proba(train_data)
        train_acc = model.get_accuracy(train_labels, train_probas)

        test_probas = model.predict_proba(test_data)
        test_acc = model.get_accuracy(test_labels, test_probas)

        print("\nUsing {}:".format(experiment_names[model_index]))
        print("    Train Accuracy is: ", train_acc)
        print("    Test Accuracy is: ", test_acc)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    print('\n\nLogging results\n\n')
    acc_logger = Logger("logs/" + log_name + "/accuracies")
    plot_path = "./logs/" + log_name
    log_results(train_accuracy, acc_logger, plot_path, 'Accuracy on training data')
    log_results(test_accuracy, acc_logger, plot_path, 'Accuracy on testing data')


if __name__ == '__main__':

    # text and tabular experiments:
    dataset_names = ['sst-2', 'imdb', 'obs', 'cardio', 'breast-cancer']
    # dataset_names = ['obs', 'cardio', 'breast-cancer']

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    for name in dataset_names:
        print("\n\n\n# # # # # # # # # # # #")
        print("#  ", name, "experiment  #")
        print("# # # # # # # # # # # #")
        run_experiments(read_text_data('../datasets/' + name + '/'), name, date)

    # Image experiments
    print("\n\n\n# # # # # # # # # # # #")
    print("#  fashion experiment #")
    print("# # # # # # # # # # # #")
    run_experiments(load_image_data(), 'fashion', date)
