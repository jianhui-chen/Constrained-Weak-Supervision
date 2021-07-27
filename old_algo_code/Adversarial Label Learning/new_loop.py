import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import getAccuracy, runBaselineTests
from sklearn.metrics import accuracy_score
from log import Logger, log_accuracy
import json
import math

from models import ALL, LabelEstimator, GECriterion


def new_run_experiment(data_obj, multiple_weak_signals, constant_bound=False):
    """
        Runs experiment with the given dataset
        :param data: dictionary of validation and test data
        :type data: dict
        :param multiple_weak_signals: 
        :type: 
    """

    experiment_names = ["ALL w/ constant bounds", "ALL w/ computed bounds", "Avergaing Baseline", "GE Crit"]

    # retrieve data from object
    data = data_obj.data
    dev_data = data['dev_data'][0].T
    dev_labels = data['dev_data'][1]
    train_data, train_labels = data['train_data']
    train_data = train_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = dev_data.shape


    # loop through all weak signals provided to preform expirments 
    for num_loops, weak_signals in enumerate(multiple_weak_signals, 1): #begins from 1

        
        weak_signal_ub            = weak_signals['error_bounds']
        weak_signal_probabilities = weak_signals['probabilities']

        train_accuracy            = []
        test_accuracy             = []

        # Create new models for experiments
        all_model_const = ALL()
        all_model       = ALL()
        ave_model       = LabelEstimator()
        ge_model        = GECriterion()
        models = [all_model_const, all_model, ave_model, ge_model]

        # loop through models and train them
        for model_np, model in enumerate(models):

            # fit the model depening on what needs to be provided 
            print("Running: " + experiment_names[model_np] + " with " + str(num_loops) + " weak signals...")
            if model_np == 0:
                model = model.fit(train_data, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3)
            elif model_np == 3:
                model = model.fit(train_data, weak_signal_probabilities, weak_signal_ub, train_labels)
            else:
                model = model.fit(train_data, weak_signal_probabilities, weak_signal_ub)

            # Calculate and Report accuracy
            train_probas = model.predict_proba(train_data)
            test_probas = model.predict_proba(test_data)
          
            train_acc = model.get_accuracy(train_labels, train_probas)
            test_acc = model.get_accuracy(test_labels, test_probas)

            print("The accuracy on the train data is", train_acc)
            print("The accuracy on the test data is", test_acc)
            print("\n\n")

            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

        # Log the accuracy results 
        logger = Logger("logs/All/ Accuracies with " + str(num_loops) + " Weak signal")
        log_accuracy(logger, train_accuracy, 'Accuracy on Validation Data')
        log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')


        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")


  