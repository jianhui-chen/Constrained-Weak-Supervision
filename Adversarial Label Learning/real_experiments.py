import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import runBaselineTests, getModelAccuracy, getWeakSignalAccuracy, runBaselineTests
from sklearn.metrics import accuracy_score
from log import Logger, log_accuracy
from models import ALL
import json

from models import ALL, LabelEstimator, GECriterion

def run_experiment(data_obj, multiple_weak_signals):
    """
        Runs experiment with the given dataset
        :param data: dictionary of validation and test data
        :type data: dict
        :param multiple_weak_signals: collection of weak signal probabilities and error bounds
        :type: list of dictionaries, one entry for each weaksignal
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





def bound_experiment(data_obj, w_data_dict):
    # """
    # Runs experiment with the given dataset
    # :param data: dictionary of train and test data
    # :type data: dict
    # :param weak_signal_data: data representing the different views for the weak signals
    # :type: array
    # :param num_weak_signal: number of weak signals
    # :type num_weak_signal: int
    # :param bound: error bound of the weak signal
    # :type bound: int
    # :return: outputs from the bound experiments
    # :rtype: dict
    # """

    """
    :param data: dictionary of train and test data
    :type: dict
    :param weak_signal_data: data representing the different views for the weak signals
    :type: array
    :param path: relative path to save the bounds experiment results
    :type: string
    """

    data = data_obj.data
    path = data_obj.sp

    # set up your variables
    num_weak_signal = 3
    num_experiments = 100

    results = {}
    results['Error bound'] = []
    results['Accuracy'] = []
    results['Ineq constraint'] = []
    results['Weak_signal_ub'] = []
    results['Weak_test_accuracy'] = []

    bounds = np.linspace(0, 1, num_experiments)

    for i in range(num_experiments):
        logger = Logger("logs/bound/" + data_obj.n + "/" + str(i))

        dev_data = data['dev_data'][0].T
        training_labels = data['dev_data'][1]
        train_data, train_labels = data['train_data']
        train_data = train_data.T
        test_data = data['test_data'][0].T
        test_labels = data['test_data'][1]

        num_features, num_data_points = dev_data.shape

        weak_signal_ub = w_data_dict['error_bounds']
        weak_signal_probabilities = w_data_dict['probabilities']

        weights = np.zeros(num_features)

        print("Running tests...")

        optimized_weights, ineq_constraint = train_all(train_data, weights, weak_signal_probabilities, bounds[i], logger, max_iter=10000)

        # calculate test probabilities
        test_probabilities = probability(test_data, optimized_weights)
        # calculate error bound on test data
        test_probabilities = np.round(test_probabilities)
        error_bound = (test_probabilities.dot(1 - test_labels) + (1 - test_probabilities).dot(test_labels)) / test_labels.size
        test_accuracy = getModelAccuracy(test_probabilities, test_labels)

        print("")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("The error_bound of learned model on the weak signal is", weak_signal_ub)
        print("The test accuracy of the weak signal is", w_data_dict['test_accuracy'])
        print("The error_bound of learned model on the test data is", error_bound[0])
        print("The accuracy of the model on the test data is", test_accuracy)

        # Save results for this experiement
        results['Error bound'].append(error_bound[0])
        results['Accuracy'].append(test_accuracy)
        results['Ineq constraint'].append(ineq_constraint[0])
        results['Weak_signal_ub'].append(weak_signal_ub[0])
        results['Weak_test_accuracy'].append(w_data_dict['test_accuracy'][0])

    print("Saving results to file...")

    with open(path, 'w') as file:
        json.dump(results, file, indent=4, separators=(',', ':'))
    file.close()


def dependent_error_exp(data_obj, w_data_dicts):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    :param data_and_weak_signal_data: tuple of data and weak signal data
    :type: tuple
    :param path: relative path to save the bounds experiment results
    :type: string
    """

    data = data_obj.data
    path = data_obj.sp

    # set up your variables
    #num_experiments = 10

    accuracy = {}
    accuracy ['ALL'] = []
    accuracy['GE'] = []
    accuracy['BASELINE'] = []
    accuracy ['WS'] = []

    # all_accuracy = []
    # baseline_accuracy = []
    # ge_accuracy = []
    # weak_signal_accuracy = []

    for num_weak_signals, w_data_dict in enumerate(w_data_dicts, 1):

        logger = Logger("logs/error/" + data_obj.n + "/" + str(num_weak_signals))



        dev_data = data['dev_data'][0].T
        training_labels = data['dev_data'][1]
        train_data, train_labels = data['train_data']
        train_data = train_data.T
        test_data = data['test_data'][0].T
        test_labels = data['test_data'][1]

        num_features, num_data_points = dev_data.shape

        weak_signal_ub = w_data_dict['error_bounds']
        weak_signal_probabilities = w_data_dict['probabilities']
        weak_test_accuracy = w_data_dict['test_accuracy']

        weights = np.zeros(num_features)

        print("Running tests...")

        optimized_weights, ineq_constraint = train_all(train_data, weights, weak_signal_probabilities, weak_signal_ub, logger, max_iter=5000)

        # calculate test probabilities
        test_probabilities = probability(test_data, optimized_weights)
        # calculate test accuracy
        test_accuracy = getModelAccuracy(test_probabilities, test_labels)

        print("")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Experiment %d"%num_weak_signals)
        print("We trained %d learnable classifiers with %d weak signals" %(1, num_weak_signals))
        print("The accuracy of the model on the test data is", test_accuracy)
        print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
        print("")

        # calculate ge criteria
        print("Running tests on ge criteria...")
        ge_results = ge_criterion_train(train_data.T, train_labels, weak_signal_probabilities, num_weak_signals)
        ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, ge_results)))
        print("The accuracy of ge criteria on test data is", ge_test_accuracy)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # calculate baseline
        print("Running tests on the baselines...")
        baselines = runBaselineTests(train_data, weak_signal_probabilities)
        b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
        print("The accuracy of the baseline models on test data is", b_test_accuracy)
        print("")

        accuracy['ALL'].append(test_accuracy)
        accuracy['GE'].append( w_data_dict['test_accuracy'][-1])
        accuracy['BASELINE'].append(ge_test_accuracy)
        accuracy['WS'].append(b_test_accuracy[-1] )
    


    print("Saving results to file...")
    filename = path

    with open(filename, 'w') as file:
        json.dump(accuracy, file, indent=4, separators=(',', ':'))
    file.close()
