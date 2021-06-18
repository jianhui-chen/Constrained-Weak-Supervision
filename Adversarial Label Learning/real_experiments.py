import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import runBaselineTests, getModelAccuracy, getWeakSignalAccuracy
from sklearn.metrics import accuracy_score
from log import Logger, log_accuracy
import json



def run_experiment(data_obj, w_data_dicts, constant_bound=False):
    """
    Runs experiment with the given dataset
    :param data: dictionary of validation and test data
    :type data: dict
    :param w_data_dicts: 
    :type: 
    """

    adversarial_data_dicts = []
    weak_models = []

    data = data_obj.data

    for num_weak_signals, w_data_dict in enumerate(w_data_dicts, 1): #begins from 1
        # initializes logger
        logger = Logger("logs/standard/" + data_obj.n + "/" + str(num_weak_signals))


        training_data = data['training_data'][0].T
        training_labels = data['training_data'][1]
        val_data, val_labels = data['validation_data']
        val_data = val_data.T
        test_data = data['test_data'][0].T
        test_labels = data['test_data'][1]

        num_features, num_data_points = training_data.shape

        weak_signal_ub = w_data_dict['error_bounds']
        # weak_signal_ub = np.ones(w_data_dict['error_bounds'].shape) * 0.3
        
        # Following line doesn't seem to be used anywhere?
        #models = w_data_dict['models']
        weak_signal_probabilities = w_data_dict['probabilities']

        weights = np.zeros(num_features)

        print("Running tests...")
        if constant_bound:
            optimized_weights, y = train_all(val_data, weights, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3, logger, max_iter=10000)
        else:
            optimized_weights, y = train_all(val_data, weights, weak_signal_probabilities, weak_signal_ub, logger, max_iter=10000)

        # calculate validation results
        learned_probabilities = probability(val_data, optimized_weights)
        validation_accuracy = getModelAccuracy(learned_probabilities, val_labels)

        # calculate test results
        learned_probabilities = probability(test_data, optimized_weights)
        test_accuracy = getModelAccuracy(learned_probabilities, test_labels)

        # calculate weak signal results
        weak_val_accuracy = w_data_dict['validation_accuracy']
        weak_test_accuracy = w_data_dict['test_accuracy']

        adversarial_data_dict = {}
        adversarial_data_dict['validation_accuracy'] = validation_accuracy
        adversarial_data_dict['test_accuracy'] = test_accuracy

        weak_model = {}
        weak_model['num_weak_signals'] = num_weak_signals
        weak_model['validation_accuracy'] = weak_val_accuracy
        weak_model['test_accuracy'] = weak_test_accuracy

        print("")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("We trained %d learnable classifiers with %d weak signals" %(1, num_weak_signals))
        print("The accuracy of learned model on the validatiion data is", validation_accuracy)
        print("The accuracy of weak signal(s) on the validation data is", weak_val_accuracy)
        print("The accuracy of the model on the test data is", test_accuracy)
        print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
        print("")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # calculate baseline
        print("Running tests on the baselines...")
        baselines = runBaselineTests(val_data, weak_signal_probabilities) #remove the transpose to enable it run
        b_validation_accuracy = getWeakSignalAccuracy(val_data, val_labels, baselines)
        b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
        print("The accuracy of the baseline models on test data is", b_test_accuracy)
        print("")
        weak_model['baseline_val_accuracy'] = b_validation_accuracy
        weak_model['baseline_test_accuracy'] = b_test_accuracy
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # calculate ge criteria
        print("Running tests on ge criteria...")
        model = ge_criterion_train(val_data.T, val_labels, weak_signal_probabilities, num_weak_signals)
        ge_validation_accuracy = accuracy_score(val_labels, np.round(probability(val_data, model)))
        ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
        print("The accuracy of ge criteria on validation data is", ge_validation_accuracy)
        print("The accuracy of ge criteria on test data is", ge_test_accuracy)
        weak_model['gecriteria_val_accuracy'] = ge_validation_accuracy
        weak_model['gecriteria_test_accuracy'] = ge_test_accuracy
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        adversarial_data_dicts.append(adversarial_data_dict)
        weak_models.append(weak_model)

        val_accuracy = [adversarial_data_dict['validation_accuracy'], weak_model['validation_accuracy'][num_weak_signals - 1], weak_model['baseline_val_accuracy'][0], weak_model['gecriteria_val_accuracy']]
        log_accuracy(logger, val_accuracy, 'Accuracy on Validation Data')

        test_accuracy = [adversarial_data_dict['test_accuracy'], weak_model['test_accuracy'][num_weak_signals - 1], weak_model['baseline_test_accuracy'][0], weak_model['gecriteria_test_accuracy']]
        log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')

    # Old code
    # log_accuracy(data_obj, 3, adversarial_data_dicts, weak_models)

    return adversarial_data_dicts, weak_models






def bound_experiment(data_obj, w_data_dict):
    # """
    # Runs experiment with the given dataset
    # :param data: dictionary of validation and test data
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
    :param data: dictionary of validation and test data
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

        training_data = data['training_data'][0].T
        training_labels = data['training_data'][1]
        val_data, val_labels = data['validation_data']
        val_data = val_data.T
        test_data = data['test_data'][0].T
        test_labels = data['test_data'][1]

        num_features, num_data_points = training_data.shape

        weak_signal_ub = w_data_dict['error_bounds']
        weak_signal_probabilities = w_data_dict['probabilities']

        weights = np.zeros(num_features)

        print("Running tests...")

        optimized_weights, ineq_constraint = train_all(val_data, weights, weak_signal_probabilities, bounds[i], logger, max_iter=10000)

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



        training_data = data['training_data'][0].T
        training_labels = data['training_data'][1]
        val_data, val_labels = data['validation_data']
        val_data = val_data.T
        test_data = data['test_data'][0].T
        test_labels = data['test_data'][1]

        num_features, num_data_points = training_data.shape

        weak_signal_ub = w_data_dict['error_bounds']
        weak_signal_probabilities = w_data_dict['probabilities']
        weak_test_accuracy = w_data_dict['test_accuracy']

        weights = np.zeros(num_features)

        print("Running tests...")

        optimized_weights, ineq_constraint = train_all(val_data, weights, weak_signal_probabilities, weak_signal_ub, logger, max_iter=5000)

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
        model = ge_criterion_train(val_data.T, val_labels, weak_signal_probabilities, num_weak_signals)
        ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
        print("The accuracy of ge criteria on test data is", ge_test_accuracy)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # calculate baseline
        print("Running tests on the baselines...")
        baselines = runBaselineTests(val_data, weak_signal_probabilities)
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
