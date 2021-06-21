import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import runBaselineTests, getModelAccuracy, getWeakSignalAccuracy
from sklearn.metrics import accuracy_score
from log import Logger, log_accuracy
import json
import math



def new_run_experiment(data_obj, w_data_dicts, constant_bound=False):
    """
    Runs experiment with the given dataset
    :param data: dictionary of validation and test data
    :type data: dict
    :param w_data_dicts: 
    :type: 
    """

    adversarial_acc_dicts = []
    w_acc_dicts = []

    experiment_names = ["ALL w/ constant bounds", "ALL w/ computed bounds", "Avergaing Baseline"]

    num_experoments = 7

    w_data_dicts = [w_data_dicts[0], w_data_dicts[1], w_data_dicts[2], w_data_dicts[0], w_data_dicts[1], w_data_dicts[2], w_data_dicts[2]]

    # # Loops
    # #     1. ALL w/ constant bounds and 1 weak signal
    # #     2. ALL w/ constant bounds and 2 weak signals
    # #     3. ALL w/ constant bounds and 3 weak signals
    # #     4. ALL w/ computed bounds and 1 weak signal
    # #     5. ALL w/ computed bounds and 2 weak signals
    # #     6. ALL w/ computed bounds and 3 weak signals
    # #     7. Avergaing Baseline



    data = data_obj.data

    training_data = data['training_data'][0].T
    training_labels = data['training_data'][1]
    val_data, val_labels = data['validation_data']
    val_data = val_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = training_data.shape


    for num_loops, w_data_dict in enumerate(w_data_dicts, 1): #begins from 1


        # Set up variables and logger
        curr_expirment   = math.floor((num_loops - 1) / 3 )
        num_weak_signals = (num_loops - 1) % 3 + 1
        logger = Logger("logs/standard/" + data_obj.n + "/" + experiment_names[curr_expirment] + " with " + str(num_weak_signals) + " weak signals")

        weak_signal_ub = w_data_dict['error_bounds']
        # weak_signal_ub = np.ones(w_data_dict['error_bounds'].shape) * 0.3
        weak_signal_probabilities = w_data_dict['probabilities']

        # # Following line doesn't seem to be used anywhere?
        # #models = w_data_dict['models']

        weights = np.zeros(num_features)

        print("Running tests...")

        
        # if constant_bound:
        #     optimized_weights, y = train_all(val_data, weights, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3, logger, max_iter=10000)
        # else:
        #     optimized_weights, y = train_all(val_data, weights, weak_signal_probabilities, weak_signal_ub, logger, max_iter=10000)

        # # calculate validation results
        # learned_probabilities = probability(val_data, optimized_weights)
        # validation_accuracy = getModelAccuracy(learned_probabilities, val_labels)

        # # calculate test results
        # learned_probabilities = probability(test_data, optimized_weights)
        # test_accuracy = getModelAccuracy(learned_probabilities, test_labels)

        # # calculate weak signal results
        # weak_val_accuracy = w_data_dict['validation_accuracy']
        # weak_test_accuracy = w_data_dict['test_accuracy']

        # adversarial_acc_dict = {}
        # adversarial_acc_dict['validation_accuracy'] = validation_accuracy
        # adversarial_acc_dict['test_accuracy'] = test_accuracy

        # w_acc_dict = {}
        # w_acc_dict['num_weak_signals'] = num_weak_signals
        # w_acc_dict['validation_accuracy'] = weak_val_accuracy
        # w_acc_dict['test_accuracy'] = weak_test_accuracy

        # print("")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("We trained %d learnable classifiers with %d weak signals" %(1, num_weak_signals))
        # print("The accuracy of learned model on the validatiion data is", validation_accuracy)
        # print("The accuracy of weak signal(s) on the validation data is", weak_val_accuracy)
        # print("The accuracy of the model on the test data is", test_accuracy)
        # print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
        # print("")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # # calculate baseline
        # print("Running tests on the baselines...")
        # baselines = runBaselineTests(val_data, weak_signal_probabilities) #remove the transpose to enable it run
        # b_validation_accuracy = getWeakSignalAccuracy(val_data, val_labels, baselines)
        # b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
        # print("The accuracy of the baseline models on test data is", b_test_accuracy)
        # print("")
        # w_acc_dict['baseline_val_accuracy'] = b_validation_accuracy
        # w_acc_dict['baseline_test_accuracy'] = b_test_accuracy
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # adversarial_acc_dicts.append(adversarial_acc_dict)
        # w_acc_dicts.append(w_acc_dict)

        # val_accuracy = [adversarial_acc_dict['validation_accuracy'], w_acc_dict['validation_accuracy'][num_weak_signals - 1], w_acc_dict['baseline_val_accuracy'][0], w_acc_dict['gecriteria_val_accuracy']]
        # log_accuracy(logger, val_accuracy, 'Accuracy on Validation Data')

        # test_accuracy = [adversarial_acc_dict['test_accuracy'], w_acc_dict['test_accuracy'][num_weak_signals - 1], w_acc_dict['baseline_test_accuracy'][0], w_acc_dict['gecriteria_test_accuracy']]
        # log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')
    

    # # calculate ge criteria
    # print("Running tests on ge criteria...")
    # model = ge_criterion_train(val_data.T, val_labels, weak_signal_probabilities, num_weak_signals)
    # ge_validation_accuracy = accuracy_score(val_labels, np.round(probability(val_data, model)))
    # ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
    # print("The accuracy of ge criteria on validation data is", ge_validation_accuracy)
    # print("The accuracy of ge criteria on test data is", ge_test_accuracy)
    # w_acc_dict['gecriteria_val_accuracy'] = ge_validation_accuracy
    # w_acc_dict['gecriteria_test_accuracy'] = ge_test_accuracy
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # return adversarial_acc_dicts, w_acc_dicts