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

    # # Variables we need for each loop/ diff experiment
    # #     1. Overall dictionary to add it to
    # #         Example: adversarial_acc_dicts.append(adversarial_acc_dict)
    # #
    # #     2. Overall dictionary to add it to
    # #     3. where to add it to for that dictionary
    # #         Example: w_acc_dict['baseline_val_accuracy'] = b_validation_accuracy
    # #
    # #     4. Probability function
    # #         Example: validation_accuracy = getModelAccuracy(learned_probabilities, val_labels)
    # #
    # #     5. Name of currnent experiment
    # #     6. Num weak signals (if anyt)
    # #         Example:  experiment_names = ["ALL w/ constant bounds", "ALL w/ computed bounds", "Avergaing Baseline"]










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


        print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Running..." + experiment_names[curr_expirment] + " with " + str(num_weak_signals) + " weak signals")


        # calculate results
        # learned_probabilities = probability(val_data, optimized_weights)
        # validation_accuracy = getModelAccuracy(learned_probabilities, val_labels)
        validation_accuracy = 5
        test_accuracy = 5
        print("The accuracy on the validatiion data is", validation_accuracy)
        print("The accuracy on the test data is", test_accuracy)

        # Save Results
        # w_acc_dict['baseline_val_accuracy'] = b_validation_accuracy
        # w_acc_dict['baseline_test_accuracy'] = b_test_accuracy


        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")


        # val_accuracy = [adversarial_acc_dict['validation_accuracy'], w_acc_dict['validation_accuracy'][num_weak_signals - 1], w_acc_dict['baseline_val_accuracy'][0], w_acc_dict['gecriteria_val_accuracy']]
        # log_accuracy(logger, val_accuracy, 'Accuracy on Validation Data')

        # test_accuracy = [adversarial_acc_dict['test_accuracy'], w_acc_dict['test_accuracy'][num_weak_signals - 1], w_acc_dict['baseline_test_accuracy'][0], w_acc_dict['gecriteria_test_accuracy']]
        # log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')


  
        
        # # calculate weak signal results... IDK WHAT TO DO WITH THESE #####
        # weak_val_accuracy = w_data_dict['validation_accuracy']
        # weak_test_accuracy = w_data_dict['test_accuracy']

        # ONLY NEED ONE OF THESE 
        # adversarial_acc_dicts.append(adversarial_acc_dict)
        # w_acc_dicts.append(w_acc_dict)
    

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