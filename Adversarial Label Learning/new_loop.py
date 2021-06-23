import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import getAccuracy, runBaselineTests
from sklearn.metrics import accuracy_score
from log import Logger, log_accuracy
import json
import math

from models import ALL, Baseline



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
# #         Example: w_acc_dict['baseline_train_accuracy'] = b_train_accuracy
# #
# #     4. Probability function
# #         Example: train_accuracy = getModelAccuracy(learned_probabilities, train_labels)
# #
# #     5. Name of currnent experiment
# #     6. Num weak signals (if anyt)
# #         Example:  experiment_names = ["ALL w/ constant bounds", "ALL w/ computed bounds", "Avergaing Baseline"]



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

    data = data_obj.data

    dev_data = data['dev_data'][0].T
    dev_labels = data['dev_data'][1]
    train_data, train_labels = data['train_data']
    train_data = train_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = dev_data.shape
    


    for num_loops, w_data_dict in enumerate(w_data_dicts, 1): #begins from 1

        """
        curr_expirment   = math.floor((num_loops - 1) / 3 )
        num_weak_signals = (num_loops - 1) % 3 + 1
        print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Running: " + experiment_names[curr_expirment] + " with " + str(num_weak_signals) + " weak signals...")
        
        # Set up logger and variables
        logger                    = Logger("logs/standard/" + data_obj.n + "/" + experiment_names[curr_expirment] + " with " + str(num_weak_signals) + " weak signals")
        """
        
        weak_signal_ub            = w_data_dict['error_bounds']
        weak_signal_probabilities = w_data_dict['probabilities']
        # weights                   = np.zeros(num_features)
        train_accuracy            = []
        test_accuracy             = []


        all_model_const = ALL(weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3, log_name="constant")
        all_model       = ALL(weak_signal_probabilities, weak_signal_ub, log_name="nonconstant")
        baseline_model  = Baseline(weak_signal_probabilities, weak_signal_ub)

        models = [all_model_const, all_model, baseline_model]

        for model_np, model in enumerate(models):
            print("Running: " + experiment_names[model_np] + " with " + str(num_loops) + " weak signals...")
            model = model.fit(train_data)

            train_probas = model.predict_proba(train_data)
            test_probas = model.predict_proba(test_data)

            train_acc = model.get_accuracy(train_labels, train_probas)
            test_acc = model.get_accuracy(test_labels, test_probas)

            print("The accuracy on the train data is", train_acc)
            print("The accuracy on the test data is", test_acc)

            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

        
        logger = Logger("logs/All/ Accuracies with " + str(num_loops) + " Weak signal")
        log_accuracy(logger, train_accuracy, 'Accuracy on Validation Data')
        log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')

        


        """
        # Train the data if needed
        if num_loops < 4:
            optimized_weights, y = train_all(train_data, weights, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3, logger, max_iter=10000)
        elif num_loops >= 4 and num_loops < 7:
            optimized_weights, y = train_all(train_data, weights, weak_signal_probabilities, weak_signal_ub, logger, max_iter=10000)
        else:
            baslines = runBaselineTests(train_data, weak_signal_probabilities) #remove the transpose to enable it run
        


        # calculate results
        train_accuracy = getAccuracy(train_data, train_labels, baslines, optimized_weights)
        test_accuracy       = getAccuracy(test_data, test_labels, baslines, optimized_weights)
        print("The accuracy on the train data is", train_accuracy)
        print("The accuracy on the test data is", test_accuracy)
        """


        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")


  
        
        # # calculate weak signal results... IDK WHAT TO DO WITH THESE #####
        # weak_train_accuracy = w_data_dict['train_accuracy']
        # weak_test_accuracy = w_data_dict['test_accuracy']

        # ONLY NEED ONE OF THESE 
        # adversarial_acc_dicts.append(adversarial_acc_dict)
        # w_acc_dicts.append(w_acc_dict)
    

    # calculate ge criteria
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Running tests on ge criteria...")
    num_weak_signals = 3
    model = ge_criterion_train(train_data.T, train_labels, weak_signal_probabilities, num_weak_signals)
    ge_train_accuracy = accuracy_score(train_labels, np.round(probability(train_data, model)))
    ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
    print("The accuracy of ge criteria on validation data is", ge_train_accuracy)
    print("The accuracy of ge criteria on test data is", ge_test_accuracy)
    # w_acc_dict['gecriteria_train_accuracy'] = ge_train_accuracy
    # w_acc_dict['gecriteria_test_accuracy'] = ge_test_accuracy
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # return adversarial_acc_dicts, w_acc_dicts





# def new_run_experiment(data_obj, w_data_dicts, constant_bound=False):
#     """
#     Runs experiment with the given dataset
#     :param data: dictionary of validation and test data
#     :type data: dict
#     :param w_data_dicts: 
#     :type: 
#     """

#     adversarial_acc_dicts = []
#     w_acc_dicts = []

#     data = data_obj.data

#     for num_weak_signals, w_data_dict in enumerate(w_data_dicts, 1): #begins from 1
#         # initializes logger
#         logger = Logger("logs/standard/" + data_obj.n + "/" + str(num_weak_signals))


#         dev_data = data['dev_data'][0].T
#         dev_labels = data['dev_data'][1]
#         train_data, train_labels = data['train_data']
#         train_data = train_data.T
#         test_data = data['test_data'][0].T
#         test_labels = data['test_data'][1]

#         num_features, num_data_points = dev_data.shape

#         weak_signal_ub = w_data_dict['error_bounds']
#         # weak_signal_ub = np.ones(w_data_dict['error_bounds'].shape) * 0.3
        
#         # Following line doesn't seem to be used anywhere?
#         #models = w_data_dict['models']
#         weak_signal_probabilities = w_data_dict['probabilities']

#         weights = np.zeros(num_features)

#         print("Running tests...")
#         if constant_bound:
#             optimized_weights, y = train_all(train_data, weights, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3, logger, max_iter=10000)
#         else:
#             optimized_weights, y = train_all(train_data, weights, weak_signal_probabilities, weak_signal_ub, logger, max_iter=10000)

#         # calculate validation results
#         learned_probabilities = probability(train_data, optimized_weights)
#         train_accuracy = getModelAccuracy(learned_probabilities, train_labels)

#         # calculate test results
#         learned_probabilities = probability(test_data, optimized_weights)
#         test_accuracy = getModelAccuracy(learned_probabilities, test_labels)

#         # calculate weak signal results
#         # weak_train_accuracy = w_data_dict['train_accuracy']
#         # weak_test_accuracy = w_data_dict['test_accuracy']

#         adversarial_acc_dict = {}
#         adversarial_acc_dict['train_accuracy'] = train_accuracy
#         adversarial_acc_dict['test_accuracy'] = test_accuracy

#         w_acc_dict = {}
#         w_acc_dict['num_weak_signals'] = num_weak_signals
#         # w_acc_dict['train_accuracy'] = weak_train_accuracy
#         # w_acc_dict['test_accuracy'] = weak_test_accuracy

#         print("")
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#         print("We trained %d learnable classifiers with %d weak signals" %(1, num_weak_signals))
#         print("The accuracy of learned model on the validatiion data is", train_accuracy)
#         # print("The accuracy of weak signal(s) on the validation data is", weak_train_accuracy)
#         print("The accuracy of the model on the test data is", test_accuracy)
#         # print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
#         print("")
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#         # calculate baseline
#         print("Running tests on the baselines...")
#         baselines = runBaselineTests(train_data, weak_signal_probabilities) #remove the transpose to enable it run
#         b_train_accuracy = getWeakSignalAccuracy(train_data, train_labels, baselines)
#         b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
#         print("The accuracy of the baseline models on test data is", b_test_accuracy)
#         print("")
#         w_acc_dict['baseline_train_accuracy'] = b_train_accuracy
#         w_acc_dict['baseline_test_accuracy'] = b_test_accuracy
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#         # calculate ge criteria
#         print("Running tests on ge criteria...")
#         model = ge_criterion_train(train_data.T, train_labels, weak_signal_probabilities, num_weak_signals)
#         ge_train_accuracy = accuracy_score(train_labels, np.round(probability(train_data, model)))
#         ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
#         print("The accuracy of ge criteria on validation data is", ge_train_accuracy)
#         print("The accuracy of ge criteria on test data is", ge_test_accuracy)
#         w_acc_dict['gecriteria_train_accuracy'] = ge_train_accuracy
#         w_acc_dict['gecriteria_test_accuracy'] = ge_test_accuracy
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#         adversarial_acc_dicts.append(adversarial_acc_dict)
#         w_acc_dicts.append(w_acc_dict)

#         # train_accuracy = [adversarial_acc_dict['train_accuracy'], w_acc_dict['train_accuracy'][num_weak_signals - 1], w_acc_dict['baseline_train_accuracy'][0], w_acc_dict['gecriteria_train_accuracy']]
#         # log_accuracy(logger, train_accuracy, 'Accuracy on Validation Data')

#         # test_accuracy = [adversarial_acc_dict['test_accuracy'], w_acc_dict['test_accuracy'][num_weak_signals - 1], w_acc_dict['baseline_test_accuracy'][0], w_acc_dict['gecriteria_test_accuracy']]
#         # log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')

#     # Old code
#     # log_accuracy(data_obj, 3, adversarial_acc_dicts, w_acc_dicts)

#     return adversarial_acc_dicts, w_acc_dicts

