import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import getAccuracy, runBaselineTests
from sklearn.metrics import accuracy_score
from log import Logger, log_accuracy
import json
import math

from models import ALL, LabelEstimator, GECriterion



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



def new_run_experiment(data_obj, multiple_weak_signals, constant_bound=False):
    """
    Runs experiment with the given dataset
    :param data: dictionary of validation and test data
    :type data: dict
    :param multiple_weak_signals: 
    :type: 
    """

    adversarial_acc_dicts = []
    w_acc_dicts = []

    experiment_names = ["ALL w/ constant bounds", "ALL w/ computed bounds", "Avergaing Baseline", "GE Crit"]

    data = data_obj.data

    dev_data = data['dev_data'][0].T
    dev_labels = data['dev_data'][1]
    train_data, train_labels = data['train_data']
    train_data = train_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = dev_data.shape
    
    """
    print(multiple_weak_signals[2]['probabilities'].shape)
    print(multiple_weak_signals[2]['probabilities'].shape[0])
    print(multiple_weak_signals[0]['probabilities'].shape[0])
    print(multiple_weak_signals[2]['probabilities'][0])
    exit()
    """

    for num_loops, weak_signals in enumerate(multiple_weak_signals, 1): #begins from 1

        """
        curr_expirment   = math.floor((num_loops - 1) / 3 )
        num_weak_signals = (num_loops - 1) % 3 + 1
        print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Running: " + experiment_names[curr_expirment] + " with " + str(num_weak_signals) + " weak signals...")
        
        # Set up logger and variables
        logger                    = Logger("logs/standard/" + data_obj.n + "/" + experiment_names[curr_expirment] + " with " + str(num_weak_signals) + " weak signals")
        """
        
        weak_signal_ub            = weak_signals['error_bounds']
        weak_signal_probabilities = weak_signals['probabilities']
    
        train_accuracy            = []
        test_accuracy             = []

   
        all_model_const = ALL()
        all_model       = ALL()
        ave_model  = LabelEstimator()
        ge_model = GECriterion()

        models = [all_model_const, all_model, ave_model, ge_model]


        for model_np, model in enumerate(models):
            print("Running: " + experiment_names[model_np] + " with " + str(num_loops) + " weak signals...")
            if model_np == 0:
                model = model.fit(train_data, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3)
            elif model_np == 3:
                model = model.fit(train_data, weak_signal_probabilities, weak_signal_ub, train_labels)
            else:
                model = model.fit(train_data, weak_signal_probabilities, weak_signal_ub)

            """
            print(model.weights)
            exit()
            """
            #print(train_data.shape)
            train_probas = model.predict_proba(train_data)
           
       
            test_probas = model.predict_proba(test_data)
          
            train_acc = model.get_accuracy(train_labels, train_probas)
       
            test_acc = model.get_accuracy(test_labels, test_probas)

            print("The accuracy on the train data is", train_acc)
            print("The accuracy on the test data is", test_acc)

            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

        """
        logger = Logger("logs/All/ Accuracies with " + str(num_loops) + " Weak signal")
        log_accuracy(logger, train_accuracy, 'Accuracy on Validation Data')
        log_accuracy(logger, test_accuracy, 'Accuracy on Testing Data')
        """
        


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


  