from setup_model import *
from sklearn.linear_model import LogisticRegression
import numpy as np
import codecs, json, time, sys
from sklearn import preprocessing
import random, gc, time, sys
from tensorflow.python.keras import backend as K
from sklearn.base import clone
from data_utilities import *
from sklearn.model_selection import train_test_split
from setup_model import build_constraints, mlp_model


def get_textsupervision_data(dataset, supervision='manual', true_bounds=False):

    data_set = dataset.copy()
    train_data = data_set['data_features']
    weak_signals = data_set['weak_signals']
    train_labels = data_set['labels']
    test_data = data_set['test_features']
    test_labels = data_set['test_labels']
    has_test_data = data_set['has_test_data']
    has_labels = data_set['has_labels']

    num_weak_signals = weak_signals.shape[0]
    num_classes = weak_signals.shape[2]

    if not has_test_data:
        # get indices for test data
        n = train_data.shape[0]
        indexes = np.arange(n)
        np.random.seed(2000)
        test_indexes = np.random.choice(n, int(n*0.2), replace=False)
        weak_signals = np.delete(weak_signals, test_indexes, axis=1)

        weak_signals_mask = weak_signals >=0

        if has_labels:
            test_labels = train_labels[test_indexes]
            test_data = train_data[test_indexes]
            train_indexes = np.delete(indexes, test_indexes)
            train_labels = train_labels[train_indexes]
            train_data = train_data[train_indexes]

            # get indexes of labels to train on
            num_unlabeled, indexes = get_labeled_index(train_labels, num_classes)
            np.random.seed(2000)
            np.random.shuffle(indexes)
            labeled_data, labeled_labels = train_data[indexes], train_labels[indexes]
        else:
            test_labels = test_indexes.copy()
            train_labels = np.delete(indexes, test_indexes)
            test_data = train_data[test_indexes]
            train_data = train_data[train_labels]

    else:
        if has_labels:
            # get indexes of labels to train on
            num_unlabeled, indexes = get_labeled_index(train_labels, num_classes)
            np.random.seed(2000)
            np.random.shuffle(indexes)
            labeled_data, labeled_labels = train_data[indexes], train_labels[indexes]
        else:
            train_labels = np.random.choice([0,1],train_data.shape[0])

    # calculate true bounds for the human signals
    true_error_rates, true_precisions = get_validation_bounds(train_labels, weak_signals)

    # for debbugging to check performance on supervised learning
    # model = mlp_model(300,1)
    # model.fit(train_data, train_labels, batch_size=32, epochs=30, verbose=1)
    # predicted_labels = model.predict(train_data)
    # accuracy = accuracy_score(train_labels, predicted_labels)
    # print("The accuracy of the train labels is: ", accuracy)
    # predicted_labels = model.predict(test_data)
    # accuracy = accuracy_score(test_labels, predicted_labels)
    # print("The accuracy of the test labels is: ", accuracy)

    # # model = LogisticRegression(solver="lbfgs", max_iter=1000)
    # # model.fit(train_data, np.argmax(train_labels,axis=1))
    # # predicted_labels = model.predict_proba(train_data)
    # # accuracy = accuracy_score(train_labels, predicted_labels)
    # # print("The accuracy of the train labels is: ", accuracy)
    # # predicted_labels = model.predict_proba(test_data)
    # # accuracy = accuracy_score(test_labels, predicted_labels)
    # # print("The accuracy of the test labels is: ", accuracy)
    # sys.exit(0)

    # define variables
    weak_data = dict()
    model_names = []
    weak_probabilities = []
    error_bounds = []
    precisions = []

    data = ''

    if has_labels:
        if supervision != 'manual':

            for i, model in enumerate(getNewModel()):
                results = dict()
                modelname = str(model).split('(')[0]
                data_set['modelname'] = modelname

                try:
                    data = np.load('results/new_results/'+modelname+'.npy', allow_pickle=True)[()]
                    print("Succesfully loaded data...")
                    predicted_labels = data['predicted_labels']
                    accuracy = data['accuracy']
                except:
                    filename = 'results/new_results/'+modelname+'.npy'
                    output = {}

                    if 'model' in modelname:
                        model = mlp_model(train_data.shape[1], train_labels.shape[1])
                        model.fit(labeled_data,
                                  labeled_labels,
                                  batch_size=32,
                                  epochs=100,
                                  verbose=1)
                        predicted_labels = model.predict(train_data)
                        accuracy = accuracy_score(train_labels, predicted_labels)
                    else:
                        model.fit(labeled_data, np.argmax(labeled_labels,axis=1))
                        predicted_labels = model.predict_proba(train_data)
                        accuracy = accuracy_score(train_labels, predicted_labels)

                    output['predicted_labels'] = predicted_labels
                    output['accuracy'] = accuracy
                    np.save(filename, output)

                weak_probabilities.append(predicted_labels)
                print("%s has accuracy of %f" % (modelname, accuracy))

                # true bounds for the pseudolabels
                error_rates, precision = calculate_bounds(train_labels, predicted_labels)
                error_bounds.append(error_rates)
                precisions.append(precision)
                model_names.append(modelname)

            # concatenate pseudolabels and human weak signals
            weak_probabilities = np.concatenate((weak_probabilities, weak_signals))
            precisions = np.concatenate((precisions, true_precisions))
            error_bounds = np.concatenate((error_bounds, true_error_rates))

        else:
            weak_probabilities = weak_signals
            precisions = np.asarray(true_precisions)
            error_bounds = np.asarray(true_error_rates)

    else:
        m,n,k = weak_signals.shape
        weak_probabilities = weak_signals

    human_model_names = ['human_labels_' + str(i + 1) for i in range(num_weak_signals)]
    model_names.extend(human_model_names)

    num_weak_signals = len(model_names)

    # dump snorkel signals to file
    # assert num_weak_signals == weak_probabilities.shape[0]
    # convert_snorkel_signals(weak_probabilities, num_weak_signals)

    # new modification, comment to use true bounds
    if not true_bounds:
        m,n,k = weak_probabilities.shape
        err = 0.01
        precisions = np.ones((m,k)) * 0.6
        error_bounds = np.ones((m,k)) * err
        # error_bounds, precisions = get_estimated_bounds(weak_probabilities)

    # coverage = np.sum(weak_probabilities != -1, axis=1)*(1/weak_probabilities.shape[1])
    # coverage = np.sum(weak_probabilities >0.5, axis=1)*(1/weak_probabilities.shape[1])
    # print("coverage:", coverage)
    # # print("precisions: ", np.asarray(precisions))
    # errors, precisions = get_estimated_bounds(weak_probabilities)
    # print("True error: ", np.asarray(error_bounds))
    # print("estimated error: ", np.asarray(errors * np.exp(coverage)))
    # print("weak_probabilities", weak_probabilities.shape)
    # sys.exit(0)

    # calculate snorkel model label accuracy
    # m,n,k = weak_probabilities.shape
    # weak_probabilities = np.rint(weak_probabilities)
    # weak_signals = weak_probabilities.transpose((1,0,2))
    # weak_signals = weak_signals.reshape(n,m*k)

    # Ls = np.ones(weak_signals.shape)
    # Ls[weak_signals==-1] =0
    # Ls[weak_signals==0] =-1
    # est_accs = np.ravel(1 - error_bounds)
    # # print("est accuracy", est_accs)
    # lo = np.log( est_accs / (1.0 - est_accs))
    # Yp = 1 / (1 + np.exp(-np.ravel(Ls.dot(lo))))
    # label_accuracy = accuracy_score(train_labels, Yp)
    # print("Snorkel label accuracy", label_accuracy)
    # sys.exit(0)

    # model = LogisticRegression(solver="lbfgs", max_iter=1000)
    # model.fit(train_data, np.round(Yp.ravel()))
    # # model = mlp_model(train_data.shape[1])
    # # model.fit(train_data, Yp.ravel(), batch_size=32, epochs=5, verbose=1)
    # train_predictions = model.predict(train_data)
    # test_predictions = model.predict(test_data)

    # # calculate train accuracy
    # train_accuracy = accuracy_score(train_labels, train_predictions)
    # # calculate test results
    # test_accuracy = accuracy_score(test_labels, test_predictions)

    # print('Snorkel train_acc: %f, test_accu: %f' %(train_accuracy, test_accuracy))
    # sys.exit(0)

    weak_signals_mask = weak_probabilities >=0
    weak_data['weak_signals'] = weak_probabilities
    weak_data['active_mask'] = weak_signals_mask
    weak_data['precision'] = np.asarray(precisions)
    weak_data['error_bounds'] = np.asarray(error_bounds)

    data = dict()
    data['model_names'] = model_names
    data['weak_model'] = weak_data
    data['train_data'] = train_data, train_labels
    data['test_data'] = test_data, test_labels
    data['has_labels'] = has_labels, has_test_data

    return data

# second version of rank experiment
    # labels = []
    # majority_labels = []
    # m,n,k = weak_signals.shape
    # new_constraint_set = {}
    # weak_signals = weak_probabilities.copy()
    # from train_algorithm import run_constraints
    # for i in range(m-1):

    #     true_error_rates, true_precisions = get_validation_bounds(train_labels, weak_signals)
    #     new_constraint_set = set_up_constraint(weak_signals, np.asarray(true_precisions), np.asarray(true_error_rates))['error']

    #     assert m == 100
    #     new_constraint_set['constraints'] = ['error']
    #     new_constraint_set['train_labels'] = train_labels
    #     new_constraint_set['weak_signals'] = weak_signals
    #     new_constraint_set['active_mask'] = weak_signals_mask
    #     new_constraint_set['num_weak_signals'] = m
    #     new_constraint_set['optim'] = 'min'

    #     y = run_constraints(np.random.rand(weak_probabilities.shape[1],1), 0.1, new_constraint_set, optim='min')
    #     label_accuracy = accuracy_score(train_labels, y)
    #     print('Learned label accuracy: %f: ' %label_accuracy)
    #     labels.append(label_accuracy)
    #     majority_vote = majority_vote_signal(np.round(weak_signals), m)
    #     majority_labels.append(accuracy_score(train_labels, majority_vote))

    #     weak_signals[i+1,:,:] = np.random.rand(n,1)

    # output = {}
    # output['Adversarial model'] = labels
    # output['majority_vote'] = majority_labels

    # print("Saving to file...")
    # filename = 'results/new_results/rank_results.json'
    # writeToFile(output, filename)
    # sys.exit(0)
