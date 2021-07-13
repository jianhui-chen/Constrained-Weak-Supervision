import numpy as np
import codecs, json
import random, gc, time, sys
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras import backend as K
#from setup_model import set_up_constraint, mlp_model, mlp_model, accuracy_score, writeToFile, prepare_mmce
from setup_model import set_up_constraint, mlp_model, accuracy_score, writeToFile, prepare_mmce
from image_utilities import get_supervision_data
from train_CLL import train_algorithm
from text_utilities import get_textsupervision_data
from mmce import MinimaxEntropy_crowd_model
from data_utilities import *

# from sklearn.metrics import f1_score, precision_score, recall_score
from train_stochgall import train_stochgall

def run_experiment(data_set, savename):
    """
    Runs experiment with the given dataset

    :param data_and_weakmodel: dictionary of weak signal model and data
    :type data_and_weakmodel: dict
    """

    # set all the variables
    constraint_keys = ["error"]
    loss = 'multilabel'
    batch_size = 32

    data = get_supervision_data(data_set, weak_signals='pseudolabel', true_bounds=False)
    weak_model = data['weak_model']
    weak_signal_probabilities = weak_model['weak_signals']
    active_signals = weak_model['active_mask']

    model_names = data['model_names']
    train_data, train_labels = data['train_data']
    test_data, test_labels = data['test_data']
    labeled_onehot_labels = data['labeled_labels']
    img_rows, img_cols = data['img_rows'], data['img_cols']
    channels = data['channels']

    # build up data_info for the algorithm
    data_info = dict()
    data_info['train_data'], data_info['train_labels'] = data['train_data']

    # # only for stoch-gall experiment
    # data_info['test_data'], data_info['test_labels'] = data['test_data']
    # data_info['img_rows'], data_info['img_cols'] = img_rows, img_cols
    # data_info['channels'] = channels
    # loss = 'multiclass'

    print("train_data", train_data.shape)
    print("test_data", test_data.shape)

    max_weak_signals = weak_signal_probabilities.shape[0]
    num_weak_signals = max_weak_signals
    random_train = []
    random_test = []
    random_label_accuracy = []

    # points = np.linspace(0,0.5,20)

    # for num_weak_signals in range(start, max_weak_signals + 1):
    baseline_label_accuracy = []
    baseline_train_accuracy = []
    baseline_test_accuracy = []

    mv_weak_labels = majority_vote_signal(weak_signal_probabilities, num_weak_signals)
    data_info['num_weak_signals'] = num_weak_signals

    # mmce_crowd_labels = np.round(weak_signal_probabilities)
    # mmce_crowd_labels[mmce_crowd_labels==0]= -1
    # mmce_crowd_labels, transformed_labels = prepare_mmce(mmce_crowd_labels, train_labels)
    # mmce_labels, mmce_error_rate = MinimaxEntropy_crowd_model(mmce_crowd_labels, transformed_labels)
    # mmce_labels -=1
    # model = mlp_model(img_rows, img_cols, channels, loss)
    # mmce_labels = tf.one_hot(mmce_labels, 10)
    # model.fit(train_data, mmce_labels, batch_size=batch_size, epochs=20, verbose=1)
    # test_predictions = model.predict(test_data)
    # test_accuracy = accuracy_score(test_labels, test_predictions)

    # mmce = {}
    # filename = 'results/mmce_results.json'
    # mmce['label_accuracy'] = 1 - mmce_error_rate
    # mmce['test_accuracy'] = test_accuracy
    # mmce['experiment'] = savename
    # print('MMCE test_accu: %f' %test_accuracy)
    # with open(filename, 'a') as file:
    #     json.dump(mmce, file, indent=4, separators=(',', ':'))
    # file.close()

    bounds_accuracy = []

    for i in range(3):
        new_constraint_set = set_up_constraint(weak_signal_probabilities[:num_weak_signals, :, :],
                                               weak_model['precision'][:num_weak_signals, :],
                                               weak_model['error_bounds'][:num_weak_signals, :])
        new_constraint_set['constraints'] = constraint_keys
        new_constraint_set['weak_signals'] = weak_signal_probabilities[:num_weak_signals, :, :] * active_signals[:num_weak_signals, :, :]
        new_constraint_set['num_weak_signals'] = num_weak_signals

        print("Running tests...")
        y = train_stochgall(data_info, new_constraint_set)
        #y = train_algorithm(new_constraint_set)

        results = {}
        label_accuracy = accuracy_score(train_labels, y)
        print(label_accuracy)
        print("Running constrained label learning...")
        model = mlp_model(img_rows, img_cols, channels)
        model.fit(train_data, np.round(y), batch_size=batch_size, epochs=20, verbose=1)

        # update cll results
        random_label_accuracy.append(label_accuracy)

        y_pred = model.predict(train_data)
        train_accuracy = accuracy_score(train_labels, y_pred)
        random_train.append(train_accuracy)

        # calculate test results
        y_pred = model.predict(test_data)
        test_accuracy = accuracy_score(test_labels, y_pred)
        random_test.append(test_accuracy)

        print('CLL train_acc: %f, test_accu: %f' %(train_accuracy, test_accuracy))

        K.clear_session()
        del model
        gc.collect()

        # calculate majority vote baseline
        print("Running tests on the baselines...")
        label_accuracy = accuracy_score(train_labels, mv_weak_labels)
        baseline_label_accuracy.append(label_accuracy)
        print("Accuracy of the baseline labels is ", label_accuracy)
        baseline = mlp_model(img_rows, img_cols, channels, loss)
        baseline.fit(train_data, mv_weak_labels, batch_size=batch_size, epochs=20, verbose=1)

        # calculate train results
        y_pred = baseline.predict(train_data)
        baseline_train_acc = accuracy_score(train_labels, y_pred)
        baseline_train_accuracy.append(baseline_train_acc)

        # calculate test results
        y_pred = baseline.predict(test_data)
        baseline_test_acc = accuracy_score(test_labels, y_pred)
        baseline_test_accuracy.append(baseline_test_acc)

        print("The accuracy of the baseline models on training data is", baseline_train_acc)
        print("The accuracy of the baseline models on test data is", baseline_test_acc)
        print("")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    adversarial_model = {}
    # adversarial_model['cll_label_accuracy'] = random_label_accuracy
    # adversarial_model['cll_train_accuracy'] = random_train
    # adversarial_model['cll_test_accuracy'] = random_test

    mv_baseline = {}
    mv_baseline['num_weak_signal'] = num_weak_signals



    #########################################################################################
    # # Comment if baseline not supervised, uncomment above
    # print("Running supervised experiment as baseline...")
    # baseline = mlp_model(img_rows, img_cols, channels, loss)
    # baseline.fit(train_data, train_labels, batch_size=batch_size, epochs=20, verbose=1)
    # label_accuracy = 1.0
    #########################################################################################

    output = {}
    adversarial_model['stats'] = [np.mean(random_label_accuracy), np.std(random_label_accuracy), np.mean(random_test), np.std(random_test)]
    mv_baseline['stats'] = [np.mean(baseline_label_accuracy),np.std(baseline_label_accuracy),np.mean(baseline_test_accuracy), np.std(baseline_test_accuracy)]
    output['Adversarial model'] = adversarial_model
    output['Weak model'] = mv_baseline

    print("Saving to file...")
    filename = 'results/new_results/'+savename+'_results.json'
    writeToFile(output, filename)

    filename = 'results/snorkel/'+savename+'/supervision_data.npy'
    output = {}
    output['train'] = train_data, train_labels
    output['test'] = test_data, test_labels
    output['num_weak_signals'] = num_weak_signals
    np.save(filename, output)


def run_text_experiment(data_set, savename):
    """
    Runs experiment with the given dataset

    :param data_and_weakmodel: dictionary of weak signal model and data
    :type data_and_weakmodel: dict
    """

    # set all the variables
    constraint_keys = ["error"]
    batch_size = 32

    data = get_textsupervision_data(data_set, supervision='manual', true_bounds=False)
    weak_model = data['weak_model']
    weak_signal_probabilities = weak_model['weak_signals']
    active_signals = weak_model['active_mask']
    print(active_signals)
    exit()

    model_names = data['model_names']
    train_data, train_labels = data['train_data']
    test_data, test_labels = data['test_data']
    has_labels, has_test_data = data['has_labels']

    # print(train_labels[train_labels==1].shape)
    # print(train_labels.shape)
    # print(test_labels[test_labels==1].shape)
    # print(test_labels.shape)
    # exit()

    max_weak_signals = weak_signal_probabilities.shape[0]
    num_weak_signals= max_weak_signals

    # build up data_info for the algorithm
    data_info = dict()
    data_info['train_data'], data_info['train_labels'] = data['train_data']
    data_info['test_data'], data_info['test_labels'] = data['test_data']
    print("train_data", train_data.shape)
    print("test_data", test_data.shape)
    print("No of weak signals", num_weak_signals)

    start = 1

    train_accuracies = []
    test_accuracies = []
    label_accuracies = []

    baseline_label_accuracies = []
    baseline_train_accuracies = []
    baseline_test_accuracies = []

    mv_weak_labels = majority_vote_signal(weak_signal_probabilities, num_weak_signals)
    n,k = mv_weak_labels.shape
    data_info['num_weak_signals'] = num_weak_signals

    # mmce_crowd_labels, transformed_labels = prepare_mmce(weak_signal_probabilities, train_labels)
    # mmce_labels, mmce_error_rate = MinimaxEntropy_crowd_model(mmce_crowd_labels, transformed_labels)
    # mmce_labels -=1
    # model = mlp_model(train_data.shape[1], k)
    # if k > 1:
    #     mmce_labels = tf.one_hot(mmce_labels, k)
    # model.fit(train_data, mmce_labels, batch_size=batch_size, epochs=20, verbose=1)
    # test_predictions = model.predict(test_data)
    # test_accuracy = accuracy_score(test_labels, test_predictions)

    # mmce = {}
    # filename = 'results/mmce_results.json'
    # mmce['label_accuracy'] = 1 - mmce_error_rate
    # mmce['test_accuracy'] = test_accuracy
    # mmce['experiment'] = savename
    # print('MMCE test_accu: %f' %test_accuracy)
    # with open(filename, 'a') as file:
    #     json.dump(mmce, file, indent=4, separators=(',', ':'))
    # file.close()

    for i in range(3):
        # print(weak_signal_probabilities.shape)
        # print(weak_model['precision'].shape)
        # print(weak_model['error_bounds'].shape)
        # print(weak_signal_probabilities[:num_weak_signals, :, :].shape)
        # exit()
        new_constraint_set = set_up_constraint(weak_signal_probabilities[:num_weak_signals, :, :],
                                               weak_model['precision'][:num_weak_signals, :],
                                               weak_model['error_bounds'][:num_weak_signals, :])
        new_constraint_set['constraints'] = constraint_keys
        new_constraint_set['weak_signals'] = weak_signal_probabilities[:num_weak_signals, :, :] * active_signals[:num_weak_signals, :, :]

        # print(new_constraint_set['weak_signals'].shape)
        # print(num_weak_signals)
        # exit()

        #Adding this in
        # new_constraint_set['loss'] = 'multiclass' # do i want multilabel
        new_constraint_set['loss'] = 'multilabel'
        new_constraint_set['num_weak_signals'] = num_weak_signals

        print("Running tests...")

        # print(train_data.shape)
        # print(test_data.shape)
        # print(k)
   

        results = train_stochgall(data_info, new_constraint_set)
        #y = train_algorithm(new_constraint_set)
        #label_accuracy = accuracy_score(train_labels, y)
        print("ALL label accuracy: %f" % results['label_accuracy'])
        print("mv label accuracy: %f" % accuracy_score(train_labels, mv_weak_labels))

        #PRINT MORE RESULTS

        # Below is stuff needed for CLL 
        """
        results = {}
        model = mlp_model(train_data.shape[1], k)
        model.fit(train_data, y, batch_size=batch_size, epochs=20, verbose=1)
        train_predictions = model.predict(train_data)
        test_predictions = model.predict(test_data)

        adversarial_model = {}

        if has_labels:
            label_accuracy = accuracy_score(train_labels, y)
            print("Running constrained label learning...")
            # update cll results
            label_accuracies.append(label_accuracy)

            # calculate train accuracy
            train_accuracy = accuracy_score(train_labels, train_predictions)
            train_accuracies.append(train_accuracy)

            # calculate test results
            test_accuracy = accuracy_score(test_labels, test_predictions)
            test_accuracies.append(test_accuracy)

            print('CLL train_acc: %f, test_accu: %f' %(train_accuracy, test_accuracy))

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            label_accuracies.append(label_accuracy)
        else:
            if has_test_data:
                 # calculate test results
                test_accuracy = accuracy_score(test_labels, test_predictions)

                print('RCL test_accu: %f' %test_accuracy)

                adversarial_model['rcl_test_accuracy'] = test_accuracy
                adversarial_model['rcl_precision'] = precision_score(test_labels, test_predictions)
                adversarial_model['rcl_recall'] = recall_score(test_labels, test_predictions)
                adversarial_model['rcl_fscore'] = f1_score(test_labels, test_predictions)
            else:
                adversarial_model['train_predictions'] = np.round(train_predictions.ravel()).tolist()
                adversarial_model['train_indexes'] = train_labels.tolist()
            # adversarial_model['test_predictions'] = test_predictions.tolist()
            # adversarial_model['test_indexes'] = test_labels.tolist()

        K.clear_session()
        del model
        gc.collect()
        """

        mv_baseline = {}
        mv_baseline['num_weak_signal'] = num_weak_signals

        # calculate majority vote baseline
        print("Running tests on the baselines...")
        baseline = mlp_model(train_data.shape[1], k)
        baseline.fit(train_data, mv_weak_labels, batch_size=batch_size, epochs=20, verbose=1)
        # baseline = LogisticRegression(solver="lbfgs", max_iter=1000)
        # baseline.fit(train_data, np.rint(mv_weak_labels.ravel()))

    #     #######################################################################################
    #     # Comment if baseline not supervised, uncomment above
    #     # print("Running supervised experiment as baseline...")
    #     # baseline = mlp_model(train_data.shape[1], k)
    #     # baseline.fit(train_data, train_labels, batch_size=batch_size, epochs=20, verbose=1)
    #     # mv_weak_labels = train_labels.copy()
    #     ########################################################################################

        # Baseline calculations: predict on the data
        train_predictions = baseline.predict(train_data)
        test_predictions = baseline.predict(test_data)

        if has_labels:
            label_accuracy = accuracy_score(train_labels, mv_weak_labels)
            print("Accuracy of the baseline labels is ", label_accuracy)
            baseline_train_acc = accuracy_score(train_labels, train_predictions)
            baseline_test_acc = accuracy_score(test_labels, test_predictions)

            baseline_label_accuracies.append(label_accuracy)
            baseline_train_accuracies.append(baseline_train_acc)
            baseline_test_accuracies.append(baseline_test_acc)

            mv_baseline['baseline_label_accuracy'] = baseline_label_accuracies
            mv_baseline['baseline_train_accuracy'] = baseline_train_accuracies
            mv_baseline['baseline_test_accuracy'] = baseline_test_accuracies

            print("The accuracy of the baseline models on training data is", baseline_train_acc)
            print("The accuracy of the baseline models on test data is", baseline_test_acc)
            print("")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        else:
            if has_test_data:
                 # calculate test results
                test_accuracy = accuracy_score(test_labels, test_predictions)

                print('MV test_accu: %f' %test_accuracy)

                mv_baseline['mv_test_accuracy'] = test_accuracy
                mv_baseline['mv_precision'] = precision_score(test_labels, test_predictions)
                mv_baseline['mv_recall'] = recall_score(test_labels, test_predictions)
                mv_baseline['mv_fscore'] = f1_score(test_labels, test_predictions)
            else:
                mv_baseline['train_predictions'] = np.round(train_predictions.ravel()).tolist()

    output = {}
    adversarial_model['stats'] = [np.mean(label_accuracies), np.std(label_accuracies), np.mean(test_accuracies), np.std(test_accuracies)]
    mv_baseline['stats'] = [np.mean(baseline_label_accuracies),np.std(baseline_label_accuracies), \
                        np.mean(baseline_test_accuracies), np.std(baseline_test_accuracies)]
    output['Adversarial model'] = adversarial_model
    output['Weak model'] = mv_baseline

    """
    print("Saving to file...")
    filename = 'results/new_results/'+savename+'_results.json'
    # filename = savename+'_results.json'
    writeToFile(output, filename)

    filename = 'results/snorkel/'+savename+'/supervision_data.npy'
    output = {}
    output['train'] = train_data, train_labels
    output['test'] = test_data, test_labels
    output['num_weak_signals'] = num_weak_signals
    np.save(filename, output)
    """


def run_snorkel_experiment(data_set, pathname, loss='multilabel'):

    path = 'results/snorkel/'+pathname
    data = np.load(path+'supervision_data.npy', allow_pickle=True)[()]
    # img_rows, img_cols = data_set['img_rows'], data_set['img_cols']
    # channels, num_classes = data_set['channels'], data_set['num_classes']
    train_data, train_labels = data['train']
    test_data, test_labels = data['test']
    num_weak_signals = data['num_weak_signals']

    output = {}
    #  Write snorkel matrix to a file
    snorkel_marginals = codecs.open(path+'label_matrix.json', 'r', encoding='utf-8').read()
    snorkel_labels = np.asarray(json.loads(snorkel_marginals)).T

    n,k = snorkel_labels.shape
    snorkel_labels = np.round(snorkel_labels) if k == 1 else snorkel_labels

    print("train_labels", train_labels.shape)
    print("snorkel_labels", snorkel_labels.shape)

    label_accuracy = accuracy_score(train_labels, snorkel_labels)
    print("Accuracy of snorkel train labels", label_accuracy)

    train_accuracies = []
    test_accuracies = []

    for i in range(3):
        # model = mlp_model(img_rows, img_cols, channels, loss)
        model = mlp_model(train_data.shape[1], k)
        model.fit(train_data, snorkel_labels, batch_size=32, epochs=20, verbose=1)

        y_pred = model.predict(train_data)
        train_accuracy = accuracy_score(train_labels, y_pred)
        train_accuracies.append(train_accuracy)

        y_pred = model.predict(test_data)
        test_accuracy = accuracy_score(test_labels, y_pred)
        test_accuracies.append(test_accuracy)

        K.clear_session()
        del model
        gc.collect()

        print("")
        print("The accuracy of the model on the train data is", train_accuracy)
        print("The accuracy of the model on the test data is", test_accuracy)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    output['num_weak_signals'] = num_weak_signals
    output['snorkel_label_accuracy'] = label_accuracy
    output['Snorkel_train_accuracy'] = train_accuracies
    output['Snorkel_test_accuracy'] = test_accuracies
    output['stats'] = [np.mean(test_accuracies), np.std(test_accuracies)]

    writeToFile(output, path+'snorkel_results.json')


def get_results(snorkel=False):

    # experiments for datasets used in ALL
    # run_text_experiment(read_text_data('../datasets/breast-cancer/'),'all_datasets/breastcancer')
    # run_text_experiment(read_text_data('../datasets/obs-network/'),'all_datasets/obs')
    # run_text_experiment(read_text_data('../datasets/cardiotocography/'),'all_datasets/cardio')
    # run_text_experiment(read_text_data('../datasets/clave-direction/'),'all_datasets/clave')
    # run_text_experiment(read_text_data('../datasets/credit-card/'),'all_datasets/creditcard')
    # run_text_experiment(read_text_data('../datasets/statlog-landsite-satellite/'),'all_datasets/statlog')
    # run_text_experiment(read_text_data('../datasets/phishing/'),'all_datasets/phishing')
    # run_text_experiment(read_text_data('../datasets/winequality/'),'all_datasets/wine')

    # experiments for AV news data
    # data = 'world_first_l3/'
    # run_text_experiment(read_text_data('../datasets/av-news/'+data), 'results/av_news/'+data+'crash')

    if snorkel:
        # run_snorkel_experiment(load_svhn(),'svhn/')
        # run_snorkel_experiment(load_fashion_mnist(),'fmnist/')
        # run_snorkel_experiment('../datasets/imbd/','imbd/')
        # run_snorkel_experiment('../datasets/trec-6/','trec-6/')
        # run_snorkel_experiment('../datasets/sst-2/','sst-2/')
        # run_snorkel_experiment(synthetic_experiment(),'synthetic/')
        # run_snorkel_experiment('../datasets/yelp/','yelp/')
        pass
    else:
        run_text_experiment(read_text_data('../datasets/imdb/'),'imbd')
        #run_text_experiment(read_text_data('../datasets/yelp/'),'yelp')
        run_text_experiment(read_text_data('../datasets/sst-2/'), 'sst-2')
        #run_text_experiment(read_text_data('../datasets/trec-6/'),'trec-6')
        #run_experiment(load_svhn(),'svhn')
        #run_experiment(load_fashion_mnist(),'fmnist')

        # run_text_experiment(synthetic_experiment(),'synthetic-independent')
        pass

get_results(snorkel=False)
