import numpy as np

from data_readers import read_text_data
#from models import ALL, MultiALL, CLL
from utilities import set_up_constraint
# from data_utilities import load_fashion_mnist # Don't do *, error
from image_utilities import get_supervision_data
from load_image_data import *

# Import models for testing
from models import ALL, MultiALL
from LabelEstimators import LabelEstimator, CLL
from GEModel import GECriterion 


"""
    Plan:
        1. get all algorithms to work one data set

    
    Datasets:
        1. SST-2
        2. IMDB
        3. Cardio
        4. OBS
        5. Breast Cancer
       
    Algorithms:
        1. Binary only All
        2. Multi ALL
        3. CLL     
        4. Maybe Data Consitency??????
"""


def log_results(train_accuracy, test_accuracy):
    """ 
        prints out results from the experiment

        Parameters
        ----------
        :param ??????: 
        :type ???????: 

        Returns
        -------
        nothing
    """
    print('\n\nLogging results\n\n')


def run_experiments(dataset):
    """ 
        sets up and runs expeirments on various algorithm

        Parameters
        ----------
        :param dataset: contains training set, testing set, and weak signals 
                        of read in data
        :type dataset: dictionary of ndarrays

        Returns
        -------
        nothing
    """
    

    # retrieve data from object
    # data = data_obj.data
    # train_data, train_labels = data['train_data']
    # train_data = train_data.T
    # test_data = data['test_data'][0].T
    # test_labels = data['test_data'][1]

    # print(dataset.keys())
    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape

    
    
    
    # print("\n\ntesting variables:")
    # print("\n weak sig: ", weak_signals)
    # print("shape: ", weak_signals.shape)
    # print("type: ", type(weak_signals))

    # exit()
    # print("\n train_data: ", train_features)
    # print("shape: ", train_features.shape)
    # print("type: ", type(train_features))

    # print("\n train_labels: ", train_labels)
    # print("shape: ", train_labels.shape)
    # print("type: ", type(train_labels))

    # print("\n test_data: ", test_features)
    # print("shape: ", test_features.shape)
    # print("type: ", type(test_features))

    # print("\n test_labels: ", test_labels)
    # print("shape: ", test_labels.shape)
    # print("type: ", type(test_labels))



    # set up variables
    train_accuracy            = []
    test_accuracy             = []

    # set up error bounds.... different for every algorithm
    # binary_all_weak_errors = np.zeros((m, k)) + 0.3

    try:
        weak_errors = dataset['weak_errors']
    except:
        weak_errors = np.ones((m, k)) * 0.01
    cll_weak_errors = set_up_constraint(weak_signals, weak_errors)

    error_set = [weak_errors, weak_errors, cll_weak_errors]


    # set up algorithms
    experiment_names = ["Binary-Label ALL", "Multi-Label ALL", "CLL"]
    binary_all = ALL(max_iter=5000, log_name="Binary-Label ALL")
    multi_all = MultiALL()
    Constrained_Labeling = CLL(log_name="CLL")

    models = [binary_all, multi_all, Constrained_Labeling]
    # models = [multi_all, Constrained_Labeling]

    # loop for number of weak signals????

    # Loop through each algorithm
    for model_np, model in enumerate(models):
        print("\n\nWORKING  WITH:", experiment_names[model_np])

        # # skip 
        # if model_np == 2 or model_np == 0:
        if model_np == 0 or model_np==1:
            continue 


        model.fit(train_data, weak_signals, error_set[model_np])

        """Predict_proba"""
        train_probas = model.predict_proba(train_data)
        train_acc = model.get_accuracy(train_labels, train_probas)

        test_probas = model.predict_proba(test_data)
        test_acc = model.get_accuracy(test_labels, test_probas)

        print("\nresults using predict_proba:")
        print("    Train Accuracy is: ", train_acc)
        print("    Test Accuracy is: ", test_acc)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    log_results(train_accuracy, test_accuracy)




if __name__ == '__main__':

    print("\n\n        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("        | WELCOME TO OUR EXPIRIMENTS  |")
    print("        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Expiriments:
    # dataset_names = ['obs']
    # dataset_names = ['sst-2', 'imdb', 'obs']

    # text data
    # for name in dataset_names:

    #     print("\n\n\n# # # # # # # # # # # #")
    #     print("#  ", name, "experiment  #")
    #     print("# # # # # # # # # # # #")
    #     run_experiments(read_text_data('../datasets/' + name + '/'))

    # Image data
    run_experiments(load_image_data())
