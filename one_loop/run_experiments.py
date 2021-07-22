import numpy as np

from data_readers import read_text_data
from models import ALL, MultiALL, CLL
from utilities import set_up_constraint



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

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape

    # set up variables
    train_accuracy            = []
    test_accuracy             = []

    # set up error bounds.... different for every algorithm
    # binary_all_weak_errors = np.zeros((m, k)) + 0.3
    weak_errors = np.ones((m, k)) * 0.01
    cll_weak_errors = set_up_constraint(weak_signals, weak_errors)

    error_set = [weak_errors, weak_errors, cll_weak_errors]


    # set up algorithms
    experiment_names = ["Binary-Label ALL", "Multi-Label ALL", "CLL"]
    binary_all = ALL(max_iter=5000, log_name="Binary-Label ALL")
    multi_all = MultiALL()
    Constrained_Labeling = CLL(log_name="CLL")

    models = [binary_all, multi_all, Constrained_Labeling]


    # loop for number of weak signals????

    # Loop through each algorithm
    for model_np, model in enumerate(models):
        print("\n\nWorking with", experiment_names[model_np])

        # # skip 
        # if model_np == 2 or model_np == 0:
        #     continue 

        model.fit(train_data, weak_signals, error_set[model_np])


        """Predict_proba"""
        train_probas = model.predict_proba(train_data)
        train_acc = model.get_accuracy(train_labels, train_probas)

        test_probas = model.predict_proba(test_data)
        test_acc = model.get_accuracy(test_labels, test_probas)

        print("\n\nRESULTS USING predict_proba:")
        print("    Train Accuracy is: ", train_acc)
        print("    Test Accuracy is: ", test_acc)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    log_results(train_accuracy, test_accuracy)






# # Expiriments:
# # ------------

print("\n\n# # # # # # # # # # # #")
print("#  sst-2  experiment  #")
print("# # # # # # # # # # # #\n")
run_experiments(read_text_data('../datasets/sst-2/'))

# print("\n\n# # # # # # # # # # #")
# print("#  imdb experiment  #")
# print("# # # # # # # # # # #\n")
# run_experiments(read_text_data('../datasets/imdb/'))
