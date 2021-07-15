import numpy as np

from data_readers import read_text_data
from models import ALL, MultiALL, CLL



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


def print_results():
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
    print('\n\nThese are the results\n\n')


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

    # set up error bounds
    # weak_errors = np.ones((m, k)) * 0.01
    weak_errors = np.zeros((m, k)) + 0.3


    # set up algorithms
    experiment_names = ["Binary-Lavel ALL", "Multi-Label ALL", "CLL"]
    binary_all = ALL()
    multi_all = MultiALL()
    Constrained_Labeling = CLL()
    models = [binary_all, multi_all, Constrained_Labeling]


    # loop for number of weak signals

    # Loop through each algorithm
    for model_np, model in enumerate(models):
        print("\nWorking with", experiment_names[model_np])

        # debugging
        # print("\n\nvairbales: ")
        # print("\ntrain_data: ", train_data)
        # print("train_data shape: ", train_data.shape)
        # print("train_data type: ", type(train_data),"\n\n")
        # print("\nweak_signals: ", weak_signals)
        # print("weak_signals shape: ", weak_signals.shape)
        # print("weak_signals type: ", type(weak_signals),"\n\n")
        # exit()

        model.fit(train_data, weak_signals, weak_errors)
        break

    # print results
    print_results()






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
