import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime

from data_readers import read_text_data
from utilities import set_up_constraint
# from data_utilities import load_fashion_mnist # Don't do *, error
from image_utilities import get_supervision_data
from load_image_data import load_image_data
from log import Logger 

# Import models for testing
from models import ALL, MultiALL
from LabelEstimators import LabelEstimator, CLL
from GEModel import GECriterion 
from PIL import Image



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


def log_results(values, acc_logger, plot_path, title):
    """ 
        prints out results from the experiment

        Parameters
        ----------
        values: list of floats, size is 3 (same as current num algorithms)
            list of accuracies (between 0 and 1) to graph

        acc_logger: object of Logger class
            current logger object that will be used to write out to 
            tensor board

        plot_path: str 
            path to where matplotlib png will be stored

        title: str 
            name of current graph to be used as a tittle 

        Returns
        -------
        nothing
    """
    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    method_names = ['BinaryALL', 'MultiALL','CLL']

    # add labels on graph 
    for i, v in enumerate(values):
        ax.text(i - 0.25, v + 0.01, str(round(v, 5)), color='seagreen', fontweight='bold')
    # ax.bar(methods, values, color=['skyblue', 'saddlebrown', 'olivedrab', 'plum'])
    ax.bar(method_names, values, color=['skyblue', 'saddlebrown', 'olivedrab'])


    # set y demensions of plots
    min_value = min(values)
    max_value = max(values)
    plt.ylim([min_value - 0.1, max_value + 0.1])

    # Save plot, then load into tensorboard
    plt.savefig(plot_path + "/plot.png", format='png')
    with acc_logger.writer.as_default():
        image = tf.io.read_file(plot_path + "/plot.png")
        image = tf.image.decode_png(image, channels=4)
        summary_op = tf.summary.image(title, [image], step=0)
        acc_logger.writer.flush()


def run_experiments(dataset, set_name, date):
    """ 
        sets up and runs expeirments on various algorithm

        Parameters
        ----------
        dataset : dictionary of ndarrays
            contains training set, testing set, and weak signals 
            of read in data
        
        set_name : str
            current name of dataset for logging purposes 

        date : str
            current date and time in format Y_m_d-I:M:S_p

        Returns
        -------
        nothing
    """

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape

    # set up variables
    train_accuracy = []
    test_accuracy = []
    log_name = date + "/" + set_name


    # set up error bounds.... different for every algorithm
    try:
        weak_errors = dataset['weak_errors']
    except:
        weak_errors = np.ones((m, k)) * 0.01
    cll_weak_errors = set_up_constraint(weak_signals, weak_errors)

    error_set = [weak_errors, weak_errors, cll_weak_errors]


    # set up algorithms
    experiment_names = ["Binary-Label ALL", "Multi-Label ALL", "CLL"]
    binary_all = ALL(max_iter=10000, log_name=log_name+"/BinaryALL")
    multi_all = MultiALL()
    Constrained_Labeling = CLL(log_name=log_name+"/CLL")

    models = [binary_all, multi_all, Constrained_Labeling]

    # Loop through each algorithm
    for model_np, model in enumerate(models):
        print("\n\nWORKING WITH:", experiment_names[model_np])

        # # skip 
        if model_np == 2 or model_np == 0:
        # if model_np == 0 or model_np==1:
            continue 


        model.fit(train_data, weak_signals, error_set[model_np])

        """Predict_proba"""
        train_probas = model.predict_proba(train_data)
        # print(train_probas)
        # print("predict proba ", train_probas.shape)
        # print(train_labels.shape)
        train_acc = model.get_accuracy(train_labels, train_probas)

        test_probas = model.predict_proba(test_data)
        test_acc = model.get_accuracy(test_labels, test_probas)

        print("\nresults using predict_proba:")
        print("    Train Accuracy is: ", train_acc)
        print("    Test Accuracy is: ", test_acc)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)


    print('\n\nLogging results\n\n')
    acc_logger = Logger("logs/" + log_name + "/accuracies")
    plot_path =  "./logs/" + log_name
    log_results(train_accuracy, acc_logger, plot_path, 'Accuracy on training data')
    log_results(test_accuracy, acc_logger, plot_path, 'Accuracy on testing data')





if __name__ == '__main__':

    print("\n\n        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("        | WELCOME TO OUR EXPIRIMENTS  |")
    print("        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # text Expiriments:
    dataset_names = ['sst-2', 'imdb', 'obs', 'cardio', 'breast-cancer']
    # dataset_names = ['obs', 'cardio', 'breast-cancer']

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    for name in dataset_names:
        print("\n\n\n# # # # # # # # # # # #")
        print("#  ", name, "experiment  #")
        print("# # # # # # # # # # # #")
        run_experiments(read_text_data('../datasets/' + name + '/'), name, date)

    # # Image Expiriments
    # run_experiments(load_image_data())
