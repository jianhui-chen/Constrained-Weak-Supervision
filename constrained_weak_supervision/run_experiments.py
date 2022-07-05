import numpy as np
import tensorflow as tf

from datetime import datetime
from utilities import *

# Import models
from ALL import ALL
# from MultiALL import MultiALL
#from CLL import CLL
#from DCWS import DataConsistency


"""
    Binary Datasets:
        1. SST-2
        2. IMDB
        3. Cardio
        4. Phishing
        5. Breast Cancer
        6. Yelp
    
    Multi-Class Datasets:
        1. Fashion Mnist
        2. SVHN

    Algorithms:
        1. ALL (Binary datasets only)
        2. Multi-ALL
        3. CLL     
        4. DCWS

    Weak Signals:
        -1 for abstain, 1 for positive labels and 0 for negative labels
        Supports 2D weak signals for both binary and multi-class data: num_examples vs num_signals
            For multi-class the signals are given as class labels [0, 1, 2, 3]
            -1 is abstain
            See synthetic experiment

        Supports One-vs-rest weak signals: num_signals vs num_examples vs num_classes
            Weak signals are 3D array
            See run experiments

"""

def synthetic_experiment():
    # Test with default 2D weak signals for all methods
    # Test with ovr signals
    pass


def run_experiments(dataset):
    """ 
        sets up and runs experiments on various algorithms

        Parameters
        ----------
        dataset : dictionary of ndarrays
            contains training set, testing set, and weak signals 
            of read in data
        
        Returns
        -------
        nothing
    """

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, num_classes = weak_signals.shape

    #####################################################################
    #### ALL
    
    if num_classes <= 2:
        all_model = ALL()
        all_model.fit(train_data, weak_signals)
        predicted_labels = all_model.predict_proba(train_data)
        predicted_classes = all_model.predict(train_data)

        assert all_model.get_score(np.round(predicted_labels), predicted_classes, metric='accuracy') == 1.0

        print(f"The train accuracy of ALL is: {all_model.get_score(train_labels, predicted_labels, metric='accuracy')}")
        test_pred = all_model.predict_proba(test_data)
        print(f"The test F-score of ALL is: {all_model.get_score(test_labels, test_pred, metric='accuracy')}")
        print()
    
    ###################################################################
    #### MultiALL

    # multiall = MultiALL()
    # multiall.fit(train_data, weak_signals)
    # predicted_labels = multiall.predict_proba(train_data)
    # predicted_classes = multiall.predict(train_data)

    # assert all_model.get_score(predicted_labels, predicted_classes) == 1.0

    # print(f"The train accuracy of MultiALL is: {multi_all.get_score(self, train_labels, predicted_labels, metric='accuracy')}")
    # test_pred = multiall.predict_proba(test_data)
    # print(f"The test F-score of MultiALL is: {multi_all.get_score(self, test_labels, test_pred, metric="f1")}")
    # print()
    ###################################################################
    ### CLL
    """
    cll = CLL()
    cll.fit(weak_signals)
    predicted_labels = cll.predict_proba(weak_signals)
    predicted_classes = cll.predict(predicted_labels)

    print(f"The train accuracy of CLL is: {cll.get_score(train_labels, predicted_labels, metric='accuracy')}")
    print()
    """
    ###################################################################
    ### DCWS
    """
    dcws = DataConsistency()
    dcws.fit(train_data, weak_signals)
    predicted_labels = np.squeeze(dcws.predict_proba(train_data))
    predicted_classes = dcws.predict(train_data)

    print(f"The train accuracy of DCWS is: {dcws.get_score(train_labels, predicted_labels)}")
    test_pred = np.squeeze(dcws.predict_proba(test_data))
    print(f"The test F-score of DCWS is: {dcws.get_score(test_labels, test_pred, metric='f1')}")
    print()
    """
    # ###################################################################
    #### Train an end model
    """
    model = mlp_model(train_data.shape[1], num_classes)
    model.fit(train_data, train_labels, batch_size=32, epochs=20, verbose=1)
    test_predictions = np.squeeze(model.predict(test_data))
    print(f"The test accuracy is: {dcws.get_score(test_labels, test_predictions)}")
    """


if __name__ == '__main__':
    # print("Running synthetic experiments...")

    print("Running real experiments...")
    # text and tabular experiments:
    # run_experiments(read_text_data('../datasets/imbd/'))
    # run_experiments(read_text_data('../datasets/yelp/'))
    run_experiments(read_text_data('/Users/jianhuichen/Diamonds/here.git/datasets/sst-2/'))
    # run_experiments(load_svhn(),'svhn')
    # run_experiments(load_fashion_mnist(),'fmnist')


    # experiments for datasets used in ALL
    # run_experiments(read_text_data('../datasets/breast-cancer/'))
    # run_experiments(read_text_data('../datasets/obs-network/'))
    # run_experiments(read_text_data('../datasets/cardiotocography/'))
    # run_experiments(read_text_data('../datasets/phishing/'))
