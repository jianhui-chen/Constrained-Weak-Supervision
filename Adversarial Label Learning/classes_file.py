import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"""
This is an extra file to show progress on the updated version of the Data Class
Can be later used with the overhead file running all the code
"""

class Data:

    def __get_weak_signals(self):
        """ private method """
        data = self.data
    
        # code to get weak signals –– create_weak_signals_view
        train_data, train_labels = data['training_data']
        val_data, val_labels = data['validation_data']
        test_data, test_labels = data['test_data']

        weak_signal_train_data = []
        weak_signal_val_data = []
        weak_signal_test_data = []

        for i in range(len(self.v)):
            f = self.v[i]

            weak_signal_train_data.append(train_data[:, f:f+1])
            weak_signal_val_data.append(val_data[:, f:f+1])
            weak_signal_test_data.append(test_data[:, f:f+1])

        weak_signal_data = [weak_signal_train_data, weak_signal_val_data, weak_signal_test_data]

        return weak_signal_data
    
    def __init__(self, views, datapath, savepath, load_and_process):
        self.v = views
        self.dp = datapath
        self.sp = savepath
        self.data = load_and_process(datapath)
        self.w_data = self.__get_weak_signals()
        self.num_sig = 0

    def get_views(self):
        return self.v

    def get_datapath(self):
        return self.dp

    def get_savepath(self):
        return self.sp
    
    def get_num_signals(self):
        return self.num_sig

    
    def __train_weak_signals(self):
        """
        Trains different views of weak signals

        :param data: dictionary of training and test data
        :type data: dict
        :param weak_signal_data: data representing the different views for the weak signals
        :type: array
        :param num_weak_signals: number of weak_signals
        type: in
        :return: dictionary containing of models, probabilities and error bounds of weak signals
        :rtype: dict
        """

        train_data, train_labels = self.data['training_data']
        val_data, val_labels = self.data['validation_data']
        test_data, test_labels = self.data['test_data']

        n, d = train_data.shape

        weak_signal_train_data = self.w_data[0]
        weak_signal_val_data = self.w_data[1]
        weak_signal_test_data = self.w_data[2]

        weak_signals = []
        stats = np.zeros(self.num_sig)
        w_sig_probabilities = []
        w_sig_test_accuracies = []
        weak_val_accuracy = []


        for i in range(self.num_sig):
            # fit model
            model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
            model.fit(weak_signal_train_data[i], train_labels)
            weak_signals.append(model)

            # evaluate probability of P(X=1)
            probability = model.predict_proba(weak_signal_val_data[i])[:, 1]
            score = val_labels * (1 - probability) + (1 - val_labels) * probability
            stats[i] = np.sum(score) / score.size
            w_sig_probabilities.append(probability)

            # evaluate accuracy for validation data
            weak_val_accuracy.append(accuracy_score(val_labels, np.round(probability)))

            # evaluate accuracy for test data
            test_predictions = model.predict(weak_signal_test_data[i])
            w_sig_test_accuracies.append(accuracy_score(test_labels, test_predictions))


        model = {}
        model['models'] = weak_signals
        model['probabilities'] = np.array(w_sig_probabilities)
        model['error_bounds'] = stats
        model['validation_accuracy'] = weak_val_accuracy
        model['test_accuracy'] = w_sig_test_accuracies

        return model


    def get_data(self, total_weak_signals):
       
        w_models = []
        self.num_sig = total_weak_signals

        for num_weak_signals in range(1, total_weak_signals + 1):
            w_models.append(self.__train_weak_signals())


        return w_models

    

    