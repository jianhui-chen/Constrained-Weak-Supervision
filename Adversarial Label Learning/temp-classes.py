import pandas as pd
import numpy as np
from abc import ABC, abstractmethod, abstractproperty

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"""
Includes the following Classes:
Data
- BreastCancer
- Cardio
- Obs

LoadData
"""

class Data:
    """
    Abstract Class Data

    Outlines methods that need to be implemented for each data type

    Should return necessary data from specific data groups

    """

    @abstractproperty
    def views(self):# what exactly are the views used for?
        pass

    @abstractproperty
    def datapath(self):
        pass
    
    @abstractproperty
    def savepath(self):
        pass
    
    def get_views(self):
        return self.views

    def get_datapath(self):
        return self.datapath

    def get_savepath(self):
        return self.savepath


    # maybe change to just process and have default_reader load?
    @abstractmethod
    def get_data(self):
        """ should be same as previously written code """
        pass



class BreastCancer(Data):

    views                  = {0:0, 1:10, 2:20}
    datapath               = 'datasets/breast-cancer/wdbc.data'
    savepath               = 'results/json/breast_cancer.json'

    def get_data(self):
        df = pd.read_csv(datapath, header=None)
        #Remove the first column of the data (the ID)
        df.drop([0], axis=1, inplace=True)
        #replace labels 'B' with 0 and 'M' with 1
        df[1] = df[1].replace({'B': 0, 'M': 1})
        data_matrix = df.values
        #Split the data into 70% training and 30% test set
        data_labels = data_matrix[:, :1].ravel() 
        data_matrix = data_matrix[:, 1:]
        train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels, test_size=0.3, shuffle=True, stratify=data_labels)

        #Normalize the features of the data
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        assert train_labels.size == train_data.shape[0]
        assert test_labels.size == test_data.shape[0]

        data = {}

        val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels, test_size=0.4285, shuffle=True, stratify=train_labels)

        data['training_data'] = weak_supervision_data, weak_supervision_labels
        data['validation_data'] = val_data, val_labels
        data['test_data'] = test_data, test_labels

        return data

class Cardio(Data):
    
    views                  = {0:1, 1:10, 2:18}
    datapath               = 'datasets/cardiotocography/cardio.csv'
    savepath               = 'results/json/cardio.json'

    def get_data(self):
        df = pd.read_csv(datapath)
        #Use class 1 and 2 labels only in the data
        mask_1 = df['CLASS'] == 1
        mask_2 = df['CLASS'] == 2
        df = df[mask_1 | mask_2].copy(deep=True)

        #replace labels '1' with 0 and '2' with 1
        df['CLASS'] = df['CLASS'].replace({1: 0, 2: 1})
        data_matrix = df.values

        #Split the data into 70% training and 30% test set
        data_labels = data_matrix[:,-1:].ravel() 
        data_matrix = data_matrix[:,:-1]
        train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)
        
        #Normalize the features of the data
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        assert train_labels.size == train_data.shape[0]
        assert test_labels.size == test_data.shape[0]

        data = {}

        val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_labels)

        data['training_data'] = weak_supervision_data, weak_supervision_labels
        data['validation_data'] = val_data, val_labels
        data['test_data'] = test_data, test_labels

        return data

class Obs(Data):
    
    views                  = {0:1, 1:2, 2:20}
    datapath               = 'datasets/obs-network/obs_network.data'
    savepath               = 'results/json/obs_network.json'

    def get_data(self):
        df = pd.read_csv(datapath, header=None)
        #Use 'NB-Wait' and "'NB-No Block'" labels only in the data
        mask_no_block = df[21] == "'NB-No Block'"
        mask_wait = df[21] == 'NB-Wait'
        df = df[mask_no_block | mask_wait].copy(deep=True)

        # remove rows with missing values
        df = df[df[13] != '?']

        #replace node status feature 'NB', 'B' and 'P NB' with 1, 2, 3
        df[19] = df[19].replace(['NB', 'B', "'P NB'"], [1, 2, 3])

        #replace labels "'NB-No Block'" with 0 and 'NB-Wait' with 1
        df[21] = df[21].replace({"'NB-No Block'": 0, 'NB-Wait': 1})
        data_matrix = df.values

        #Split the data into 70% training and 30% test set
        data_labels = data_matrix[:,-1:].ravel() 
        data_matrix = data_matrix[:,:-1]
        train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)

        #Normalize the features of the data
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        assert train_labels.size == train_data.shape[0]
        assert test_labels.size == test_data.shape[0]

        data = {}

        val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_labels)

        data['training_data'] = weak_supervision_data, weak_supervision_labels
        data['validation_data'] = val_data, val_labels
        data['test_data'] = test_data, test_labels

        return data


class LoadData:     # I'm not sure if we want to keep this as a class... seems inefficient

    def get_all_data(self, data_object):
        data = data_object.get_data()
        
        # code to get weak signals –– create_weak_signals_view
        train_data, train_labels = data['training_data']
        val_data, val_labels = data['validation_data']
        test_data, test_labels = data['test_data']

        weak_signal_train_data = []
        weak_signal_val_data = []
        weak_signal_test_data = []

        for i in range(len(data_object.get_views())):
            f = data_object.get_views()[i]

            weak_signal_train_data.append(train_data[:, f:f+1])
            weak_signal_val_data.append(val_data[:, f:f+1])
            weak_signal_test_data.append(test_data[:, f:f+1])

        weak_signal_data = [weak_signal_train_data, weak_signal_val_data, weak_signal_test_data]

        return data, weak_signal_data


    def train_weak_signals(self, data, weak_signal_data, num_weak_signals):
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

            train_data, train_labels = data['training_data']
            val_data, val_labels = data['validation_data']
            test_data, test_labels = data['test_data']

            n, d = train_data.shape

            weak_signal_train_data = weak_signal_data[0]
            weak_signal_val_data = weak_signal_data[1]
            weak_signal_test_data = weak_signal_data[2]

            weak_signals = []
            stats = np.zeros(num_weak_signals)
            w_sig_probabilities = []
            w_sig_test_accuracies = []
            weak_val_accuracy = []


            for i in range(num_weak_signals):
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


    def load_data(self, data_object, total_weak_signals): # return array of models for each signal, will loop when doing run

        data, weak_signal_data = get_all_data(data_object)

        w_models = []

        for num_weak_signals in range(1, total_weak_signals + 1):
            w_models.append(train_weak_signals(data, weak_signal_data, num_weak_signals))

        return data, w_models