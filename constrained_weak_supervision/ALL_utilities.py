# import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
# from sklearn import preprocessing


import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



"""
This is an extra file to show progress on the updated version of the Data Class
Can be later used with the overhead file running all the code
"""

class Data:

    def __load_and_process_data(self, datapath, load_data):

        """
            Calls function that gets the data, and then splits it into training, validation, and testing sets

            :param datapath: location of data
            :type datapath: string
            :param load_data: funtion specific to data set that gets data from the path and cleans it
            :type load_data: function
        """

        data_matrix, data_labels = load_data(datapath)

        #Split the data into 70% training and 30% test set
        train_dev_data, test_data, train_dev_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)

        #Normalize the features of the data
        scaler = preprocessing.StandardScaler().fit(train_dev_data)
        train_dev_data = scaler.transform(train_dev_data)
        test_data = scaler.transform(test_data)

        assert train_dev_labels.size == train_dev_data.shape[0]
        assert test_labels.size == test_data.shape[0]

        data = {}

        #Split the remaining data into 57.15% training and 42.85% test set
        train_data, weak_supervision_data, train_labels, weak_supervision_labels = train_test_split(train_dev_data, train_dev_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_dev_labels)

        data['dev_data'] = weak_supervision_data, weak_supervision_labels
        data['train_data'] = train_data, train_labels
        data['test_data'] = test_data, test_labels

        return data

    def __init__(self, views, datapath, load_data):
        self.v = views
        self.dp = datapath
        self.data = self.__load_and_process_data(datapath, load_data)








def read_text_data(datapath):
    """ 
        Read text datasets

        Parameters
        ----------
        :param datapath: file path to data files
        :type  datapath: string

        Returns
        -------
        :returns: training set, testing set, and weak signals 
                  of read in data
        :return type: dictionary of ndarrays
    """

    train_data = np.load(datapath + 'data_features.npy', allow_pickle=True)[()]
    weak_signals = np.load(datapath + 'weak_signals.npy', allow_pickle=True)[()]
    train_labels = np.load(datapath + 'data_labels.npy', allow_pickle=True)[()]
    test_data = np.load(datapath +'test_features.npy', allow_pickle=True)[()]
    test_labels = np.load(datapath + 'test_labels.npy', allow_pickle=True)[()]

    if len(weak_signals.shape) == 2:
        weak_signals = np.expand_dims(weak_signals.T, axis=-1)

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals
    return data

"""

    train_data = np.load(datapath + 'data_features.npy', allow_pickle=True)[()]
    weak_signals = np.load(datapath + 'weak_signals.npy', allow_pickle=True)[()]
    train_labels = np.load(datapath + 'data_labels.npy', allow_pickle=True)[()]
    test_data = np.load(datapath +'test_features.npy', allow_pickle=True)[()]
    test_labels = np.load(datapath + 'test_labels.npy', allow_pickle=True)[()]


    data = data_obj.data
    train_data, train_labels = data['train_data']
    train_data = train_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]




"""

# ------------------------------------------------------------------------- #
# Code for Loading and processing all data sets                             #
# ------------------------------------------------------------------------- #

def breast_cancer_load_and_process_data(path):

    """Loads breast cancer data from `path`"""
    df = pd.read_csv(path, header=None)
    #Remove the first column of the data (the ID)
    df.drop([0], axis=1, inplace=True)
    #replace labels 'B' with 0 and 'M' with 1
    df[1] = df[1].replace({'B': 0, 'M': 1})
    #Convert to array
    data_matrix = df.to_numpy()

    #Seperate Data and labels 
    data_labels = data_matrix[:, :1].ravel() 
    data_matrix = data_matrix[:, 1:]

    

    return data_matrix, data_labels


def cardio_load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path)
    #Use class 1 and 2 labels only in the data
    mask_1 = df['CLASS'] == 1
    mask_2 = df['CLASS'] == 2
    df = df[mask_1 | mask_2].copy(deep=True)

    #replace labels '1' with 0 and '2' with 1
    df['CLASS'] = df['CLASS'].replace({1: 0, 2: 1})
    #Convert to array
    data_matrix = df.to_numpy()

    #Seperate Data and labels 
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]

    return data_matrix, data_labels

def obs_load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path, header=None)
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

    #Convert to array
    data_matrix = df.to_numpy()

    #Seperate Data and labels 
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]

    return data_matrix, data_labels

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from data_readers import obs_load_and_process_data, cardio_load_and_process_data, breast_cancer_load_and_process_data
from classes_file import Data


# """
#     Note:
#     Binary ALL has three sets –– Dev, Train, Test
#     Multi ALL and CLL has 2 sets –– Train, Test
#         Dev and Train are combined into one
# """

def get_weak_signal_data(data_obj):
    """
    Isolates dev and train data for fitting model for weak signals and
    calculating probability and error bounds.
    """
    data = data_obj.data

    dev_data, dev_labels = data['dev_data']
    train_data, train_labels = data['train_data']

    weak_signal_dev_data = []       # Used for fitting model
    weak_signal_train_data = []     # Used for calculating the probabilities + error bounds

    for i in range(len(data_obj.v)):
        f = data_obj.v[i]

        weak_signal_dev_data.append(dev_data[:, f:f+1])
        weak_signal_train_data.append(train_data[:, f:f+1])

    weak_signal_data = [weak_signal_dev_data, weak_signal_train_data]

    return weak_signal_data


def train_weak_signals(data_obj, num_weak_signals):
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

    dev_data, dev_labels = data_obj.data['dev_data']
    train_data, train_labels = data_obj.data['train_data']

    n, d = dev_data.shape

    w_data = get_weak_signal_data(data_obj)

    # This is to train the LR model + get statistics
    weak_signal_dev_data = w_data[0]    # Used for fitting the model
    weak_signal_train_data = w_data[1]  # Used for stats

    error_bounds = []
    probabilities = []

    for i in range(num_weak_signals):
        # fit model
        lr_model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        lr_model.fit(weak_signal_dev_data[i], dev_labels)

        # get probabilities and error bounds
        probability = lr_model.predict_proba(weak_signal_train_data[i])[:, 1]

        score = train_labels * (1 - probability) + (1 - train_labels) * probability
        error_bound = np.sum(score) / score.size
        error_bounds.append(error_bound)
        probabilities.append(probability)

    weak_signals = {}
    weak_signals['probabilities'] = np.array(probabilities).T
    weak_signals['error_bounds'] = np.array(error_bounds)

    return weak_signals


def get_weak_signals(data_obj, num_weak_signals, weak_signal_path=None):
    """
    Generates weak_signal probabilities and error bounds or gets them from the data path

    Parameters
    ----------
    data_obj : Data object
        Has access to data pre-divided into dev, train, and test.
        dev and train can be combined as needed

    max_weak_signals : int
        Gives number of weak signals

    weak_signal_path : string, default=None
        If not default, gives path to load weak_signal information from.
        Throws

    Returns
    -------
    PLACEHOLDER NAME : dict
        Dictionary containing the probas and error bounds

    """
    if weak_signal_path is not None:
        # Have console command stating this is in progress

        weak_signals = {}
        weak_signals['probabilities'] = np.load(weak_signal_path + str(num_weak_signals) +
                                                '_weak_signals/weak_signals_probabilities.npy', allow_pickle=True)[()]
        weak_signals['error_bounds'] = np.load(weak_signal_path + str(num_weak_signals) +
                                               '_weak_signals/weak_signals_error_bounds.npy', allow_pickle=True)[()]

    else:
        # Console command
        weak_signals = train_weak_signals(data_obj, num_weak_signals)

    return weak_signals


def file_generator(datapath, savepath, views, load_and_process_data) :
    """
    Breaks down data from provided dataset and seperates it into files


    Parameters
    ----------
    datapath : string
        Path to where entire dataset is located

    savepath : string
        Path to folder where files will be stored

    views : list of ints,
        locations in data set where weak signals will be generated from

    load_and_process_data : function
        function that helps to read and clean provided dataset


    Returns
    -------
    nothing
    """

    obs_data = Data(views, datapath, load_and_process_data)

    # Generate multiple weak_signal probabilities and error bounds
    #     NOTE: Can edit later so that only runs function once, also
    #           so it returns only weak signals and not errors too
    multiple_weak_signals = []
    for num_weak_signals in range(1, 3 + 1):
        weak_signals = get_weak_signals(obs_data, num_weak_signals)
        multiple_weak_signals.append(weak_signals)

    # Get data to store
    data = obs_data.data
    train_data, train_labels = data['train_data']
    test_data = data['test_data'][0]
    test_labels = data['test_data'][1]

    # save data, labels, and weak_signals
    np.save(savepath+'data_features.npy', train_data)
    np.save(savepath+'test_features.npy', test_data)
    np.save(savepath+'data_labels.npy', train_labels)
    np.save(savepath+'test_labels.npy', test_labels)
    np.save(savepath+'weak_signals.npy', multiple_weak_signals[2]['probabilities'])


print("\n\n working on OBS \n\n" )
file_generator('obs/obs_network.data', './obs/', [1, 2, 20], obs_load_and_process_data)

print("\n\n working on Cardio \n\n" )
file_generator('cardio/cardio.csv', './cardio/', [1, 10, 18], cardio_load_and_process_data)

print("\n\n working on Cancer \n\n" )
file_generator('breast-cancer/wdbc.data', './breast-cancer/', [0, 10, 20], breast_cancer_load_and_process_data)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"""
Weak signals will be in the form of (# signals, # training examples)
    This has to be transposed, but all the code is already like this :(

    Binary ALL has three sets –– Dev, Train, Test
    Multi ALL and CLL has 2 sets –– Train, Test
        Dev and Train are combined into one

We can access both probabilities and error bounds of weak signals here
    Not sure about -1, 0, 1 flags

Can choose to pass in a function to generate weak signals

Can pass in datapath to save signal data
"""

def get_weak_signal_data(data_obj):
    """
    Isolates dev and train data for fitting model for weak signals and
    calculating probability and error bounds.

    """
    data = data_obj.data

    # code to get weak signals –– create_weak_signals_view
    dev_data, dev_labels = data['dev_data']
    train_data, train_labels = data['train_data']

    weak_signal_dev_data = []       # Used for fitting model
    weak_signal_train_data = []     # Used for calculating the probabilities + error bounds

    for i in range(len(data_obj.v)):
        f = data_obj.v[i]

        weak_signal_dev_data.append(dev_data[:, f:f+1])
        weak_signal_train_data.append(train_data[:, f:f+1])

    weak_signal_data = [weak_signal_dev_data, weak_signal_train_data]

    return weak_signal_data


def train_weak_signals(data_obj, num_weak_signals, savepath):
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

    dev_data, dev_labels = data_obj.data['dev_data']
    train_data, train_labels = data_obj.data['train_data']

    n, d = dev_data.shape

    w_data = get_weak_signal_data(data_obj)

    # This is to train the LR model + get statistics
    weak_signal_dev_data = w_data[0]    # Used for fitting the model
    weak_signal_train_data = w_data[1]  # Used for stats

    error_bounds = []
    probabilities = []

    for i in range(num_weak_signals):
        # fit model
        lr_model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        lr_model.fit(weak_signal_dev_data[i], dev_labels)

        # get probabilities and error bounds
        probability = lr_model.predict_proba(weak_signal_train_data[i])[:, 1]
        score = train_labels * (1 - probability) + (1 - train_labels) * probability
        error_bound = np.sum(score) / score.size
        error_bounds.append(error_bound)
        probabilities.append(probability)


    weak_signals = {}
    weak_signals['probabilities'] = np.array(probabilities)
    weak_signals['error_bounds'] = np.array(error_bounds)

    # np.save(savepath + str(num_weak_signals) + '_weak_signals/weak_signals_probabilities.npy', np.array(probabilities))
    # np.save(savepath + str(num_weak_signals) + '_weak_signals/weak_signals_error_bounds.npy', np.array(error_bounds))

    return weak_signals


def get_weak_signals(data_obj, num_weak_signals, savepath,
                     weak_signal_proba_func=None, weak_signal_error_func=None,
                     weak_signal_path=None):
    """
    Generates weak_signal probabilities and error bounds

    Parameters
    ----------
    data_obj : Data object
        Has access to data pre-divided into dev, train, and test.
        dev and train can be combined as needed

    max_weak_signals : int
        Gives number of weak signals

    savepath : string
        Expect to end in / and be the name of the algo

    weak_signal_proba_func : function, default=None
        Will use to get probabilities, can adjust the ones from multi ALL and
        CLL to fit this format

    weak_signal_error_func : function, default=None
        Same as above, for error bounds, varies for CLL and ALL

    weak_signal_path : string, default=None
        If not default, gives path to load weak_signal information from.
        Throws

    Returns
    -------
    PLACEHOLDER NAME : dict
        Dictionary containing the probas and error bounds

    """
    if weak_signal_path is not None:
        # Have console command stating this is in progress

        weak_signals = {}
        weak_signals['probabilities'] = np.load(weak_signal_path + str(num_weak_signals) + '_weak_signals/weak_signals_probabilities.npy', allow_pickle=True)[()]
        weak_signals['error_bounds'] = np.load(weak_signal_path + str(num_weak_signals) + '_weak_signals/weak_signals_error_bounds.npy', allow_pickle=True)[()]

    elif ((weak_signal_proba_func is not None) and
          (weak_signal_error_func is not None)):    # Expect to save within those functions

        # Have console command stating this is in progress

        weak_signals={}         # placeholder until we decide how these functions are called

    else:

        # Console command

        weak_signals = train_weak_signals(data_obj, num_weak_signals, savepath)

    return weak_signals




def get_multiple_weak_signals(data_obj, min_weak_signals, max_weak_signals,
                              savepath, weak_signal_proba_func=None,
                              weak_signal_error_func=None,
                              weak_signal_path=None):
    """
    Generates multiple weak_signal probabilities and error bounds

    Parameters
    ----------
    data_obj : Data object
        Has access to data pre-divided into dev, train, and test.
        dev and train can be combined as needed

    min_weak_signals : int
        Gives Minimum number of signals we want to have

    max_weak_signals : int
        Gives max number of weak signals

    weak_signal_proba_func : function, default=None
        Will use to get probabilities, can adjust the ones from multi ALL and
        CLL to fit this format

    weak_signal_error_func : function, default=None
        Same as above, for error bounds, varies for CLL and ALL

    weak_signal_path : string, default=None
        If not default, gives path to load weak_signal information from.
        Throws

    Returns
    -------
    PLACEHOLDER NAME : list
        List of dictionaries, might want to change

    """

    multiple_weak_signals = []

    for num_weak_signals in range(min_weak_signals, max_weak_signals + 1):
        weak_signals = get_weak_signals(data_obj, num_weak_signals, savepath,
                                        weak_signal_proba_func=weak_signal_proba_func,
                                        weak_signal_error_func=weak_signal_error_func,
                                        weak_signal_path=weak_signal_path)
        multiple_weak_signals.append(weak_signals)

    return multiple_weak_signals

