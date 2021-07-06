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

    np.save(savepath + str(num_weak_signals) + '_weak_signals/weak_signals_probabilities.npy', np.array(probabilities))
    np.save(savepath + str(num_weak_signals) + '_weak_signals/weak_signals_error_bounds.npy', np.array(error_bounds))

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