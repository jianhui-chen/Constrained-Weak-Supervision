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
        weak_signals['probabilities'] = np.load(weak_signal_path + str(num_weak_signals) + '_weak_signals/weak_signals_probabilities.npy', allow_pickle=True)[()]
        weak_signals['error_bounds'] = np.load(weak_signal_path + str(num_weak_signals) + '_weak_signals/weak_signals_error_bounds.npy', allow_pickle=True)[()]
        
    else:
        # Console command
        weak_signals = train_weak_signals(data_obj, num_weak_signals)

    return weak_signals




def file_generator(datapath, savepath, views, load_and_process_data) :
    """
    breaks down data from provided dataset and seperates it into files
    
    
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
