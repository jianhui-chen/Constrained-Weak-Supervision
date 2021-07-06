import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def get_weak_signals(data_obj):

    # get data, and then split it into groups
    # data_matrix, data_labels = self.data
    # data = self.load_and_process_data(data_matrix, data_labels)

    data = data_obj.data

    # code to get weak signals –– create_weak_signals_view
    dev_data, dev_labels = data['dev_data']
    train_data, train_labels = data['train_data']
    test_data, test_labels = data['test_data']

    weak_signal_dev_data = []       # Used for fitting model
    weak_signal_train_data = []     # Used for calculating the probabilities + error bounds
    weak_signal_test_data = []      # Currently not used

    for i in range(len(data_obj.v)):
        f = data_obj.v[i]

        weak_signal_dev_data.append(dev_data[:, f:f+1])
        weak_signal_train_data.append(train_data[:, f:f+1])
        weak_signal_test_data.append(test_data[:, f:f+1])

    weak_signal_data = [weak_signal_dev_data, weak_signal_train_data, weak_signal_test_data]


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
    test_data, test_labels = data_obj.data['test_data']

    n, d = dev_data.shape

    w_data = get_weak_signals(data_obj)

    # This is to train the LR model + get statistics
    weak_signal_dev_data = w_data[0]    # Used for fitting the model
    weak_signal_train_data = w_data[1]  # Used for stats
    weak_signal_test_data = w_data[2]   # This is not used at all

    #weak_signals = []
    stats = np.zeros(num_weak_signals)
    w_sig_probabilities = []
    # w_sig_test_accuracies = []
    #weak_train_accuracy = []


    for i in range(num_weak_signals):
        # fit model
        lr_model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        lr_model.fit(weak_signal_dev_data[i], dev_labels)
        #weak_signals.append(lr_model)
        # evaluate probability of P(X=1)
        probability = lr_model.predict_proba(weak_signal_train_data[i])[:, 1]
        score = train_labels * (1 - probability) + (1 - train_labels) * probability
        stats[i] = np.sum(score) / score.size
        w_sig_probabilities.append(probability)

        # evaluate accuracy for validation data
        #weak_train_accuracy.append(accuracy_score(train_labels, np.round(probability)))

        # # evaluate accuracy for test data
        # test_predictions = lr_model.predict(weak_signal_test_data[i])
        # w_sig_test_accuracies.append(accuracy_score(test_labels, test_predictions))
    

    # print("weak_signals:", weak_signals)
    # print("\n\nprobabilities:", np.array(w_sig_probabilities), "\n\n")
    # print("\n\error_bounds:", stats, "\n\n")


    w_data_dict = {}
    # w_data_dict['models'] = weak_signals
    w_data_dict['probabilities'] = np.array(w_sig_probabilities)
    w_data_dict['error_bounds'] = stats

    # This is later used for comparison, so we don't have to calculate again
    # w_data_dict['train_accuracy'] = weak_train_accuracy
    # w_data_dict['test_accuracy'] = w_sig_test_accuracies

    return w_data_dict

def get_w_data_dicts(data_obj, min_weak_signals, total_weak_signals, weak_sig_datapath=None):
       
    w_data_dicts = []

    #train weak signals when none are presented
    if weak_sig_datapath is None:
        for num_weak_signals in range(min_weak_signals, total_weak_signals + 1):
            w_data_dicts.append(train_weak_signals(data_obj, num_weak_signals))

    #train get weak signals from path
    # This mimics what's in the generate_weak_signals code
    else:
        w_data_dict = {}
        w_data_dict['probabilities'] = np.load(weak_sig_datapath + 'weak_signals.npy', allow_pickle=True)[()]

        try:
            w_data_dict['error_bounds'] = np.load(weak_sig_datapath + 'error_rates.npy', allow_pickle=True)[()]
        except: 
            w_data_dict['error_bounds'] = np.zeros(w_data_dict['probabilities'].size) + 0.3


    return w_data_dicts




# # # # # # # # # # # # # # # # # # # # # # # # # 
# Maybe use in future as a model                #
# # # # # # # # # # # # # # # # # # # # # # # # # 
#
# def read_text_data(datapath):
#     # read in the data
#     data_features = np.load(datapath+'data_features.npy', allow_pickle=True)[()]
#     weak_signals = np.load(datapath+'weak_signals.npy', allow_pickle=True)[()]
#     try:
#         labels = np.load(datapath+'data_labels.npy', allow_pickle=True)[()]
#         has_labels = True
#     except:
#         labels = False
#         has_labels = False

#     try:
#         test_features = np.load(datapath+'test_features.npy', allow_pickle=True)[()]
#         test_labels = np.load(datapath+'test_labels.npy', allow_pickle=True)[()]
#         has_test_data = True
#     except:
#         test_features = False
#         test_labels = False
#         has_test_data = False

#     if len(weak_signals.shape) == 2:
#         n,m = weak_signals.shape
#         assert n>=m
#         weak_signals = weak_signals.T.reshape(m,n,1)


#     data = {}
#     data['data_features'] = data_features
#     data['weak_signals'] = weak_signals
#     data['labels'] = labels
#     data['test_features'] = test_features
#     data['test_labels'] = test_labels
#     data['has_test_data'] = has_test_data
#     data['has_labels'] = has_test_data

#     return data