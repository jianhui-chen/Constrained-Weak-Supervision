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
    train_data, train_labels = data['training_data']
    val_data, val_labels = data['validation_data']
    test_data, test_labels = data['test_data']

    weak_signal_train_data = []
    weak_signal_val_data = []
    weak_signal_test_data = []

    for i in range(len(data_obj.v)):
        f = data_obj.v[i]

        weak_signal_train_data.append(train_data[:, f:f+1])
        weak_signal_val_data.append(val_data[:, f:f+1])
        weak_signal_test_data.append(test_data[:, f:f+1])

    weak_signal_data = [weak_signal_train_data, weak_signal_val_data, weak_signal_test_data]

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

    train_data, train_labels = data_obj.data['training_data']
    val_data, val_labels = data_obj.data['validation_data']
    test_data, test_labels = data_obj.data['test_data']

    n, d = train_data.shape

    w_data = get_weak_signals(data_obj)

    weak_signal_train_data = w_data[0]
    weak_signal_val_data = w_data[1]
    weak_signal_test_data = w_data[2]

    weak_signals = []
    stats = np.zeros(num_weak_signals)
    w_sig_probabilities = []
    w_sig_test_accuracies = []
    weak_val_accuracy = []


    for i in range(num_weak_signals):
        # fit model
        lr_model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        lr_model.fit(weak_signal_train_data[i], train_labels)
        weak_signals.append(lr_model)

        # evaluate probability of P(X=1)
        probability = lr_model.predict_proba(weak_signal_val_data[i])[:, 1]
        score = val_labels * (1 - probability) + (1 - val_labels) * probability
        stats[i] = np.sum(score) / score.size
        w_sig_probabilities.append(probability)

        # evaluate accuracy for validation data
        weak_val_accuracy.append(accuracy_score(val_labels, np.round(probability)))

        # evaluate accuracy for test data
        test_predictions = lr_model.predict(weak_signal_test_data[i])
        w_sig_test_accuracies.append(accuracy_score(test_labels, test_predictions))


    w_data_dict = {}
    w_data_dict['models'] = weak_signals
    w_data_dict['probabilities'] = np.array(w_sig_probabilities)
    w_data_dict['error_bounds'] = stats

    # This is later used for comparison, so we don't have to calculate again
    w_data_dict['validation_accuracy'] = weak_val_accuracy
    w_data_dict['test_accuracy'] = w_sig_test_accuracies

    return w_data_dict

def get_w_data_dicts(data_obj, min_weak_signals, total_weak_signals):
       
    w_data_dicts = []

    for num_weak_signals in range(min_weak_signals, total_weak_signals + 1):
        w_data_dicts.append(train_weak_signals(data_obj, num_weak_signals))


    return w_data_dicts