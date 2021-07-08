import numpy as np


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
