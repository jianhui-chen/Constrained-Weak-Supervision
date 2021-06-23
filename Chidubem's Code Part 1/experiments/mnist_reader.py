import os
import gzip
import numpy as np
from sklearn import preprocessing

def load_data(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_dataset(datapath):
    """
    Loads dataset

    :param datapath: path to data
    :type datapath: string
    :param dataloader: script to load the dataset
    :type dataloader: python module
    :return: dictionary containing tuples of training, and test data
    :rtype: dict
    """
    train_data, train_labels = load_data(datapath, kind='train')
    test_data, test_labels = load_data(datapath, kind='t10k')

    data = {}

    data['training_data'] = (train_data, train_labels)
    data['test_data'] = (test_data, test_labels)

    return data


def processFashionMnist(path, ones_class, zeros_class):

    """
    Processes fashion mnist dataset to be used for binary classification

    :param ones_zeros_class: integers for which fashion mnist class to train on
    type: int
    return: dictionary containing tuples of training and test data
    rtype: dict
    """

    label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}


    data = load_dataset(path)

    train_data = data['training_data'][0]
    train_labels = data['training_data'][1]
    test_data = data['test_data'][0]
    test_labels = data['test_data'][1]

    #Normalize the feature of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Process training data to binary
    b_train_labels = train_labels[(train_labels==ones_class) | (train_labels==zeros_class)]
    indices = np.where(np.in1d(train_labels, b_train_labels))[0]
    b_train_data = train_data[indices]

    assert b_train_data.shape[0] == b_train_labels.size

    # Process test data to binary
    b_test_labels = test_labels[(test_labels==ones_class) | (test_labels==zeros_class)]
    indices = np.where(np.in1d(test_labels, b_test_labels))[0]
    b_test_data = test_data[indices]

    assert b_test_data.shape[0] == b_test_labels.size

    # Change training labels to 0 or 1
    train_labels = np.zeros(b_train_labels.size)
    train_labels[b_train_labels == ones_class] = 1

    # Chnage test labels to 0 or 1
    test_labels = np.zeros(b_test_labels.size)
    test_labels[b_test_labels == ones_class] = 1

    data = {}

    train_size = int(train_labels.size / 2)

    data['training_data'] = (b_train_data[:train_size, :], train_labels[:train_size])
    data['validation_data'] = (b_train_data[train_size:, :], train_labels[train_size:])
    data['test_data'] = (b_test_data, test_labels)

    return data


def create_weak_signal_view(path, ones_class, zeros_class):

    data = processFashionMnist(path, ones_class, zeros_class)

    train_data, train_labels = data['training_data']
    val_data, val_labels = data['validation_data']
    test_data, test_labels = data['test_data']

    weak_signal_train_data = []
    weak_signal_val_data = []
    weak_signal_test_data = []

    #for fashion mnist dataset, select the 1/4 feature, middle feature and the 3/4 feature as weak signals
    views = {0:195, 1:391, 2:587}

    for i in range(3):
        #pick a random feature for the individual weak signals
        f = views[i]

        weak_signal_train_data.append(train_data[:, f:f+1])
        weak_signal_val_data.append(val_data[:, f:f+1])
        weak_signal_test_data.append(test_data[:, f:f+1])

    weak_signal_data = [weak_signal_train_data, weak_signal_val_data, weak_signal_test_data]

    return data, weak_signal_data


def run_experiment(run, save):

    """
    :param run: method that runs real experiment given data
    :type: function
    :param save: method that saves experiment results to JSON file
    :type: function
    :return: none
    """

    classes = [(3,7), (5,9), (4,8)]
    classes = [(4,8)]

    # set up your variables
    total_weak_signals = 3
    num_experiments = 1

    for item in classes:
        ones_class = item[0]
        zeros_class = item[1]

        for i in range(num_experiments):

            data, weak_signal_data= create_weak_signal_view('datasets/fashion-mnist', ones_class, zeros_class)
            for num_weak_signal in range(1, total_weak_signals + 1):
                adversarial_model, weak_model = run(data, weak_signal_data, num_weak_signal)
                print("Saving results to file...")
                adversarial_model['classes_used'] = [ones_class,zeros_class]
                # save(adversarial_model, weak_model, 'results/json/fashion-mnist.json')

            # for num_weak_signal in range(1, total_weak_signals + 1):
            #     adversarial_model, weak_model = run(data, weak_signal_data, num_weak_signal, constant_bound=True)
            #     print("Saving results to file...")
            #     adversarial_model['classes_used'] = [ones_class,zeros_class]
            #     save(adversarial_model, weak_model, 'results/json/fashion-mnist.json')
