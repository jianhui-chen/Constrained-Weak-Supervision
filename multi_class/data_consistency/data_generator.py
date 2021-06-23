import os
import sys
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification
from sklearn.model_selection import train_test_split

SEED = 1000


def separable_train_data(no_of_samples, no_of_labels):
    """ Create separable train data"""
    data, y = make_blobs(n_samples=no_of_samples, centers=2,
                         n_features=2, cluster_std=2, random_state=SEED)
    return data, y


def noisy_train_data(no_of_samples, noise_ratio=0):
    """ Create noisy train data"""
    data, y = make_circles(n_samples=no_of_samples, factor=.5,
                           noise=noise_ratio, random_state=SEED)
    return data, y


def make_weaksignals(labels, no_weak_signals):
    """ Create weak signals"""
    n = labels.size
    Ys = labels.copy()
    Ys[Ys == 0] = -1

    np.random.seed(SEED)
    ws_accs = 0.1 * np.random.random((no_weak_signals,)) + 0.55
    WS_COVERAGE = 1
    Ws = np.zeros((n, no_weak_signals))
    for i in range(n):
        for j in range(no_weak_signals):
            if np.random.random() < WS_COVERAGE:
                Ws[i, j] = Ys[i] if np.random.random() < ws_accs[j] else -Ys[i]

    # weak_signals to correct format
    weak_signals = Ws.copy()
    weak_signals[Ws == 0] = -1
    weak_signals[Ws == -1] = 0

    weak_signals = weak_signals.T.reshape(no_weak_signals, n, 1)
    return weak_signals


def synthetic_2D(no_samples=10000, no_signals=10, form='sep'):
    """ Create sybthetic 2D data"""
    if form == 'sep':
        data, labels = separable_train_data(no_samples, 2)
    else:
        data, labels = noisy_train_data(no_samples, 2)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=500)
    weak_signals = make_weaksignals(train_labels, no_signals)
    train_labels = np.expand_dims(train_labels, 1)

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals
    return data


def synthetic_data(num_data=20000, num_weak_signals=10):
    """ Create synthetic data for the main experiments"""
    np.random.seed(900)

    n = num_data  # no of data points
    d = 200  # no of features
    Ys = 2 * np.random.randint(2, size=(n,)) - 1  # true labels

    feature_accs = 0.1 * np.random.random((d,)) + 0.55
    Xs = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            if np.random.random() > feature_accs[j]:
                Xs[i, j] = 1 if Ys[i] == 1 else 0
            else:
                Xs[i, j] = 0 if Ys[i] == 1 else 1

    def dependent_signals(no_signals):
        # create dependencies between weak signals
        seed = 500
        np.random.seed(seed)
        ws_accs = 0.1 * np.random.random() + 0.55
        coverage = 1.0
        weak_signal = np.zeros(n)
        for i in range(n):
            if np.random.random() < coverage:
                weak_signal[i] = Ys[i] if np.random.random(
                ) < ws_accs else -Ys[i]

        signals = [weak_signal]
        for i in range(no_signals - 1):
            signal = weak_signal.copy()
            seed += 10
            np.random.seed(seed)
            indices = np.random.choice(n, int(n * 0.05), replace=False)
            for j in indices:
                signal[j] = -1 * signal[j]
            signals.append(signal)
        signals = np.asarray(signals).T
        weak_signals = signals.copy()
        weak_signals[signals == 0] = -1
        weak_signals[signals == -1] = 0
        return weak_signals

    # Weak signals for the main experiments in the paper
    weak_signals = make_weaksignals(Ys, num_weak_signals)

    # Weak signals for dependent experiments in the appendix
    # weak_signals = dependent_signals(num_weak_signals)
    # weak_signals = np.expand_dims(weak_signals.T, axis=-1)

    # Convert Y and weak_signals to correct format
    train_data = Xs
    train_labels = 0.5 * (Ys + 1)

    indexes = np.arange(n)
    np.random.seed(2000)
    test_indexes = np.random.choice(n, int(n * 0.2), replace=False)
    weak_signals = np.delete(weak_signals, test_indexes, axis=1)

    test_labels = train_labels[test_indexes]
    test_data = train_data[test_indexes]
    train_indexes = np.delete(indexes, test_indexes)
    train_labels = train_labels[train_indexes]
    train_data = train_data[train_indexes]

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals

    return data


if __name__ == '__main__':
    pass
