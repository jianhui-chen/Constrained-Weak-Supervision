import os, sys
import gzip
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.python.keras.datasets import cifar10
import scipy.io as sio


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(
        .59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


def load_svhn():

    data = {}
    categories = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine"
    ]

    def load_images(path):
        train_images = sio.loadmat(path + '/train_32x32.mat')
        test_images = sio.loadmat(path + '/test_32x32.mat')
        return train_images, test_images

    def normalize_images(images):
        imgs = images["X"]
        imgs = np.transpose(imgs, (3, 0, 1, 2))
        labels = images["y"]
        # replace label "10" with label "0"
        labels[labels == 10] = 0
        return imgs, labels

    train_images, test_images = load_images('../datasets/svhn')
    train_images, train_labels = normalize_images(train_images)
    test_images, test_labels = normalize_images(test_images)

    train_size, img_rows, img_cols, channels = train_images.shape
    test_size, img_rows, img_cols, channels = test_images.shape
    y_train, y_test = train_labels.ravel(), test_labels.ravel()

    data['img_rows'] = img_rows
    data['img_cols'] = img_cols
    data['channels'] = channels
    data['num_classes'] = 10
    data['categories'] = categories
    data['path'] = 'svhn'

    x_train = train_images.reshape(train_size, img_rows * img_cols * channels)
    x_test = test_images.reshape(test_size, img_rows * img_cols * channels)

    data['train_data'] = x_train, y_train
    data['test_data'] = x_test, y_test

    return data


def load_cifar_10():
    # Returns dictionary of training and test data

    data = {}
    categories = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
        "horse", "ship", "truck"
    ]

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = grayscale(x_train)
    # x_test = grayscale(x_test)

    train_size, img_rows, img_cols, channels = x_train.shape
    test_size, img_rows, img_cols, channels = x_test.shape
    y_train, y_test = y_train.ravel(), y_test.ravel()

    data['img_rows'] = img_rows
    data['img_cols'] = img_cols
    data['channels'] = channels
    data['num_classes'] = 10
    data['categories'] = categories
    data['path'] = 'cifar10'

    x_train = x_train.reshape(train_size, img_rows * img_cols * channels)
    x_test = x_test.reshape(test_size, img_rows * img_cols * channels)

    data['train_data'] = x_train, y_train
    data['test_data'] = x_test, y_test

    return data


def load_fashion_mnist():
    # Returns dictionary of training and test data
    data = {}

    categories = [
        "t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt",
        "sneaker", "bag", "ankle boot"
    ]

    def load_data(path, kind='train'):
        """Load MNIST data from `path`"""

        # label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        #               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    data['train_data'] = load_data('../datasets/fashion-mnist', kind='train')
    data['test_data'] = load_data('../datasets/fashion-mnist', kind='t10k')
    data['img_rows'] = 28
    data['img_cols'] = 28
    data['channels'] = 1
    data['num_classes'] = 10
    data['categories'] = categories
    data['path'] = 'fmnist'

    return data

def read_text_data(datapath):
    # read in the data
    data_features = np.load(datapath+'data_features.npy', allow_pickle=True)[()]
    weak_signals = np.load(datapath+'weak_signals.npy', allow_pickle=True)[()]
    try:
        labels = np.load(datapath+'data_labels.npy', allow_pickle=True)[()]
        has_labels = True
    except:
        labels = False
        has_labels = False

    try:
        test_features = np.load(datapath+'test_features.npy', allow_pickle=True)[()]
        test_labels = np.load(datapath+'test_labels.npy', allow_pickle=True)[()]
        has_test_data = True
    except:
        test_features = False
        test_labels = False
        has_test_data = False

    if len(weak_signals.shape) == 2:
        n,m = weak_signals.shape
        assert n>=m
        weak_signals = weak_signals.T.reshape(m,n,1)


    data = {}
    data['data_features'] = data_features
    data['weak_signals'] = weak_signals
    data['labels'] = labels
    data['test_features'] = test_features
    data['test_labels'] = test_labels
    data['has_test_data'] = has_test_data
    data['has_labels'] = has_test_data

    return data


def get_estimated_bounds(weak_probabilities):
    "Estimate the error and precision of the weak signals via matrix completion"

    m,n,k = weak_probabilities.shape
    weak_probabilities = np.rint(weak_probabilities)
    weak_signals = weak_probabilities.transpose((1,0,2))
    weak_signals = weak_signals.reshape(n,m*k)

    print(weak_signals)

    def error_estimation():
        "Code adapted from Snorkel tutorial"
        Ls = np.ones(weak_signals.shape)
        Ls[weak_signals==-1] =0
        Ls[weak_signals==0] =-1
        Z = np.dot(Ls.T, Ls) / (np.dot(np.abs(Ls).T, np.abs(Ls)))

        # Here we set up TF placeholder variables for Z and q
        z = tf.placeholder(tf.float32, Z.shape)
        q = tf.Variable(tf.random.normal([ Z.shape[0],1], mean=0.5, stddev=.15))

        # y = qq^T
        y = q * tf.transpose(q)

        # Here we just zero-out the diagonals, because we don't care about learning
        # them (they are always 1 since an LF will always agree with itself!)
        diag  = tf.zeros((Z.shape[0]))
        mask  = tf.ones((Z.shape))
        mask  = tf.linalg.set_diag(mask, diag)
        y_aug = tf.multiply(y, mask)
        z_aug = tf.multiply(z, mask)

        # Our loss function: sum((Z - qq^T)^2)
        lr = tf.constant(.001, name='learning_rate')
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
        loss = tf.reduce_sum((z_aug - y_aug) * (z_aug - y_aug)) #+ 0.0005*tf.reduce_sum(y_aug*y_aug)
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(1000):
            sess.run(train_step, feed_dict={z : Z})
        q_final  = sess.run(q)
        est_accs = (q_final+1)/2
        est_errors = 1 - np.ravel(est_accs).reshape(m,k)

        return est_errors

    error_bounds = error_estimation()
    precisions = np.ones((m,k)) * 0.5
    return error_bounds, precisions


def get_datapoints(labels, num_data_points, label_class):
    """
    Return indices for number of datapoints with specified class

    :param labels: True labels for the data
    :type labels: ndarray
    :param num_data_points: The number of datapoints to return
    :type num_data_points: int
    :param label_class: The class of the labels needed
    :type label_class: int
    :return: indices of the datapoints with specified class
    :rtype: ndarray
    """

    indices = np.where(labels == label_class)[0]
    return indices[:num_data_points]


def to_binary(labels, label_class):
    """
    Return indices for number of datapoints with specified class

    :param labels: True labels for the data
    :type labels: ndarray
    :param label_class: The class of the labels needed
    :type label_class: int
    :return: binary labels with the label class as ones
    :rtype: ndarray
    """
    binary_labels = np.zeros(labels.shape)
    binary_labels[labels == label_class] = 1
    return binary_labels

def sample_from_signal(weak_signal, sample_rate=0.2):
    "Uniformly sample from a weak signal"
    sample = np.random.random_sample(weak_signal.shape) * sample_rate
    sample[weak_signal==0] = 0.5 - sample[weak_signal==0]
    sample[weak_signal==1.0] = sample[weak_signal==1.0] + 0.5
    return sample


def weak_signal_average(weak_signals, num_weak_signals):
    average_labels = weak_signals[:num_weak_signals, :, :]
    average_labels = np.mean(average_labels, axis=0)
    return average_labels


def majority_vote_signal(weak_signals, num_weak_signals):
    baseline_weak_labels = weak_signals[:num_weak_signals, :, :]
    baseline_weak_labels = np.rint(baseline_weak_labels)
    mv_weak_labels = np.ones(baseline_weak_labels.shape)
    mv_weak_labels[baseline_weak_labels==-1] =0
    mv_weak_labels[baseline_weak_labels==0] =-1
    mv_weak_labels = np.sign(np.sum(mv_weak_labels, axis=0))
    break_ties = np.random.randint(2, size=int(np.sum(mv_weak_labels==0)))
    mv_weak_labels[mv_weak_labels==0] = break_ties
    mv_weak_labels[mv_weak_labels==-1] = 0
    return mv_weak_labels


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


def downsample_classes(labels, label_class):
    """
    Return indices for balanced classes

    :param labels: True labels for the data
    :type labels: ndarray
    :return: indices for major and minor classes
    :rtype: ndarray
    """
    # Separate majority and minority classes
    unique_labels = np.unique(labels)
    num_class = unique_labels.size
    labels_size = labels.size

    downsample_size = int(labels_size / (num_class * (num_class - 1)))
    balanced_indices = []

    # down_sample the other class
    for label in unique_labels:
        if label != label_class:
            sample = get_datapoints(labels, downsample_size, label)
            balanced_indices.append(sample)

    balanced_indices = np.asarray(balanced_indices).ravel()
    # add the class to the samples
    sample = get_datapoints(labels, int(labels_size / num_class), label_class)
    balanced_indices = np.concatenate([balanced_indices, sample])
    # balanced_indices = balanced_indices.astype(int)
    np.random.shuffle(balanced_indices)
    # Display indices for balanced classes
    return balanced_indices


def synthetic_experiment():
    np.random.seed(900)

    # n is the number of data points, and d is the dimension of the feature vector
    # that we represent each of them as
    n  = 20000
    d  = 200
    m = 10
    Ys = 2 * np.random.randint(2, size=(n,)) - 1
    # Ys = 2*np.random.choice(2,n,p=[0.7,0.3]) -1

    # We think of the binary features as functions each having some (unknown)
    # correlation with the target label, which we'll set randomly in [0.4,0.6]
    feature_accs = 0.2 * np.random.random((d,)) + 0.5
    Xs = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            if np.random.random() > feature_accs[j]:
                Xs[i,j] = 1 if Ys[i] == 1 else 0
            else:
                Xs[i,j] = 0 if Ys[i] == 1 else 1

    def dependent_signals(no_signals):
        # create dependencies between weak signals
        seed = 500
        np.random.seed(seed)
        lf_accs = 0.1 * np.random.random() + 0.5
        coverage = 0.3
        weak_signal = np.zeros(n)
        for i in range(n):
            if np.random.random() < coverage:
                weak_signal[i] = Ys[i] if np.random.random() < lf_accs else -Ys[i]

        signals = [weak_signal]
        # signals = [Ys]
        for i in range(no_signals-1):
            signal = weak_signal.copy()
            seed += 10
            np.random.seed(seed)
            indices = np.random.choice(n, int(n*0.2), replace=False)
            for j in indices:
                signal[j] = -1 * signal[j]
            signals.append(signal)

        return np.asarray(signals).T

    # Ls = dependent_signals(10)

    # Use this to initialize non-dependent weak signals
    lf_accs = 0.1 * np.random.random((m,)) + 0.6
    LF_COVERAGE = 0.3
    Ls = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if np.random.random() < LF_COVERAGE:
                Ls[i,j] = Ys[i] if np.random.random() < lf_accs[j] else -Ys[i]

    # Use this to initialize the weak signals for rank experiment
    # from numpy.linalg import matrix_rank
    # Ls = np.random.rand(n,m)
    # Ls = np.ones((n,m)) * np.random.rand(n).reshape(-1,1)
    # print("The rank of the weak signals: ", matrix_rank(Ls))
    # sys.exit(0)

    # Convert Y and weak_signals to correct format
    labels = 0.5 * (Ys + 1)
    weak_signals = Ls.copy()
    weak_signals[Ls==0] = -1
    weak_signals[Ls==-1] = 0

    n,m = weak_signals.shape
    weak_signals = weak_signals.T.reshape(m,n,1)

    data = {}
    data['data_features'] = Xs
    data['weak_signals'] = weak_signals
    data['labels'] = labels
    data['test_features'] = False
    data['test_labels'] = False
    data['has_labels'] = True
    data['has_test_data'] = False

    return data


