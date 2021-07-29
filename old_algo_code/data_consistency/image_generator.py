import os
import sys
import gzip
import gc
import pickle
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import scipy.io as sio
from sklearn.utils import shuffle
import tensorflow as tf
from scipy import spatial
import ssl



def load_svhn():
    data = {}
    datapath = '../../datasets/svhn'

    def load_images(path):
        train_images = sio.loadmat(path + '/train_32x32.mat')
        test_images = sio.loadmat(path + '/test_32x32.mat')
        return train_images, test_images

    def transform_images(images):
        imgs = images["X"]
        imgs = np.transpose(imgs, (3, 0, 1, 2))
        labels = images["y"]
        # replace label "10" with label "0"
        labels[labels == 10] = 0
        return imgs, labels

    train_images, test_images = load_images(datapath)
    train_images, train_labels = transform_images(train_images)
    test_images, test_labels = transform_images(test_images)

    data['train_data'] = train_images, np.ravel(train_labels)
    data['test_data'] = test_images, np.ravel(test_labels)
    data['datapath'] = datapath

    return data


def load_cifar_10():
    # Returns dictionary of training and test data
    data = {}
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = y_train.ravel(), y_test.ravel()
    data['train_data'] = x_train, y_train
    data['test_data'] = x_test, y_test
    data['datapath'] = '../../datasets/cifar10'

    return data


def load_fashion_mnist():
    # Returns dictionary of training and test data
    data = {}
    datapath = '../datasets/fashion-mnist'

    def load_data(path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)
        return images, labels

    images, labels = load_data(datapath, kind='train')
    test_images, test_labels = load_data(datapath, kind='t10k')

    # Convert the training and test images into 3 channels
    images = np.dstack([images]*3)
    test_images = np.dstack([test_images]*3)
    images = images.reshape(images.shape[0], 28, 28, 3)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 3)

    data['train_data'] = images, labels
    data['test_data'] = test_images, test_labels
    data['datapath'] = datapath
    return data


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


def cosine_similarity(A, b):
    numerator = np.dot(A, b)
    a = norm(A, axis=1, ord=2)
    b = norm(b)
    cosine_similarity = numerator/(a*b)
    cosine_similarity[cosine_similarity < 0] = 0
    cosine_similarity[cosine_similarity > 1] = 1
    return cosine_similarity


def euclidean_similarity(A, b):
    sim = norm(A - b, axis=1)
    sim = sim / np.max(sim)
    sim[sim > 1] = 1
    return 1 - sim


def nearest_neighbors(reference_data, data, num_classes=10):
    # Reference data is arranged in order

    first_signal = []
    second_signal = []
    i = 0

    while i < num_classes*2:
        signal_1 = cosine_similarity(data, reference_data[i])
        signal_2 = cosine_similarity(data, reference_data[i+1])
        first_signal.append(signal_1)
        second_signal.append(signal_2)
        i += 2

    weak_signals = np.asarray([first_signal, second_signal]).transpose(0, 2, 1)
    return weak_signals


def create_image_features(data, datapath, num_classes=10):

    train_data, train_labels = data['train_data']
    test_data, test_labels = data['test_data']
    img_size = 224

    img_shape = train_data.shape[1:]

    def get_pretrained_features(model, data):

        batch_size = 256
        data_features = []
        j = 0
        n = data.shape[0]
        while j < n:
            features = np.asarray([img_to_array(array_to_img(im, scale=False).resize(
                (img_size, img_size))) for im in data[j:j+batch_size]])
            # Preprocessing the input
            features = preprocess_input(features)
            # predict features
            features = model.predict(
                features, batch_size=batch_size, verbose=1)
            m, img_width, img_height, channels = features.shape
            features = features.reshape(m, img_width*img_height*channels)
            if j > 0:
                encoded_data = np.append(encoded_data, features, axis=0)
            else:
                encoded_data = features

            j += batch_size

        return encoded_data

    # def create_signal_data():
    #     signal_data = []
    #     for i in range(num_classes):
    #         indices = np.where(train_labels == i)[0]
    #         np.random.seed(200)
    #         index = np.random.choice(indices, 1)[0]
    #         image = train_data[index]
    #         flipped = tf.image.flip_left_right(image)
    #         bright = tf.image.random_brightness(image, 0.2)
    #         contrast = tf.image.random_contrast(image, 0.4, 0.7)

    #         signal_data.append([image, flipped, bright, contrast])

    #     return np.asarray(signal_data).transpose(1, 0, 2, 3, 4)

    # def create_weak_signals(model, signals, data):
    #     weak_signals = []
    #     for sig_data in signals:
    #         similarities = []
    #         signal = get_pretrained_features(model, sig_data)
    #         for class_signal in signal:
    #             similarity = euclidean_similarity(data, class_signal)
    #             similarities.append(similarity)
    #         weak_signals.append(similarities)

    #     return np.asarray(weak_signals).transpose(0, 2, 1)

    # signal_data = create_signal_data()

    # pre-trained model
    ssl._create_default_https_context = ssl._create_unverified_context

    model = VGG19(weights='imagenet', include_top=False,
                  input_shape=(img_size, img_size, 3), classes=10)
    train_data = get_pretrained_features(model, train_data)
    test_data = get_pretrained_features(model, test_data)

    # weak_signals = create_weak_signals(model, signal_data, train_data)

    K.clear_session()
    del model
    gc.collect()

    filename = datapath+'/embedding_data.pickle'
    output = {}
    output['train'] = train_data, tf.one_hot(train_labels, num_classes)
    output['test'] = test_data, tf.one_hot(test_labels, num_classes)
    # output['weak_signals'] = weak_signals
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output


def train_weak_signals(data, datapath, classes=[0, 1]):
    train_data, train_labels = data['train']
    test_data, test_labels = data['test']
    train_labels = np.argmax(train_labels.numpy(), axis=1)
    test_labels = np.argmax(test_labels.numpy(), axis=1)
    num_classes = 2
    weak_data = []
    weak_indices = []
    binary_indices = []
    test_indices = []

    for i in classes:
        # for i in range(num_classes):
        indices = np.where(train_labels == i)[0]
        np.random.seed(200)
        index = np.random.choice(indices, 2)
        images = train_data[index]
        weak_data.append(images[0])
        weak_data.append(images[1])
        weak_indices.extend(index)
        binary_indices.extend(indices)
        indices = np.where(test_labels == i)[0]
        test_indices.extend(indices)

    weak_labels = train_labels[weak_indices]
    weak_data = np.asarray(weak_data)

    # for binary data
    binary_indices = shuffle(binary_indices, random_state=200)
    train_data = train_data[binary_indices]
    train_labels = train_labels[binary_indices]

    # for multiclass data
    # train_data = np.delete(train_data, weak_indices, axis=0)
    # train_labels = np.delete(train_labels, weak_indices)
    np.random.seed(200)
    indices = np.random.choice(train_data.shape[0], 1000, replace=False)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    np.random.seed(200)
    indices = np.random.choice(test_indices, 500, replace=False)
    test_data = test_data[indices]
    test_labels = test_labels[indices]

    # neareast neighbor signals
    nn_signals = nearest_neighbors(weak_data, train_data, num_classes)
    reshaped_signals = np.round(nn_signals)

    weak_signals = []
    for signal in reshaped_signals:
        weak_signals.append(1 - signal.T[0])
        weak_signals.append(signal.T[1])
    weak_signals = np.expand_dims(weak_signals, axis=-1)

    output = {}
    output['train'] = train_data, train_labels
    output['test'] = test_data, test_labels
    output['weak_signals'] = weak_signals

    np.save(datapath+'/binary_data.npy', output)
    return output


def load_image_signals(datapath):
    filename = datapath+'/embedding_data.pickle'
    try:
        with open(filename, 'rb') as fp:
            results = pickle.load(fp)
    except:
        if 'svhn' in datapath:
            data = load_svhn()
        elif 'cifar' in datapath:
            data = load_cifar_10()
        else:
            data = load_fashion_mnist()
        results = create_image_features(data, datapath)

    try:
        results = np.load(datapath+'/binary_data.npy', allow_pickle=True)[()]
    except:
        results = train_weak_signals(results, datapath)
    return results


if __name__ == '__main__':
    data = load_image_signals('../../datasets/fashion-mnist')
    pass
