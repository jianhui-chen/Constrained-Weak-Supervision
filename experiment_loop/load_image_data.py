import os, sys
import gzip
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.python.keras.datasets import cifar10
import scipy.io as sio

from image_utilities import get_image_supervision_data


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


def load_image_data():
    orig_data = load_fashion_mnist()

    new_data = get_image_supervision_data(orig_data, weak_signals='manual', true_bounds=False) # can try with pseudolabel

    image_data = dict()
    image_data['train'] = new_data['train_data']
    image_data['test'] = new_data['test_data']
    image_data['weak_signals'] = new_data['weak_model']['weak_signals']
    image_data['weak_errors'] = new_data['weak_model']['error_bounds']

    return image_data