import numpy as np
import os
import gzip
import numpy as np
from sklearn.utils import resample
import keras
from keras.datasets import cifar10
import scipy.io as sio
from sklearn.model_selection import train_test_split
from PIL import Image

def create_tableau(images, samples, grid_rows, grid_cols, image_width):
	"""
	Tile images together to form a large image.
	:param images: list of images
	:param grid_rows: number of rows
	:param grid_cols: number of columns
	:return: large image of tiled sub-images
	"""
	pixel_width = grid_cols * image_width
	tableau = np.zeros((0, pixel_width, 3))
	k = 0
	for i in range(grid_rows):
		row = []
		for j in range(grid_cols):
			row += [images[samples[k]]]
			k += 1
		row = np.hstack(row)
		tableau = np.vstack((tableau, row))

	return tableau


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


def load_handgestures():

    lookup = dict()
    reverselookup = dict()
    count = 0
    path = '../../datasets/hand-gestures/leapgestrecog/leapGestRecog/'

    for j in os.listdir(path + '00/'):
        if not j.startswith(
                '.'):  # Ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1

    x_data = []
    y_data = []
    datacount = 0  # We'll use this to tally how many images are in our dataset
    for i in range(0, 10):  # Loop over the ten top-level folders
        for j in os.listdir(path + '0' + str(i) + '/'):
            if not j.startswith('.'):  # Again avoid hidden folders
                count = 0  # To tally images of a given gesture
                for k in os.listdir(path + '0' + str(i) + '/' + j + '/'):
                    # Loop over the images
                    img = Image.open(path + '0' + str(i) + '/' + j + '/' +
                                     k).convert('L')
                    # Read in and convert to greyscale
                    img = img.resize((160, 60))
                    arr = np.array(img)
                    x_data.append(arr)
                    count = count + 1
                y_values = np.full((count, 1), lookup[j])
                y_data.append(y_values)
                datacount = datacount + count
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1)  # Reshape to be the correct size
    x_data = x_data.reshape((datacount, 60, 160, 1))

    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        stratify=y_data,
                                                        test_size=0.3,
                                                        random_state=2000)

    data = {}
    categories = [
        "ok", "thumb", "fist_moved", "palm_moved", "palm", "c", "l", "fist",
        "down", "index"
    ]

    train_size, img_rows, img_cols, channels = x_train.shape
    test_size, img_rows, img_cols, channels = x_test.shape
    y_train, y_test = y_train.ravel(), y_test.ravel()

    data['img_rows'] = img_rows
    data['img_cols'] = img_cols
    data['channels'] = channels
    data['num_classes'] = 10
    data['categories'] = categories
    data['path'] = 'handgestures'

    x_train = x_train.reshape(train_size, img_rows * img_cols * channels)
    x_test = x_test.reshape(test_size, img_rows * img_cols * channels)

    data['train_data'] = x_train, y_train
    data['test_data'] = x_test, y_test

    return data


def load_cifar_10():
    # Returns dictionary of training and test data

    data = {}
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_size, img_rows, img_cols, channels = x_train.shape
    test_size, img_rows, img_cols, channels = x_test.shape
    y_train, y_test = y_train.ravel(), y_test.ravel()

    categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

    data['img_rows'] = img_rows
    data['img_cols'] = img_cols
    data['channels'] = channels
    data['num_classes'] = 10
    data['categories'] = categories

    x_train = x_train.reshape(train_size, img_rows*img_cols*channels)
    x_test = x_test.reshape(test_size, img_rows*img_cols*channels)

    data['train_data'] = x_train, y_train
    data['test_data'] = x_test, y_test

    return data


def load_fashion_mnist():
    # Returns dictionary of training and test data
    data = {}

    def load_data(path, kind='train'):
        """Load MNIST data from `path`"""

        # label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        #               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        # labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        # images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    categories = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt",
              "sneaker", "bag", "ankle boot"]

    # data['train_data'] = load_data('../../datasets/fashion-mnist', kind='train')
    # data['test_data'] = load_data('../../datasets/fashion-mnist', kind='t10k')
    data['train_data'] = load_data('../datasets/fashion-mnist', kind='train')
    data['test_data'] = load_data('../datasets/fashion-mnist', kind='t10k')
    data['img_rows'] = 28
    data['img_cols'] = 28
    data['channels'] = 1
    data['num_classes'] = 10
    data['categories'] = categories

    return data

def load_svhn():

    data = {}
    categories = ["zero", "one", "two", "three", "four", "five", "six",
              "seven", "eight", "nine"]

    def load_images(path):
        train_images = sio.loadmat(path+'/train_32x32.mat')
        test_images = sio.loadmat(path+'/test_32x32.mat')
        return train_images, test_images

    def normalize_images(images):
        imgs = images["X"]
        imgs = np.transpose(imgs, (3, 0, 1, 2))
        labels = images["y"]
        # replace label "10" with label "0"
        labels[labels == 10] = 0
        return imgs, labels

    train_images, test_images = load_images('../../datasets/svhn')
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

    x_train = train_images.reshape(train_size, img_rows*img_cols*channels)
    x_test = test_images.reshape(test_size, img_rows*img_cols*channels)

    data['train_data'] = x_train, y_train
    data['test_data'] = x_test, y_test

    return data
