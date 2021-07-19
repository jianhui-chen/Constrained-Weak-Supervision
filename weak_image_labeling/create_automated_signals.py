import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from sklearn.metrics import roc_curve
from helper_functions import *
import os, json, time
from sklearn import preprocessing

if not os.path.exists("weak_signals"):
    os.makedirs("weak_signals")
"""
Import image dataset
Change this to load whatever data we use
"""
# data = load_fashion_mnist()
# data = load_cifar_10()
# data = load_svhn()
print("loading data...")
data = load_fashion_mnist()
#data = load_handgestures()
print("finished loading data..")

images, labels = data['train_data']
image_width, image_height = data['img_rows'], data['img_cols']
categories = data['categories']
num_categories = data['num_classes']
channels = data['channels']

labels = np.array(labels, dtype=int)

# preprocess data
images = images * (1/255)
# scaler = preprocessing.MinMaxScaler().fit(images)
# images = scaler.transform(images)

images = images.reshape(images.shape[0], image_width, image_height, channels)

if channels > 1:
    # convert to grayscale
    images = grayscale(images)

num_images = labels.size
category_counts = np.zeros(num_categories) + 5
signal_percategory = 5
total_signals = signal_percategory * num_categories
category_check = np.arange(num_categories)
category_counts[0] = 1
# category_counts[9] = 1

samples_indexes = [np.where(labels == i)[0] for i in range(num_categories)]

start = time.time()

# mask = np.ones((image_width, image_height), dtype=bool)
# mask[:, :10] = False
# mask[:, -20:] = False
rejected_samples = []

while np.sum(category_counts) < total_signals:
    """
	Use smart sampling
	"""
    check = (category_counts == signal_percategory)
    use_category = category_check[~check]
    category = np.random.choice(use_category)

    negative_category = category_check[category_check != category]
    negative_category = np.random.choice(negative_category)

    sample = np.random.choice(samples_indexes[category])
    negative_sample = np.random.choice(samples_indexes[negative_category])

    # avoid sampling a rejected sample
    while sample in rejected_samples:
        sample = np.random.choice(samples_indexes[category])

    reference_image = images[sample]
    negative_image = images[negative_sample]

    def compare_images(a, b):
        """
		:param a: grayscale image
		:param b: grayscale image
		:return: Euclidean distance of selected region
		"""
        # patch_a = a.transpose((2, 0, 1)) * mask
        # patch_b = b.transpose((2, 0, 1)) * mask
        patch_a = a
        patch_b = b
        # return np.sum(np.abs(patch_a - patch_b))
        return np.linalg.norm(patch_a - patch_b)

    scores = np.array([compare_images(a, reference_image) for a in images])
    negative_scores = np.array([compare_images(a, negative_image) for a in images])

    scores = scores - negative_scores

    # get 1/k percentile score
    threshold = np.percentile(scores, 5)

    # set logistic so 1/k percentile is decision boundary
    probabilities = 1 / (1 + np.exp(scores - threshold))

    # sample from probabilities
    # predicted = np.random.rand(num_images) < probabilities
    predicted = probabilities > 0.5
    predicted_positive = np.nonzero(predicted)[0]
    predicted_negative = np.nonzero(~predicted)[0]

    true_precision = np.sum(
        labels[predicted_positive] == category) / predicted_positive.size
    true_fnr = np.sum(
        labels[predicted_negative] == category) / predicted_negative.size
    true_accuracy = np.sum(predicted == (labels == category)) / len(labels)
    """
	Save weak signal to file
	"""
    if (true_precision >= 0.5):
        to_save = dict()

        print("Actual precision %f, false negative rate %f, and accuracy %f" %
              (true_precision, true_fnr, true_accuracy))
        to_save["weak_signal"] = probabilities.tolist()
        to_save["threshold"] = threshold
        to_save["provided_precision"] = 1
        to_save["provided_fnr"] = 0
        to_save["provide_accuracy"] = 0
        to_save["true_precision"] = true_precision
        to_save["true_fnr"] = true_fnr
        to_save["true_accuracy"] = true_accuracy

        filename = "%s_signal_%d.json" % (categories[category],
                                          category_counts[category])

        with open(os.path.join("weak_signals", filename), 'w') as outfile:
            json.dump(to_save, outfile)
        category_counts[category] += 1
    else:
        rejected_samples.append(sample)
