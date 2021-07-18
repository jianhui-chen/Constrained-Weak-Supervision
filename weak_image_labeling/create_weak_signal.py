import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from sklearn.metrics import roc_curve
from helper_functions import *
import json
import os


if not os.path.exists("weak_signals"):
	os.makedirs("weak_signals")

"""
Import image dataset
Change this to load whatever data we use
"""
data = load_fashion_mnist()
# data = load_cifar_10()
# data = load_svhn()
# data = load_handgestures()

images, labels = data['train_data']
image_width, image_height = data['img_rows'], data['img_cols']
categories = data['categories']
num_categories = data['num_classes']
channels = data['channels']

labels = np.array(labels, dtype=int)

if channels == 1:
	# convert images to RGB arrays
	images = [np.reshape(np.array(x) / 255, (image_width, image_height)) for x in images]
	images = [np.broadcast_to(x, (3, image_width, image_height)).transpose((1, 2, 0)) for x in images]
else:
	images = [np.reshape(np.array(x) / 255, (image_width, image_height, channels)) for x in images]

	# convert to grayscale
	# images = images.reshape(images.shape[0], image_width, image_height, channels)
	# images =  grayscale(images)
	# images = [np.reshape(np.array(x) / 255, (image_width, image_height)) for x in images]
	# images = [np.broadcast_to(x, (3, image_width, image_height)).transpose((1, 2, 0)) for x in images]

num_images = len(images)
plt.figure(figsize=(8, 6))

category_counts = np.zeros(num_categories)

while True:

	"""
	Ask user to choose category
	"""
	plt.clf()
	for i, category in enumerate(categories):
		plt.text(0, (num_categories - i) / num_categories, "%d. %s" % (i, category))
	plt.axis([0, 1, 0, 1.1])
	plt.axis("off")
	plt.title("Click on a category to create a rule for.")

	plt.tight_layout()

	click = plt.ginput(1, timeout=0)[0]
	category = np.int(num_categories - np.clip(click[1], a_min=0, a_max=num_categories / num_categories) * num_categories + 0.5)

	"""
	Ask user to choose a reference image
	"""
	num_samples = 50
	grid_rows = 5
	grid_cols = 10
	samples = np.random.choice(num_images, num_samples)

	tableau = create_tableau(images, samples, grid_rows, grid_cols, image_width)

	plt.clf()
	plt.imshow(tableau)
	plt.xlabel("Click to select an example and counter-example of %s." % categories[category])
	plt.tight_layout()

	points = plt.ginput(2, timeout=0)

	selected_index = [None] * 2

	for i in range(2):
		row = int(points[i][1] / image_height)
		col = int(points[i][0] / image_width)

		selected_index[i] = row * grid_cols + col

	reference_image = images[samples[selected_index[0]]]
	negative_image = images[samples[selected_index[1]]]

	"""
	Ask user to mark region of interest
	"""

	while True:
		mask = np.zeros((image_height, image_width), dtype=bool)

		plt.subplot(211)
		plt.imshow(tableau)
		plt.subplot(2, 2, 4)

		plt.imshow(negative_image)
		plt.xlabel("Compared to this? (Click on either image to mark regions)")

		plt.subplot(2, 2, 3)

		plt.imshow(reference_image)
		plt.title("What tells you this is %s?" % categories[category])
		plt.xlabel("Click to select regions of interest. Press enter to finish.")
		plt.tight_layout()
		clicks = plt.ginput(0, timeout=0)

		for click in clicks:
			i = int(click[1] + 0.5)
			j = int(click[0] + 0.5)

			mask[(i - 3):(i + 3),
				(j - 3):(j + 3)] = True

		modified_image = reference_image.copy()
		modified_image[:, :, 0] = 0.5 * modified_image[:, :, 0] + 0.5 * mask

		plt.subplot(212)
		plt.cla()
		plt.imshow(modified_image)
		plt.xlabel("Click to accept this mask. Press a key to retry.")
		if not plt.waitforbuttonpress():
			break

	"""
	Create classifier on reference image
	"""
	# def compare_images(a, b):
	# 	"""
	# 	:param a: rgb image
	# 	:param b: rgb image
	# 	:return: Euclidean distance of selected region
	# 	"""
	# 	patch_a = a.transpose((2, 0, 1)) * mask
	# 	patch_b = b.transpose((2, 0, 1)) * mask
	# 	return np.sum(np.abs(patch_a - patch_b))
	# 	# return np.linalg.norm(patch_a - patch_b)

	# scores = np.array([compare_images(a, reference_image) for a in images])
	# negative_scores = np.array([compare_images(a, negative_image) for a in images])

	# scores = scores - negative_scores

	diff_vector = (negative_image - reference_image).transpose((2, 0, 1))
	scores = np.array([np.sum(diff_vector * mask * image.transpose((2, 0, 1))) for image in images])

	# get 1/k percentile score
	threshold = np.percentile(scores, 5)  # int(100 / num_categories))

	# set logistic so 1/k percentile is decision boundary
	probabilities = 1 / (1 + np.exp(scores - threshold))

	# plt.clf()
	# plt.plot(np.sort(probabilities))
	# plt.waitforbuttonpress()

	# sample from probabilities
	predicted = np.random.rand(num_images) < probabilities
	predicted_positive = np.nonzero(predicted)[0]
	predicted_negative = np.nonzero(~predicted)[0]

	"""
	Ask user to estimate precision and false-positive rate
	"""
	plt.clf()

	# plot positive examples
	num_samples = 20
	samples = predicted_positive[np.random.choice(len(predicted_positive), num_samples)]
	tableau = create_tableau(images, samples, 2, 10, image_width)

	plt.clf()
	image_ax = plt.axes([0.15, 0.2, 0.7, 0.7])
	image_ax.imshow(tableau)

	plt.title("The categories are %s" % ", ".join(categories))
	plt.xlabel("Estimate the ratio of these positive examples that are %s." % categories[category])

	slider_axis = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor="gray")
	slider = Slider(slider_axis, 'Accuracy', 0.0, 1.0, valinit=(1 / num_categories), valstep=0.01)

	plt.axis("off")

	while True:
		action = plt.waitforbuttonpress()
		if action:
			break

	precision = slider.val

	# plot negative examples
	num_samples = 20
	samples = predicted_negative[np.random.choice(len(predicted_negative), num_samples)]
	tableau = create_tableau(images, samples, 2, 10, image_width)

	plt.clf()
	image_ax = plt.axes([0.15, 0.2, 0.7, 0.7])
	image_ax.imshow(tableau)

	plt.title("Set the slider and press space to continue.")
	plt.xlabel("Estimate the ratio of these negative examples that are %s." % categories[category])

	slider_axis = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor="gray")
	slider = Slider(slider_axis, 'Accuracy', 0.0, 1.0, valinit=(1 - 1 / num_categories), valstep=0.01)

	while True:
		action = plt.waitforbuttonpress()
		if action:
			break

	fnr = slider.val

	"""
	Calculate estimated accuracy from precision and TNR
	"""
	accuracy = precision * predicted.mean() + (1 - fnr) * (1 - predicted.mean())

	"""
	Show actual ROC of weak rule (only possible when we have ground truth)
	"""
	fpr, tpr, thresholds = roc_curve(labels == category, probabilities)

	plt.clf()
	plt.plot(fpr, tpr, label="Weak Rule")
	plt.plot([0, 1], [0, 1], label="Random")
	plt.legend()
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")

	print("Estimated precision %f, false negative rate %f, and accuracy %f." % (precision, fnr, accuracy))

	true_precision = np.sum(labels[predicted_positive] == category) / predicted_positive.size
	true_fnr = np.sum(labels[predicted_negative] == category) / predicted_negative.size
	true_accuracy = np.sum(predicted == (labels == category)) / len(labels)
	print("Actual precision %f, false negative rate %f, and accuracy %f" % (true_precision, true_fnr, true_accuracy))

	plt.waitforbuttonpress()

	"""
	Save weak signal to file
	"""
	to_save = dict()

	to_save["weak_signal"] = probabilities.tolist()
	to_save["reference_image_id"] = selected_index
	to_save["mask"] = mask.tolist()
	# to_save["width"] = int(width)
	# to_save["height"] = int(height)
	# to_save["threshold"] = threshold
	to_save["provided_precision"] = precision
	to_save["provided_fnr"] = fnr
	to_save["provide_accuracy"] = accuracy
	to_save["true_precision"] = true_precision
	to_save["true_fnr"] = true_fnr
	to_save["true_accuracy"] = true_accuracy

	filename = "%s_signal_%d.json" % (categories[category], category_counts[category])

	with open(os.path.join("weak_signals", filename), 'w') as outfile:
		json.dump(to_save, outfile)

	category_counts[category] += 1
