""" 
file for logging calculations
    Currently focusing on Tensorboard logs, with add for normal logs
Borrowing code from Jeasine Ma
"""

import tensorflow as tf
# is it possible to do from tensorflow import Summary, summary
#from tensorflow.compat.v1 import summary
#from tensorflow.compat.v1 import Summary
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        tf.summary.scalar(tag, value, step=step)

        self.writer.flush()

        """    
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        """


def log_accuracy(logger, values, title):

    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    methods = ['ALL', 'Baseline', 'GE Criteria', 'Weak Signal']

    # add labels on graph 
    for i, v in enumerate(values):
        ax.text(i - 0.25, v + 0.01, str(round(v, 5)), color='seagreen', fontweight='bold')
    ax.bar(methods,values)

    # set y demensions of plots
    min_value = min(values)
    plt.ylim([min_value - 0.1, 1])

    plt.savefig("./logs/standard/plot.png", format='png')

    with logger.writer.as_default():
        image = tf.io.read_file("./logs/standard/plot.png")
        image = tf.image.decode_png(image, channels=4)
        summary_op = tf.summary.image(title, [image], step=0)
        logger.writer.flush()



