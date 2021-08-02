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
            Name of the scalar value
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



"""
Not Part of Class
"""

def log_results(values, acc_logger, plot_path, title):
    """ 
        prints out results from the experiment

        Parameters
        ----------
        values: list of floats, size is 3 (same as current num algorithms)
            list of accuracies (between 0 and 1) to graph

        acc_logger: object of Logger class
            current logger object that will be used to write out to 
            tensor board

        plot_path: str 
            path to where matplotlib png will be stored

        title: str 
            name of current graph to be used as a tittle 

        Returns
        -------
        nothing
    """
    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # Check if it is a multi class example or one with abstaining signals
    # so there will be no Binary ALL present 
    if len(values) == 4:
        method_names = ['ALL', 'ALL 15,000','ALL 20,000', "ALL converg"]
        bar_colors = ['skyblue', 'saddlebrown', 'olivedrab', 'plum']
    elif len(values) == 3:
        method_names = ['BinaryALL', 'BinaryALL 15,000','BinaryALL 20,000']
        bar_colors = ['skyblue', 'saddlebrown', 'olivedrab']
    else:
        method_names = ['MultiALL','CLL', 'DataConsis']
        bar_colors = ['skyblue', 'saddlebrown', 'olivedrab']

    # add labels on graph 
    for i, v in enumerate(values):
        ax.text(i - 0.25, v + 0.01, str(round(v, 5)), color='seagreen', fontweight='bold')
    ax.bar(method_names, values, color=bar_colors)


    # set y demensions of plots
    min_value = min(values)
    max_value = max(values)
    if max_value + 0.1 > 1:
        plt.ylim([min_value - 0.1, 1])
    else:
        plt.ylim([min_value - 0.1, max_value + 0.1])

    # Save plot, then load into tensorboard
    plt.savefig(plot_path + "/plot.png", format='png')
    with acc_logger.writer.as_default():
        image = tf.io.read_file(plot_path + "/plot.png")
        image = tf.image.decode_png(image, channels=4)
        summary_op = tf.summary.image(title, [image], step=0)
        acc_logger.writer.flush()


