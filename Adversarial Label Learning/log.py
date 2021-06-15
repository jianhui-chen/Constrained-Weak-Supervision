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


    
    def log_accuracy(self, num_weak_signals, adversarial_models, weak_models):

        print("\n\n in log_accuracy")

        for x in range(0, num_weak_signals):

            # Adversarial Label Learning
            tf.summary.scalar("Accuracy of learned model on the validatiion data", adversarial_models[x]['validation_accuracy'], step=x)
            tf.summary.scalar("Accuracy of learned model on the test data", adversarial_models[x]['test_accuracy'], step=x)
            tf.summary.scalar("Accuracy of weak signal(s) on the validation data", weak_models[x]['validation_accuracy'][x], step=x)
            tf.summary.scalar("Accuracy of accuracy of weak signal(s) on the test data", weak_models[x]['test_accuracy'][x], step=x)

            # baseline 
            tf.summary.scalar("Accuracy of the baseline models on the validation data", weak_models[x]['baseline_val_accuracy'][0], step=x)
            tf.summary.scalar("Accuracy of the baseline models on the test data", weak_models[x]['baseline_test_accuracy'][0], step=x)

            # GE criteria
            tf.summary.scalar("Accuracy of ge criteria on the validtion data", weak_models[x]['gecriteria_val_accuracy'], step=x)
            tf.summary.scalar("Accuracy of ge criteria on the test data", weak_models[x]['gecriteria_test_accuracy'], step=x)
            self.writer.flush()



        print(" Leaving log_accuracy \n\n")