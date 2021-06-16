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


def log_accuracy(data_obj, num_weak_signals, adversarial_models, weak_models):


    # Init loggers 
    learned_logger = Logger("logs/standard/" + data_obj.n + "/learned_model")
    weak_logger = Logger("logs/standard/" + data_obj.n + "/weak_signal(s)")
    Basline_logger = Logger("logs/standard/" + data_obj.n + "/baseline_models")
    ge_logger = Logger("logs/standard/" + data_obj.n + "/ge_criteria")
     

    for x in range(0, num_weak_signals):

        with learned_logger.writer.as_default():
            learned_logger.log_scalar("Accuracy of on validation data", adversarial_models[x]['validation_accuracy'], x)
            learned_logger.log_scalar("Accuracy of on test data", adversarial_models[x]['test_accuracy'], x)
        

        with weak_logger.writer.as_default():
            weak_logger.log_scalar("Accuracy of on validation data", weak_models[x]['validation_accuracy'][x], x)
            weak_logger.log_scalar("Accuracy of on test data", weak_models[x]['test_accuracy'][x], x)
        
        with Basline_logger.writer.as_default():
            Basline_logger.log_scalar("Accuracy of on validation data", weak_models[x]['baseline_val_accuracy'][0], x)
            Basline_logger.log_scalar("Accuracy of on test data", weak_models[x]['baseline_test_accuracy'][0], x)

        with ge_logger.writer.as_default():
            ge_logger.log_scalar("Accuracy of on validation data", weak_models[x]['gecriteria_val_accuracy'], x)
            ge_logger.log_scalar("Accuracy of on test data",  weak_models[x]['gecriteria_test_accuracy'], x)       

# def log_accuracy(self, to_map_val , to_map_test, step):


#     # Adversarial Label Learning
#     tf.summary.scalar("Accuracy of on validation data", to_map_val, step=step)
#     tf.summary.scalar("Accuracy of on test data", to_map_test, step=step)
#     self.writer.flush()
