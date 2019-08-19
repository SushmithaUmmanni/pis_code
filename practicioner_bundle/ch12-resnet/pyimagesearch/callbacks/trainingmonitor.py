# -*- coding: utf-8 -*-
"""Class for Monitoring the Training Process.

Attributes:
    fig_path (str):
        path to the output plot used to visualize loss and accuracy over time.
    json_path (str, optional):
        path used to serialize the loss and accuracy values as a JSON file.
    start_at (int, optional):
        starting epoch that training is resumed at when using ctrl + c training.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    """Loggs our loss and accuracy to disk

    Arguments:
        BaseLogger {class} -- Keras BaseLogger class
    """
    def __init__(self, fig_path, json_path=None, start_at=0):
        """Initialize the training monitor

        Arguments:
            fig_path {str} -- The path to the output plot used to visualize loss and accuracy over
                              time.

        Keyword Arguments:
            json_path {str} -- The path used to serialize the loss and accuracy values as a JSON
                               file. (default: {None})
            start_at {int} -- The starting epoch that training is resumed at when using ctrl + c
                              training. (default: {0})
        """
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at
        # initialize the history dictionary
        self.history = {}

    def on_train_begin(self, logs=None):
        """Update all parameters in the logs

        Keyword Arguments:
            logs {dict} -- dictionary of hisotory logs
                           (default: {None)
        """
        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.history = json.loads(open(self.json_path).read())
                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and trim any entries that are
                    # past the starting epoch
                    for k in self.history.keys():
                        self.history[k] = self.history[k][:self.start_at]

    def on_epoch_end(self, epoch, logs=None):
        """Serialize the loss and accuracy for both the training and validation set to disk

        This functions automatically receives parameters from Keras and requires
        epoch and logs as parameters.

        Arguments:
            epoch {int} -- Epoch number

        Keyword Arguments:
            logs {dict} -- training and validation loss + accuracy for the current epoch
                           (default: {None})
        """
        if logs is None:
            logs = {}

        # loop over the logs and update the loss, accuracy, etc.for the entire training process
        for (key, value) in logs.items():
            log = self.history.get(key, [])
            log.append(float(value))
            self.history[key] = log

        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.history))
            f.close()

        # ensure at least two epochs have passed before plotting (epoch starts at zero)
        if len(self.history["loss"]) > 1:
            # plot the training loss and accuracy
            number_of_data_points = np.arange(0, len(self.history["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(number_of_data_points, self.history["loss"], label="train_loss")
            plt.plot(number_of_data_points, self.history["val_loss"], label="val_loss")
            plt.plot(number_of_data_points, self.history["acc"], label="train_acc")
            plt.plot(number_of_data_points, self.history["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.history["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.fig_path)
            plt.close()
