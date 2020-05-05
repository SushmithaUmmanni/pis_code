# -*- coding: utf-8 -*-
"""Model Serializer."""
import os
from keras.callbacks import Callback


class EpochCheckpoint(Callback):
    """
    Module for serializing weights at target epoch.

    Attributes:
        output_path (str): Path to our output dir where we will store the weights.
        every (int): Checkpoint interval, where the weight will be serialized.
        int_epoch (int): Starting epoch that training is resumed at when using ctrl + c training.
    """

    def __init__(self, output_path, every=5, start_at=0):
        """
        Initialized model checkpointer.

        Args:
            output_path (str): Path to our output dir where we will store the weights.
            every (int, optional): Checkpoint interval, where the weight will be serialized.
            start_at (int, optional): Starting epoch that training is resumed at when using
                                      ctrl + c training.
        """
        # call the parent constructor
        super(EpochCheckpoint, self).__init__()
        # store the base output path for the model, the number of epochs that must pass before
        # the model is serialized to disk and the current epoch value
        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs=None):
        """
        Serialize the weights for both the training and validation set to disk.

        This functions automatically receives parameters from Keras and requires
        epoch and logs as parameters.

        Args:
            epoch (int): Target epoch number.
            logs (dict, optional): Training and validation loss + accuracy for the current epoch.
        """
        # check to see if the model should be serialized to disk
        if (self.int_epoch + 1) % self.every == 0:
            path = os.path.sep.join([self.output_path, "epoch_{}.hdf5".format(self.int_epoch + 1)])
            self.model.save(path, overwrite=True)
        # increment the internal epoch counter
        self.int_epoch += 1
