# -*- coding: utf-8 -*-
"""Model Serializer

Class for checkpointing the model at a certain epoch

Attributes:
    output_path (str):
        path to our output dir where we will store the weights
    every (int, optional):
        checkpoint interval, where the weight will be serialized
    start_at (int, optional):
        The starting epoch that training is resumed at when using ctrl + c training.
"""
import os
from keras.callbacks import Callback


class EpochCheckpoint(Callback):
    """Serialize weights at target epoch
    """

    def __init__(self, output_path, every=5, start_at=0):
        """Initialized model checkpointer

        Arguments:
            Callback {class} -- Keras Callback class
            output_path {str} --  path to our output dir where the model weight will be stored

        Keyword Arguments:
            every {int} -- checkpoint interval (default: {5})
            start_at {int} -- The starting epoch that training is resumed at when using ctrl + c
                              training. (default: {0})
        """
        # call the parent constructor
        super(EpochCheckpoint, self).__init__()
        # store the base output path for the model, the number of epochs that must pass before
        # the model is serialized to disk and the current epoch value
        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs=None):
        """Serialize the weights for both the training and validation set to disk

        todo: do we need logs?


        Arguments:
            epoch {int} -- [description]

        Keyword Arguments:
            logs {[type]} -- [description] (default: {None})
        """
        # check to see if the model should be serialized to disk
        if (self.int_epoch + 1) % self.every == 0:
            path = os.path.sep.join([self.output_path, "epoch_{}.hdf5".format(self.int_epoch + 1)])
            self.model.save(path, overwrite=True)

        # increment the internal epoch counter
        self.int_epoch += 1
