# -*- coding: utf-8 -*-
"""Implementation of ShallowNet architecture.
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:
    """Implementation of ShallowNet architecture.

    Returns:
        obj -- ShallowNet model
    """
    @staticmethod
    def build(width, height, depth, classes):
        """Build ShallowNet model

        Arguments:
            width {int} -- The width of the input image.
            height {int} -- The height of the input image.
            depth {int} -- The number of channels (depth) of the image.
            classes {int} -- The number class labels in the classification task.

        Returns:
            obj -- ShallowNet model
        """
        # initialize the model along with the input shape to be "channels last"
        model = Sequential()
        input_shape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model
