# -*- coding: utf-8 -*-
"""Implementation of MiniGoogLeNet architecture.

This implementation is based on the original implemetation of GoogLeNet.
The authors of the net used BN before Activation layer.
This should be switched.
"""
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K


class MiniGoogLeNet:
    """Implementation of MiniGoogLeNet architecture
    """

    @staticmethod
    def conv_module(x, filter_num, filter_x_size, filter_y_size, stride, chanel_dim, padding="same"):
        """Define conv layer

        Arguments:
            x {Tensor} -- input layer to the function
            filter_num {int} -- number of filters our CONV layer is going to learn
            filter_x_size {int} -- x-size of each of the filter_num filters that will be learned
            filter_y_size {int} -- y-size of each of the filter_num filters that will be learned
            stride {int} -- stride of the CONV layer
            chanel_dim {int} -- channel dimension, derived from “channels last” or “channels first”

        Keyword Arguments:
            padding {str} -- type of padding to be applied to the CONV layer (default: {"same"})

        Returns:
            Tensor -- convolutional module
        """
        # define a CONV => BN => RELU pattern
        x = Conv2D(filter_num, (filter_x_size, filter_y_size), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanel_dim)(x)
        x = Activation("relu")(x)
        # return the block
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanel_dim):  # pylint: disable=invalid-name
        """Define inception module

        Arguments:
            x {Tensor} -- input layer
            numK1x1 {int} -- number of 1x1 filters
            numK3x3 {int} -- number of 3x3 filters
            chanel_dim {int} -- channel dimension, derived from “channels last” or “channels first”

        Returns:
            Tensor -- inception module
        """
        # define two CONV modules, then concatenate across the channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanel_dim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanel_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanel_dim)
        # return the block
        return x

    @staticmethod
    def downsample_module(x, filter_num, chanel_dim):
        """Define downsample module

        Arguments:
            x {Tensor} -- input layer
            filter_num {int} -- number of filters our CONV layer is going to learn
            chanel_dim {int} -- channel dimension, derived from “channels last” or “channels first”

        Returns:
            Tensor -- downsample module
        """
        # define the CONV module and POOL, then concatenate across the channel dimensions
        conv_3x3 = MiniGoogLeNet.conv_module(x, filter_num, 3, 3, (2, 2), chanel_dim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanel_dim)
        # return the block
        return x

    @staticmethod
    def build(width, height, depth, classes):
        """Build MiniGoogLeNet architecture

        Arguments:
            width {int} -- [description]
            height {int} -- [description]
            depth {int} -- [description]
            classes {int} -- [description]

        Returns:
            obj -- MiniGoogLeNet model
        """
        # initialize the input shape to be "channels last" and the channels dimension itself
        input_shape = (height, width, depth)
        chanel_dim = -1

        # if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanel_dim = 1

        # define the model input and first CONV module
        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanel_dim)

        # two Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanel_dim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanel_dim)
        x = MiniGoogLeNet.downsample_module(x, 80, chanel_dim)

        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanel_dim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanel_dim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanel_dim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanel_dim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanel_dim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanel_dim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanel_dim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
