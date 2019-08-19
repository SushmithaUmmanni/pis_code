# -*- coding: utf-8 -*-
"""Implementation of DeeperGoogLeNet architecture.

This implementation is based on the original implemetation of GoogLeNet.
The authors of the net used BN before Activation layer. This should be switched.

Final Dropout of 0.5 should be applied instead of 0.4.
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
from keras.regularizers import l2
from keras import backend as K


class DeeperGoogLeNet:
    """Implementation of DeeperGoogLeNet architecture
    """
    @staticmethod
    def conv_module(x, filter_num, filter_x_size, filter_y_size, stride, chan_dim,
                    padding="same", reg=0.0005, name=None):
        """Define convolutional block

        Arguments:
            x {Tensor} -- input layer to the function
            filter_num {int} -- number of filters our CONV layer is going to learn
            filter_x_size {int} -- x-size of each of the filter_num filters that will be learned
            filter_y_size {int} -- y-size of each of the filter_num filters that will be learned
            stride {int} -- stride of the CONV layer
            chanel_dim {int} -- channel dimension, derived from “channels last” or “channels first”

        Keyword Arguments:
            padding {str} -- type of padding to be applied to the CONV layer (default: {"same"})
            reg {float} -- L2 weight decay strength (default: {0.0005})
            name {str} -- block name (default: {None})

        Returns:
            Tensor -- convolutional module
        """
        # initialize the CONV, BN, and RELU layer names
        (conv_name, bn_name, act_name) = (None, None, None)
        # if a layer name was supplied, prepend it
        if name is not None:
            conv_name = name + "_conv"
            bn_name = name + "_bn"
            act_name = name + "_act"
            # define a CONV => BN => RELU pattern
            x = Conv2D(filter_num, (filter_x_size, filter_y_size),
                       strides=stride,
                       padding=padding,
                       kernel_regularizer=l2(reg),
                       name=conv_name)(x)
            x = BatchNormalization(axis=chan_dim, name=bn_name)(x)
            x = Activation("relu", name=act_name)(x)
            # return the block
            return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3,   # pylint: disable=invalid-name
                         num5x5Reduce, num5x5, num1x1Proj,  # pylint: disable=invalid-name
                         chan_dim, stage, reg=0.0005):
        """Define Inception Module

        Args:
            x (Tensor): input lauer
            num1x1 (int): number of 1x1 filters
            num3x3Reduce (int): [description]
            num3x3 (int): number of 3x3 filters
            num5x5Reduce (int): [description]
            num5x5 (int): number of 5x5 filters
            num1x1Proj (int): [description]
            chan_dim (int): [description]
            stage (str): layer identifier
            reg (float, optional): streingth of the L2 regularization. Defaults to 0.0005.

        Returns:
            Tensor: Inception module
        """
        # define the first branch of the Inception module which consists of 1x1 convolutions
        first = DeeperGoogLeNet.conv_module(
            x, num1x1, 1, 1, (1, 1), chan_dim, reg=reg, name=stage + "_first")
        # define the second branch of the Inception module consisting of 1x1 and 3x3 convolutions
        second = DeeperGoogLeNet.conv_module(
            x, num3x3Reduce, 1, 1, (1, 1), chan_dim, reg=reg, name=stage + "_second1")
        second = DeeperGoogLeNet.conv_module(
            second, num3x3, 3, 3, (1, 1), chan_dim, reg=reg, name=stage + "_second2")
        # define the third branch of the Inception module which are our 1x1 and 5x5 convolutions
        third = DeeperGoogLeNet.conv_module(
            x, num5x5Reduce, 1, 1, (1, 1), chan_dim, reg=reg, name=stage + "_third1")
        third = DeeperGoogLeNet.conv_module(
            third, num5x5, 5, 5, (1, 1), chan_dim, reg=reg, name=stage + "_third2")
        # define the fourth branch of the Inception module which is the POOL projection
        fourth = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=stage + "_pool")(x)
        fourth = DeeperGoogLeNet.conv_module(
            fourth, num1x1Proj, 1, 1, (1, 1), chan_dim, reg=reg, name=stage + "_fourth")
        # concatenate across the channel dimension
        x = concatenate([first, second, third, fourth], axis=chan_dim, name=stage + "_mixed")
        # return the block
        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):
        """Build DeeperGoogLeNet

        Args:
            width (int): image width
            height (int): image weight
            depth (int): image depth
            classes (int): number of classes
            reg (float, optional): strength of the L2 regularization. Defaults to 0.0005.

        Returns:
            obj: DeeperGoogLeNet model
        """
        # initialize the input shape to be "channels last" and the channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # define the model input, followed by a sequence of
        # CONV => POOL => (CONV * 2) => POOL layers
        inputs = Input(shape=input_shape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), chan_dim, reg=reg, name="block1")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chan_dim, reg=reg, name="block2")
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chan_dim, reg=reg, name="block3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool2")(x)

        # apply two Inception modules followed by a POOL
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chan_dim, "3a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chan_dim, "3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        # apply five Inception modules followed by POOL
        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chan_dim, "4a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chan_dim, "4b", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chan_dim, "4c", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chan_dim, "4d", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128,
                                             chan_dim, "4e", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)

        # apply a POOL layer (average) followed by dropout
        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        # softmax classifier
        x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
