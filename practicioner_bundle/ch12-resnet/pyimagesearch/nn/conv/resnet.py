# -*- coding: utf-8 -*-
"""Implementation of ResNet architecture.
"""
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


class ResNet:
    """Implementation of ResNet architecture
    """
    @staticmethod
    def residual_module(data, filter_num, stride, chan_dim, reduce=False,
                        regression=0.0001, bn_eps=2e-5, bn_momentum=0.9):
        """Define residual module

        bn_eps and bn_momentum are used to overwrite default Keras parameters

        Arguments:
            data {tensor} -- input to the residual module
            filter_num {int} -- number of filters that will be learned by
                                the final CONV in the bottleneck
            stride {int} -- stride of the convolution
            chan_dim {[type]} -- axis which will perform batch normalization

        Keyword Arguments:
            reduce {bool} -- control whether we are reducing spatial dimensions (default: {False})
            regression {float} -- regularization strength to all CONV layers in the residual module
                                  (default: {0.0001})
            bn_eps {float} -- epsilon for avoiding “division by zero” errors when normalizing
                              inputs. (default: {2e-5})
            bn_momentum {float} -- momentum value for the moving average (default: {0.9})
        """
        # the shortcut branch of the ResNet module should be initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(filter_num * 0.25),
                       (1, 1),
                       use_bias=False,
                       kernel_regularizer=l2(regression))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(filter_num * 0.25),
                       (3, 3),
                       strides=stride,
                       padding="same",
                       use_bias=False,
                       kernel_regularizer=l2(regression))(act2)

        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(filter_num, (1, 1), use_bias=False, kernel_regularizer=l2(regression))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if reduce:
            shortcut = Conv2D(filter_num,
                              (1, 1),
                              strides=stride,
                              use_bias=False,
                              kernel_regularizer=l2(regression))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bn_eps=2e-5, bn_momentum=0.9, dataset="cifar"):
        """Build ResNet

        In between each stage, we will apply an additional residual module to decrease the volume
        size. The first CONV layer (before any residual model is applied) will learn filters[0].
        Thus len(filters) = len(stages) + 1.

        Args:
            width (int): image width
            height (int): image heigth
            depth (int): image depth
            classes (int): number of classes
            stages (list): numbers of residual modules that will be stacked on top of each other
                           per stage
            filters (list): number of filters that the CONV layers will learn per stage.
            reg (float, optional): L2 regularization strength. Defaults to 0.0001.
            bn_eps (float, optional): controls the epsilon value responsible for avoiding “division
                                      by zero” errors when normalizing inputs. Defaults to 2e-5
            bn_momentum (float, optional): momentum value for the moving average normalizing
                                           inputs.. Defaults to 0.9.
            dataset (str, optional): Name of the dataset. Defaults to "cifar".

        Returns:
            obj: ResNet model
        """
        # initialize the input shape to be "channels last" and the channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # set the input and apply BN
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(inputs)
        # check if we are utilizing the CIFAR dataset
        if dataset == "cifar":
            # apply a single CONV layer
            x = Conv2D(filters[0],
                       (3, 3),
                       use_bias=False,
                       padding="same",
                       kernel_regularizer=l2(reg))(x)

        # check to see if we are using the Tiny ImageNet dataset
        elif dataset == "tiny_imagenet":
            # apply CONV => BN => ACT => POOL to reduce spatial size
            x = Conv2D(filters[0],
                       (5, 5),
                       use_bias=False,
                       padding="same",
                       kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):  # pylint: disable=consider-using-enumerate
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1],
                                       stride,
                                       chan_dim,
                                       reduce=True,
                                       bn_eps=bn_eps,
                                       bn_momentum=bn_momentum)

            # loop over the number of layers in the stage
            for _ in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1),
                                           chan_dim,
                                           bn_eps=bn_eps,
                                           bn_momentum=bn_momentum)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model
