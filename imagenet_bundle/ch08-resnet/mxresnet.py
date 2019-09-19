# -*- coding: utf-8 -*-
"""Implementation of ResNet architecture with mxnet.
"""
import mxnet as mx


class MxResNet:
    """Implementation of ResNet architecture with mxnet
    """
    # uses "bottleneck" module with pre-activation (He et al. 2016)
    @staticmethod
    def residual_module(data, filter_num, stride, reduce=False, bn_eps=2e-5, bn_momentum=0.9):
        """Define redisual module

        Arguments:
            data {tensor} -- input to the residual model
            filter_num {int} -- number of filters that will be learned by the final CONV
                                in the bottleneck
            stride {int} -- stride of the convolution

        Keyword Arguments:
            reduce {bool} -- control whether we are reducing spatial dimensions (default: {False})
            bn_eps {float} -- epsilon for avoiding "division by zero" errors when
                              normalizing inputs (default: {2e-5})
            bn_momentum {float} -- momentum value for the moving average (default: {0.9})
        """
        # the shortcut branch of the ResNet module should be
        # initialized as the input (identity) data
        shortcut = data
        # the first block: BN => RELU => CONV of the ResNet module are 1x1 CONVs
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act1 = mx.sym.Activation(data=bn1, act_type="relu")
        conv1 = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                   num_filter=int(filter_num * 0.25), no_bias=True)

        # the second block of the ResNet module are 3x3 CONVs
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act2 = mx.sym.Activation(data=bn2, act_type="relu")
        conv2 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=stride,
                                   num_filter=int(filter_num * 0.25), no_bias=True)

        # the third block of the ResNet module is another set of 1x1 # CONVs
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act3 = mx.sym.Activation(data=bn3, act_type="relu")
        conv3 = mx.sym.Convolution(data=act3, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                   num_filter=filter_num, no_bias=True)

        # this check allows us to apply the insights from Springenberg et al.
        # if we are to reduce the spatial size, apply a CONV layer to the shortcut.
        if reduce:
            shortcut = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=stride,
                                          num_filter=filter_num, no_bias=True)
        # add together the shortcut and the final CONV
        add = conv3 + shortcut
        # return the addition as the output of the ResNet module
        return add

    @staticmethod
    def build(classes, stages, filters, bn_eps=2e-5, bn_momentum=0.9):
        """Build ResNet

        Arguments:
            classes {int} -- number of classes
            stages {list} -- numbers of residual modules that will be stacked on top of each other
                             per stage
            filters {list} -- number of filters that the CONV layers will learn per stage.

        Keyword Arguments:
            bn_eps {float} -- controls the epsilon value responsible for avoiding “division
                              by zero” errors when normalizing inputs. (default: {2e-5})
            bn_momentum {float} -- momentum value for the moving average normalizing
                                   inputs. (default: {0.9})

        Returns:
            obj -- ResNet model
        """
        # data input
        data = mx.sym.Variable("data")

        # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
        bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bn_eps, momentum=bn_momentum)
        conv1_1 = mx.sym.Convolution(data=bn1_1, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                     num_filter=filters[0], no_bias=True)
        bn1_2 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act1_2 = mx.sym.Activation(data=bn1_2, act_type="relu")
        pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1),
                               kernel=(3, 3), stride=(2, 2))
        body = pool1

        # loop over the number of stages
        for i in range(0, len(stages)):  # pylint: disable=consider-using-enumerate
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            body = MxResNet.residual_module(body, filters[i + 1], stride, reduce=True,
                                            bn_eps=bn_eps, bn_momentum=bn_momentum)
            # loop over the number of layers in the stage
            for _ in range(0, stages[i] - 1):
                # apply a ResNet module
                body = MxResNet.residual_module(body, filters[i + 1], (1, 1),
                                                bn_eps=bn_eps, bn_momentum=bn_momentum)

        # apply BN => ACT => POOL
        bn2_1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act2_1 = mx.sym.Activation(data=bn2_1, act_type="relu")
        pool2 = mx.sym.Pooling(data=act2_1, pool_type="avg", global_pool=True, kernel=(7, 7))

        # softmax classifier
        flatten = mx.sym.Flatten(data=pool2)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc1, name="softmax")

        # return the network architecture
        return model
