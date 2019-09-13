# import the necessary packages
import mxnet as mx


class MxGoogLeNet:
    @staticmethod
    def conv_module(data, K, kX, kY, pad=(0, 0), stride=(1, 1)):
        # define the CONV => BN => RELU pattern
        conv = mx.sym.Convolution(data=data, kernel=(kX, kY), num_filter=K, pad=pad, stride=stride)
        bn = mx.sym.BatchNorm(data=conv)
        act = mx.sym.Activation(data=bn, act_type="relu")
        # return the block
        return act

    @staticmethod
    def inception_module(data, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj):
        # the first branch of the Inception module consists of 1x1 convolutions
        conv_1x1 = MxGoogLeNet.conv_module(data, num1x1, 1, 1)
        # the second branch of the Inception module is a set of 1x1
        # convolutions followed by 3x3 convolutions
        conv_r3x3 = MxGoogLeNet.conv_module(data, num3x3Reduce, 1, 1)
        conv_3x3 = MxGoogLeNet.conv_module(conv_r3x3, num3x3, 3, 3,pad=(1, 1))
        # the third branch of the Inception module is a set of 1x1
        # convolutions followed by 5x5 convolutions
        conv_r5x5 = MxGoogLeNet.conv_module(data, num5x5Reduce, 1, 1)
        conv_5x5 = MxGoogLeNet.conv_module(conv_r5x5, num5x5, 5, 5, pad=(2, 2))
        # the final branch of the Inception module is the POOL + projection layer set
        pool = mx.sym.Pooling(data=data, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        conv_proj = MxGoogLeNet.conv_module(pool, num1x1Proj, 1, 1)
        # concatenate the filters across the channel dimension
        concat = mx.sym.Concat(*[conv_1x1, conv_3x3, conv_5x5, conv_proj])
        # return the block
        return concat

    @staticmethod
    def build(classes):
        # data input
        data = mx.sym.Variable("data")
        # Block #1: CONV => POOL => CONV => CONV => POOL
        conv1_1 = MxGoogLeNet.conv_module(data, 64, 7, 7,pad=(3, 3), stride=(2, 2))
        pool1 = mx.sym.Pooling(data=conv1_1, pool_type="max", pad=(1, 1),
                               kernel=(3, 3), stride=(2, 2))
        conv1_2 = MxGoogLeNet.conv_module(pool1, 64, 1, 1)
        conv1_3 = MxGoogLeNet.conv_module(conv1_2, 192, 3, 3, pad=(1, 1))
        pool2 = mx.sym.Pooling(data=conv1_3, pool_type="max", pad=(1, 1),
                               kernel=(3, 3), stride=(2, 2))
