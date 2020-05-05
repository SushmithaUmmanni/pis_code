"""MXNet SqueezeNet architecture."""
# import the necessary packages
import mxnet as mx


class MxSqueezeNet:
    @staticmethod
    def squeeze(input, numFilter):
        # the first part of a FIRE module consists of a number of 1x1
        # filter squeezes on the input data followed by an activation
        squeeze_1x1 = mx.sym.Convolution(data=input, kernel=(1, 1),
        stride=(1, 1), num_filter=numFilter)
        act_1x1 = mx.sym.LeakyReLU(data=squeeze_1x1,
        act_type="elu")
        # return the activation for the squeeze
        return act_1x1

    @staticmethod
        def fire(input, numSqueezeFilter, numExpandFilter):

                # construct the 1x1 squeeze followed by the 1x1 expand
            squeeze_1x1 = MxSqueezeNet.squeeze(input, numSqueezeFilter)
            expand_1x1 = mx.sym.Convolution(data=squeeze_1x1,
                                            kernel=(1, 1), stride=(1, 1), num_filter=numExpandFilter)
            relu_expand_1x1 = mx.sym.LeakyReLU(data=expand_1x1,
                                            act_type="elu")
