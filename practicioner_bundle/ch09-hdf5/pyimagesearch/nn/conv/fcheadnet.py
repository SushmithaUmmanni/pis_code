# -*- coding: utf-8 -*-
"""Network Surgery

Demonstrate how to compute rank-1 and rank-5 accuracy for a dataset.

Examples:
    $ python inspect_model.py

Attributes:
    baseModel (int, optional):
        Specify whether or not to include the top of a CNN.
    classes
    D
"""
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense


class FCHeadNet:
    """Replace the head of a pre-trained CNN with a custom head

    Returns:
        [type] -- network model with a
    """
    @staticmethod
    def build(baseModel, classes, D):
        """[summary]

        Arguments:
            baseModel {[type]} -- [description]
            classes {[type]} -- [description]
            D {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        # initialize the head model that will be placed on top of the base, then add a FC layer
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)
        # return the model
        return headModel
