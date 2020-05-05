# -*- coding: utf-8 -*-
"""Network Surgery

Demonstrate how to compute rank-1 and rank-5 accuracy for a dataset.

Examples:
    $ python inspect_model.py

Attributes:
    base_model (obj):
        Specify whether or not to include the top of a CNN.
    classes
    num_nodes (int)
        number of nodes in the fully-connected layer
"""
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense


class FCHeadNet:
    """Replace the head of a pre-trained CNN with a custom head
    """

    @staticmethod
    def build(base_model, classes, num_nodes):
        """Build the model

        Arguments:
            base_model {obj} -- Model without the head
            classes {int} -- number of output label classes in the dataset
            num_nodes {int} -- number of nodes in the fully connected layer

        Returns:
            keras model -- network model with a custom head
        """
        # initialize the head model that will be placed on top of the base, then add a FC layer
        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(num_nodes, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        # add a softmax layer
        head_model = Dense(classes, activation="softmax")(head_model)
        # return the model
        return head_model
