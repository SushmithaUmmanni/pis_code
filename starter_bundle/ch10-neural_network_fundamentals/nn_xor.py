# -*- coding: utf-8 -*-
"""Fit neural network model to the bitwise XOR dataset.

Example:
    $ python nn_xor.py
"""
import numpy as np
from pyimagesearch.nn import NeuralNetwork


def main():
    """Run neural network on XOR dataset.
    """
    # construct the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # define our 2-2-1 neural network and train it
    network = NeuralNetwork([2, 2, 1], alpha=0.5)
    network.fit(X, y, epochs=20000)

    # now that our network is trained, loop over the XOR data points
    for (value, target) in zip(X, y):
        # make a prediction on the data point and display the result
        # to our console
        pred = network.predict(value)[0][0]
        step = 1 if pred > 0.5 else 0
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
            value, target[0], pred, step))


if __name__ == '__main__':
    main()
