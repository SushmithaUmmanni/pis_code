# -*- coding: utf-8 -*-
"""Fit a perceptron model to the bitwise OR dataset.

Example:
    $ python perceptron_or.py
"""
import numpy as np
from pyimagesearch.nn import Perceptron


def main():
    """Run perceptron algorithm on OR dataset.
    """
    # construct the OR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    # define our perceptron and train it
    print("[INFO] training perceptron...")
    perceptron = Perceptron(X.shape[1], alpha=0.1)
    perceptron.fit(X, y, epochs=20)

    # now that our perceptron is trained we can evaluate it
    print("[INFO] testing perceptron...")
    # now that our network is trained, loop over the data points
    for (value, target) in zip(X, y):
        # make a prediction on the data point and display the result
        # to our console
        pred = perceptron.predict(value)
        print("[INFO] data={}, ground-truth={}, pred={}".format(
            value, target[0], pred))


if __name__ == '__main__':
    main()
