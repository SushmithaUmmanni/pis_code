# -*- coding: utf-8 -*-
"""Train custom neural network model on MNIST dataset.

Example:
    $ python nn_mnist.py
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from pyimagesearch.nn import NeuralNetwork


def main():
    """Train neural network implementation on the MNIST dataset.
    """
    # load the MNIST dataset and apply min/max scaling to scale the pixel intensity values
    # to the range [0, 1] (each image is represented by an 8 x 8 = 64-dim feature vector)
    print("[INFO] loading MNIST (sample) dataset...")
    digits = datasets.load_digits()
    data = digits.data.astype('float')  # pylint: disable=no-member
    data = (data - data.min()) / (data.max() - data.min())
    print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

    # construct the training and testing splits
    (train_x, test_x, train_y, test_y) = \
        train_test_split(data, digits.target, test_size=0.25)  # pylint: disable=no-member

    # convert the labels from integers to vectors using one-hot-encoding
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # train the network
    print("[INFO] training network...")
    network = NeuralNetwork([train_x.shape[1], 32, 16, 10])
    print("[INFO] {}".format(network))
    network.fit(train_x, train_y, epochs=1000)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = network.predict(test_x)
    predictions = predictions.argmax(axis=1)
    print(classification_report(test_y.argmax(axis=1), predictions))


if __name__ == '__main__':
    main()
