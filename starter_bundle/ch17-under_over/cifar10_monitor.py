# -*- coding: utf-8 -*-
"""Monitor the training process.

This script will show you how to babysit a training process using the TrainingMonitor callback.

Example:
    $ python cifar10_monitor.py --output output

Attributes:
    output (str):
        The path to the output directory to store our matplotlib generated figure and serialized
        JSON training history.
"""
import argparse
import os
import matplotlib
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from pyimagesearch.callbacks import TrainingMonitor

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


def main():
    """Train and evaluate MiniVGGNet using TrainigMonitor callback.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True, help="path to the output directory")
    args = vars(args.parse_args())

    # show information on the process ID
    print("[INFO process ID: {}".format(os.getpid()))

    # load the training and testing data, then scale it into the
    # range [0, 1]
    print("[INFO] loading CIFAR-10 data...")
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    train_x = train_x.astype("float") / 255.0
    test_x = test_x.astype("float") / 255.0
    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_y)
    test_y = label_binarizer.transform(test_y)
    # initialize the label names for the CIFAR-10 dataset
    label_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    # initialize the SGD optimizer, but without any learning rate decay
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # construct the set of callbacks
    fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])

    json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
    callbacks = [TrainingMonitor(fig_path, json_path=json_path)]
    # train the network
    print("[INFO] training network...")
    model.fit(train_x, train_y, validation_data=(test_x, test_y),
              batch_size=64, epochs=100, callbacks=callbacks, verbose=1)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=64)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=label_names))


if __name__ == '__main__':
    main()
