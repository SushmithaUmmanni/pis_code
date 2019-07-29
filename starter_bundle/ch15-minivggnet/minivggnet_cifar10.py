# -*- coding: utf-8 -*-
"""Train and evaluate MiniVGGNet on Cifar10 dataset.

1. Load the CIFAR-10 dataset from disk.
2. Instantiate the MiniVGGNet architecture.
3. Train MiniVGGNet using the training data.
4. Evaluate network performance with the testing data.

Example:
    $ python minivggnet_cifar10.py --output output/cifar10_minivggnet_without_bn.png

Attributes:
    output (str):
        The path to our output training and loss plot
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


def main():
    """Train and evaluate MiniVGGNet on Cifar10 dataset.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True,
                      help="path to the output loss/accuracy plot")
    args = vars(args.parse_args())

    # load the training and testing data, then scale it into the range [0, 1]
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
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    # 40 indicates the number of epochs
    optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    # train the network
    print("[INFO] training network...")
    model_fit = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          batch_size=64, epochs=40, verbose=1)
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=64)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=label_names))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), model_fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), model_fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), model_fit.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 40), model_fit.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on CIFAR-10")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["output"])


if __name__ == '__main__':
    main()
