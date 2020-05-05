# -*- coding: utf-8 -*-
"""Train custom keras model on Cifar10 dataset.

Example:
    $ python keras_cifar10.py --output output/keras_cifar10.png

Attributes:
    dataset (str):
        The path to where our input image dataset resides on disk.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10


def main():
    """Train custom keras model on the Cifar10 dataset.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
    args = vars(args.parse_args())

    # load the training and testing data, scale it into the range [0, 1],
    # then reshape the design matrix
    print("[INFO] loading CIFAR-10 data...")
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    train_x = train_x.astype("float") / 255.0
    test_x = test_x.astype("float") / 255.0
    train_x = train_x.reshape((train_x.shape[0], 3072))
    test_x = test_x.reshape((test_x.shape[0], 3072))

    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_y)
    test_y = label_binarizer.transform(test_y)

    # initialize the label names for the CIFAR-10 dataset
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # define the 3072-1024-512-10 architecture using Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # train the model using SGD
    print("[INFO] training network...")
    sgd = SGD(0.01)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    model_fit = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=32)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 100), model_fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), model_fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), model_fit.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 100), model_fit.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["output"])


if __name__ == "__main__":
    main()
