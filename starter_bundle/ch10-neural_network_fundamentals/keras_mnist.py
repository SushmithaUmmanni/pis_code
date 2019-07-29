# -*- coding: utf-8 -*-
"""Train custom keras model on MNIST dataset.

Example:
    $ python keras_mnist.py --output output/keras_mnist.png

Attributes:
    dataset (str):
        The path to where our input image dataset resides on disk.
"""
import argparse
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
import matplotlib.pyplot as plt


def main():
    """Train custom keras model on the MNIST dataset.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True,
                      help="path to the output loss/accuracy plot")
    args = vars(args.parse_args())

    # grab the MNIST dataset (if this is your first time using this
    # dataset then the 11MB download may take a minute)
    print("[INFO] accessing MNIST...")
    ((train_x, train_y), (test_x, test_y)) = mnist.load_data()

    # each image in the MNIST dataset is represented as a 28x28x1
    # image, but in order to apply a standard neural network we must
    # first "flatten" the image to be simple list of 28x28=784 pixels
    train_x = train_x.reshape((train_x.shape[0], 28 * 28 * 1))
    test_x = test_x.reshape((test_x.shape[0], 28 * 28 * 1))
    # scale data to the range of [0, 1]
    train_x = train_x.astype("float32") / 255.0
    test_x = test_x.astype("float32") / 255.0

    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_y)
    test_y = label_binarizer.transform(test_y)

    # define the 784-256-128-10 architecture using Keras
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))

    # train the model using SGD
    print("[INFO] training network...")
    sgd = SGD(0.01)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    model_fit = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          epochs=100, batch_size=128)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=128)
    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(x) for x in label_binarizer.classes_]))

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


if __name__ == '__main__':
    main()
