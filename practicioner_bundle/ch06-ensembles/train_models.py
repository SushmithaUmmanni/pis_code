# -*- coding: utf-8 -*-
"""Constructing an Ensemble of CNNs

First, the best possible model has to be found. Then, usually 5-10 CNN models ara
used for an ensemble. Based on Jensen’s Inequality the performance of an ensemble
cannot be worse than a single model

Example:
    $ python train_models.py --output output --models models

Attributes:
    output (str):
        base output directory where we’ll save classification reports along with loss/accuracy
        plots for each of the networks we will train
    models (str):
        path to the output directory where we will be storing our serialized network weights
    num_models (int, optional)
        number of networks in our ensemble (default: 5)
"""
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


def main():
    """Train ensemble model.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True, help="path to output directory")
    args.add_argument("-m", "--models", required=True, help="path to output models directory")
    args.add_argument("-n", "--num-models", type=int, default=5, help="# of models to train")
    args = vars(args.parse_args())

    # load the training and testing data, then scale it into the range [0, 1]
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    train_x = train_x.astype("float") / 255.0
    test_x = test_x.astype("float") / 255.0

    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_y)
    test_y = label_binarizer.transform(test_y)

    # initialize the label names for the CIFAR-10 dataset
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # construct the image generator for data augmentation
    augmentation = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest"
    )

    # loop over the number of models to train
    for i in np.arange(0, args["num_models"]):
        # initialize the optimizer and model
        print("[INFO] training model {}/{}".format(i + 1, args["num_models"]))
        opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
        model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the network
        model_fit = model.fit_generator(
            augmentation.flow(train_x, train_y, batch_size=64),
            validation_data=(test_x, test_y),
            epochs=40,
            steps_per_epoch=len(train_x) // 64,
            verbose=1,
        )
        # save the model to disk
        path = [args["models"], "model_{}.model".format(i)]
        model.save(os.path.sep.join(path))

        # evaluate the network
        predictions = model.predict(test_x, batch_size=64)
        report = classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)

        # save the classification report to file
        path = [args["output"], "model_{}.txt".format(i)]
        f = open(os.path.sep.join(path), "w")
        f.write(report)
        f.close()

        # plot the training loss and accuracy
        path = [args["output"], "model_{}.png".format(i)]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, 40), model_fit.history["loss"], label="train_loss")
        plt.plot(np.arange(0, 40), model_fit.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, 40), model_fit.history["acc"], label="train_acc")
        plt.plot(np.arange(0, 40), model_fit.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy for model {}".format(i))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(os.path.sep.join(path))
        plt.close()


if __name__ == "__main__":
    main()
