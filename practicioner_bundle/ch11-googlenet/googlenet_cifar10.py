# -*- coding: utf-8 -*-
"""Training GoogLeNet on CIFAR-10

Example:
    $ python googlenet_cifar10.py --output output --model output/minigooglenet_cifar10.hdf5

Attributes:
    model (str):
        the path to the output file where GoogLeNet will be serialized after training
    output (str):
        path to our output dir where we will store any plots, logs, etc
"""
import os
import argparse
import matplotlib
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.nn.conv import MiniGoogLeNet

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

# define the total number of epochs to train for along with the initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3


def poly_decay(epoch):
    """Implement custom learning rate schedule

    Arguments:
        epoch {int} -- current epoch

    Returns:
        float -- updated learning rate
    """
    # initialize the maximum number of epochs, base learning rate, and power of the polynomial
    max_epochs = NUM_EPOCHS
    base_learning_rate = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = base_learning_rate * (1 - (epoch / float(max_epochs))) ** power
    # return the new learning rate
    return alpha


def main():
    """Train GoogLeNet on Cifar10 dataset
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args.add_argument("-o", "--output", required=True,
                      help="path to output directory (logs, plots, etc.)")
    args = vars(args.parse_args())

    # load the training and testing data, converting the images from integers to floats
    print("[INFO] loading CIFAR-10 data...")
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    train_x = train_x.astype("float")
    test_x = test_x.astype("float")

    # apply mean subtraction to the data
    mean = np.mean(train_x, axis=0)
    train_x -= mean
    test_x -= mean

    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_y)
    test_y = label_binarizer.transform(test_y)

    # construct the image generator for data augmentation
    augmentation = ImageDataGenerator(width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      horizontal_flip=True,
                                      fill_mode="nearest")

    # construct the set of callbacks
    fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
    json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
    callbacks = [TrainingMonitor(fig_path, json_path=json_path), LearningRateScheduler(poly_decay)]

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=INIT_LR, momentum=0.9)
    model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    model.fit_generator(augmentation.flow(train_x, train_y, batch_size=64),
                        validation_data=(test_x, test_y),
                        steps_per_epoch=len(train_x) // 64,
                        epochs=NUM_EPOCHS,
                        callbacks=callbacks,
                        verbose=1)

    # save the network to disk
    print("[INFO] serializing network...")
    model.save(args["model"])


if __name__ == "__main__":
    main()
