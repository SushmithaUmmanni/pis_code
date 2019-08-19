# -*- coding: utf-8 -*-
"""Training ResNet on CIFAR-10

Example:
    $ python resnet_cifar10_decay.py --output output \
                                     --model output/resnet_cifar10.hdf5

Attributes:
    model (str):
        path to our final serialized model after training
    output (str):
        base directory to where we will store any logs, plots, etc.

"""
import os
import sys
import argparse
import matplotlib
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.nn.conv import ResNet
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")
# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)


# define the total number of epochs to train for along with the initial learning rate
NUM_EPOCHS = 100
INIT_LR = 1e-1


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
    power = 1.0  # 1.0 indicates linear decay
    # compute the new learning rate based on polynomial decay
    alpha = base_learning_rate * (1 - (epoch / float(max_epochs))) ** power
    # return the new learning rate
    return alpha


def main():
    """Train ResNet on Cifar10 dataset
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args.add_argument("-o", "--output", required=True,
                      help="path to output directory (logs, plots, etc.)")
    args = vars(args.parse_args())

    # load the training and testing data, converting the images from
    # integers to floats
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
    aug = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # construct the set of callbacks
    fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
    json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
    callbacks = [
        TrainingMonitor(fig_path, json_path=json_path),
        LearningRateScheduler(poly_decay)
        ]

    # initialize the optimizer and model (ResNet-56)
    print("[INFO] compiling model...")
    opt = SGD(lr=INIT_LR, momentum=0.9)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    model.fit_generator(aug.flow(train_x, train_y, batch_size=128),
                        validation_data=(test_x, test_y),
                        steps_per_epoch=len(train_x) // 128,
                        epochs=NUM_EPOCHS,
                        callbacks=callbacks,
                        verbose=1)

    # save the network to disk
    print("[INFO] serializing network...")
    model.save(args["model"])


if __name__ == "__main__":
    main()
