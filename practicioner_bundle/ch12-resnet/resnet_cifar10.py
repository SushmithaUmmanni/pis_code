# -*- coding: utf-8 -*-
"""Training ResNet on CIFAR-10

Example:
    $ python resnet_cifar10.py --checkpoints output/checkpoints

    $ python resnet_cifar10.py --checkpoints output/checkpoints \
                               --model output/checkpoints/epoch_50.hdf5 \
                               --start-epoch 50

    # python resnet_cifar10.py --checkpoints output/checkpoints \
                               --model output/checkpoints/epoch_75.hdf5 \
                               --start-epoch 75

Attributes:
    checkpoints (str):
        path to output checkpoint directory
    model (str, optional):
        path to *specific* model checkpoint to load
    start-epoch (int, default: 10)
        epoch to restart training at
"""
import sys
import argparse
import numpy as np
import matplotlib
import keras.backend as K
from keras.models import load_model
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.nn.conv import ResNet
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")
# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)


def main():
    """Train ResNet on Cifar10 dataset
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--checkpoints", required=True,
                      help="path to output checkpoint directory")
    args.add_argument("-m", "--model", type=str,
                      help="path to *specific* model checkpoint to load")
    args.add_argument("-s", "--start-epoch", type=int, default=0,
                      help="epoch to restart training at")
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
    aug = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # if there is no specific model checkpoint supplied, then initialize
    # the network (ResNet-56) and compile the model
    if args["model"] is None:
        print("[INFO] compiling model...")
        opt = SGD(lr=1e-1)
        model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # otherwise, load the checkpoint from disk
    else:
        print("[INFO] loading {}...".format(args["model"]))
        model = load_model(args["model"])
        # update the learning rate
        print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
        K.set_value(model.optimizer.lr, 1e-5)
        print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

    # construct the set of callbacks
    callbacks = [
        EpochCheckpoint(args["checkpoints"], every=5,
                        start_at=args["start_epoch"]),
        TrainingMonitor("output/resnet56_cifar10.png",
                        json_path="output/resnet56_cifar10.json",
                        start_at=args["start_epoch"])]

    # train the network
    print("[INFO] training network...")
    model.fit_generator(
        aug.flow(train_x, train_y, batch_size=128),
        validation_data=(test_x, test_y),
        steps_per_epoch=len(train_x) // 128, epochs=100,
        callbacks=callbacks, verbose=1)


if __name__ == "__main__":
    main()
