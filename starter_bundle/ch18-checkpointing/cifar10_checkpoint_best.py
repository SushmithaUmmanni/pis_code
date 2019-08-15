# -*- coding: utf-8 -*-
"""Checkpointing Best Neural Network Only.

Here only one file will be saved. Whenever our metric improves, that file will
simply be overwritten.

Example:
    $ python cifar10_checkpoint_best.py --weights weights/best/cifar10_best_weights.hdf5

Attributes:
    weights (str):
        The path to the best model wieghts file.
"""
import argparse
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
from pyimagesearch.nn.conv import MiniVGGNet


def main():
    """Train and checkpoint the neural network.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--weights", required=True,
                      help="path to best model weights file")
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

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # construct the callback to save only the *best* model to disk based on the validation loss
    checkpoint = ModelCheckpoint(args["weights"],
                                 monitor="val_loss",
                                 save_best_only=True,
                                 verbose=1)
    callbacks = [checkpoint]
    # train the network
    print("[INFO] training network...")
    model.fit(train_x,
              train_y,
              validation_data=(test_x, test_y),
              batch_size=64,
              epochs=40,
              callbacks=callbacks,
              verbose=2)


if __name__ == '__main__':
    main()
