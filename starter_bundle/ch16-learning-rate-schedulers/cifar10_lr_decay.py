# -*- coding: utf-8 -*-
"""Implementation of a Custom Learning Rate Schedules in Keras.

We will define a  function that will drop the learning rate by a certain factor F after every D
epochs:

alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

initAlpha => initial learning rate
F => factor value controlling the rate in which the learning rate drops
D => "drop every" epochs value
E => current epoch

The larger our factor F is, the slower the learning rate will decay. Conversely, the smaller the
factor F is the faster the learning rate will decrease

Example:
    $ python cifar10_lr_decay.py --output output/lr_decay_f0.25_plot.png

Attributes:
    output (str):
        The path to our output loss/accuracy plot.
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


def step_decay(epoch):
    """Calculate learning rate for the current epoch

    Arguments:
        epoch {int} -- number of the current epoch

    Returns:
        float -- new learning rate for the current epoch
    """
    # initialize the base initial learning rate, drop factor, and epochs to drop every
    init_alpha = 0.01
    factor = 0.5  # drops the learning rate by "factor"
    drop_every = 5
    # compute learning rate for the current epoch
    alpha = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))
    # return the learning rate
    return float(alpha)


def main():
    """Train and evaluate MiniVGGNet on Cifar10 with custom written learning rate schedule.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True,
                      help="path to the output loss/accuracy plot")
    args = vars(args.parse_args())

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

    # define the set of callbacks to be passed to the model during training
    callbacks = [LearningRateScheduler(step_decay)]
    # initialize the optimizer and model
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train the network
    model_fit = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

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
