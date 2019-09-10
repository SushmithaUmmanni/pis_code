# -*- coding: utf-8 -*-
"""Plot accuracy and loss with mxnet

MXNet is missing monitoring utilities out of the box. Here you will learn how parse mxnet log
files, extract training and validation information (including loss and accuracy), and then plot
this information over time

For more information, see https://www.pyimagesearch.com/2017/12/25/plot-accuracy-loss-mxnet/

Examples:
    $ python plot_log.py --network AlexNet --dataset ImageNet

Attributes:
    network (int):
        Name of the network
    dataset (str)
        Name of the dataset
"""
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Plot accuracy and loss with mxnet
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-n", "--network", required=True,
                      help="name of network")
    args.add_argument("-d", "--dataset", required=True,
                      help="name of dataset")
    args = vars(args.parse_args())

    # define the paths to the training logs
    logs = [
        (65, "training_0.log"),			# lr=1e-2
        (85, "training_65.log"),		# lr=1e-3
        (100, "training_85.log"),		# lr=1e-4
    ]

    # initialize the list of train rank-1 and rank-5 accuracies, along with the training loss
    (train_rank1, train_rank5, train_loss) = ([], [], [])

    # initialize the list of validation rank-1 and rank-5 accuracies, along with the validation loss
    (val_rank1, val_rank5, val_loss) = ([], [], [])

    # loop over the training logs
    for (i, (end_epoch, log_path)) in enumerate(logs):
        # load the contents of the log file, then initialize the batch
        # lists for the training and validation data
        rows = open(log_path).read().strip()
        (b_train_rank_1, b_train_rank_5, b_train_loss) = ([], [], [])
        (b_val_rank_1, b_val_rank_5, b_val_loss) = ([], [], [])

        # grab the set of training epochs
        epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
        epochs = sorted([int(epoch) for epoch in epochs])
        # loop over the epochs
        for epoch in epochs:
            # find all rank-1 accuracies, rank-5 accuracies, and loss
            # values, then take the final entry in the list for each
            tmp = r'Epoch\[' + str(epoch) + r'\].*accuracy=([0]*\.?[0-9]+)'
            rank1 = re.findall(tmp, rows)[-2]
            tmp = r'Epoch\[' + str(epoch) + r'\].*top_k_accuracy_5=([0]*\.?[0-9]+)'
            rank5 = re.findall(tmp, rows)[-2]
            tmp = r'Epoch\[' + str(epoch) + r'\].*cross-entropy=([0-9]*\.?[0-9]+)'
            loss = re.findall(tmp, rows)[-2]

            # update the batch training lists
            b_train_rank_1.append(float(rank1))
            b_train_rank_5.append(float(rank5))
            b_train_loss.append(float(loss))

        # extract the validation rank-1 and rank-5 accuracies for each
        # epoch, followed by the loss
        b_val_rank_1 = re.findall(r'Validation-accuracy=(.*)', rows)
        b_val_rank_5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
        b_val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

        # convert the validation rank-1, rank-5, and loss lists to floats
        b_val_rank_1 = [float(x) for x in b_val_rank_1]
        b_val_rank_5 = [float(x) for x in b_val_rank_5]
        b_val_loss = [float(x) for x in b_val_loss]

        # check to see if we are examining a log file other than the first one, and if so,
        # use the number of the final epoch in the log file as our slice index
        if i > 0 and end_epoch is not None:
            train_end = end_epoch - logs[i - 1][0]
            val_end = end_epoch - logs[i - 1][0]
        # otherwise, this is the first epoch so no subtraction needs to be done
        else:
            train_end = end_epoch
            val_end = end_epoch

        # update the training lists
        train_rank1.extend(b_train_rank_1[0:train_end])
        train_rank5.extend(b_train_rank_5[0:train_end])
        train_loss.extend(b_train_loss[0:train_end])

        # update the validation lists
        val_rank1.extend(b_val_rank_1[0:val_end])
        val_rank5.extend(b_val_rank_5[0:val_end])
        val_loss.extend(b_val_loss[0:val_end])

    # plot the accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(train_rank1)), train_rank1, label="train_rank1")
    plt.plot(np.arange(0, len(train_rank5)), train_rank5, label="train_rank5")
    plt.plot(np.arange(0, len(val_rank1)), val_rank1, label="val_rank1")
    plt.plot(np.arange(0, len(val_rank5)), val_rank5, label="val_rank5")
    plt.title("{}: rank-1 and rank-5 accuracy on {}".format(args["network"], args["dataset"]))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # plot the losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(train_loss)), train_loss, label="train_loss")
    plt.plot(np.arange(0, len(val_loss)), val_loss, label="val_loss")
    plt.title("{}: cross-entropy loss on {}".format(args["network"], args["dataset"]))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
