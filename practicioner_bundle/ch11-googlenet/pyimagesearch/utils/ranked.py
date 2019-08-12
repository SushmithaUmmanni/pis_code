# -*- coding: utf-8 -*-
"""Measuring rank-1 and rank-5 Accuracies.
"""
import numpy as np


def rank5_accuracy(preds, labels):
    """Compute rank1 and rank5 accuaries

    Arguments:
        preds {array} -- An NxT matrix where N, the number of rows, contains the probabilities
                          associated with each class label T
        labels {array} -- The ground-truth labels for the images in the dataset.

    Returns:
        [tuple] -- rank1 and rank5 accuaries
    """
    # initialize the rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0
    # loop over the predictions and ground-truth labels
    for (prediction, ground_truth) in zip(preds, labels):
        # sort the probabilities by their index in descending order
        # so that the more confident guesses are at the front of the list
        prediction = np.argsort(prediction)[::-1]
        # check if the ground-truth label is in the top-5 predictions
        if ground_truth in prediction[:5]:
            rank5 += 1
        # check to see if the ground-truth is the #1 prediction
        if ground_truth == prediction[0]:
            rank1 += 1

    # compute the final rank-1 and rank-5 accuracies
    rank1 /= float(len(preds))
    rank5 /= float(len(preds))

    # return a tuple of the rank-1 and rank-5 accuracies
    return (rank1, rank5)
