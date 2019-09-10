# -*- coding: utf-8 -*-
""" ImageNet Helper Utility

This class generates .lst files for the training, testing, and validation splits, respectively.
"""
import os
import numpy as np


class ImageNetHelper:
    """ImageNet Helper Utility Class
    """

    def __init__(self, config):
        """Initialize the ImageNet helper

        Arguments:
            config {obj} -- network configuration file
        """
        # store the configuration object
        self.config = config
        # build the label mappings and validation blacklist
        self.label_mappings = self.build_class_labels()
        self.val_blacklist = self.build_blackist()

    def build_class_labels(self):
        """Build human readable class labels mappings

        Returns:
            dict -- label mappings between word ID and label
        """
        # load the contents of the file that maps the WordNet IDs
        # to integers, then initialize the label mappings dictionary
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        label_mappings = {}
        # loop over the labels
        for row in rows:
            # split the row into the WordNet ID, label integer, and human readable label
            (word_id, label, _) = row.split(" ")
            # update the label mappings dictionary using the word ID as the key and the label
            # as the value, subtracting `1` from the label since MATLAB is one-indexed while
            # Python is zero-indexed
            label_mappings[word_id] = int(label) - 1
        # return the label mappings dictionary
        return label_mappings

    def build_blackist(self):
        """Exclude unique integer names of the validation files due to ambiguous labels.

        Returns:
            set -- blacklisted image IDs
        """
        # load the list of blacklisted image IDs and convert them to a set
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))
        # return the blacklisted image IDs
        return rows

    def build_training_set(self):
        """Build training set

        Returns:
            tuple -- a tuple of image paths and associated integer class labels
        """
        # load the contents of the training input file that lists the partial image ID and image
        # number, then initialize the list of image paths and class labels
        rows = open(self.config.TRAIN_LIST).read().strip()
        rows = rows.split("\n")
        paths = []
        labels = []
        # loop over the rows in the input training file
        for row in rows:
            # break the row into the partial path and image number (the image number
            # is sequential and is essentially useless to us)
            (partial_path, _) = row.strip().split(" ")
            # construct the full path to the training image, then grab the word
            # ID from the path and use it to determine the integer class label
            path = os.path.sep.join([self.config.IMAGES_PATH,
                                     "train", "{}.JPEG".format(partial_path)])
            word_id = partial_path.split("/")[0]
            label = self.label_mappings[word_id]
            # update the respective paths and label lists
            paths.append(path)
            labels.append(label)
        # return a tuple of image paths and associated integer class labels
        return (np.array(paths), np.array(labels))

    def build_validation_set(self):
        """Build validation set

        Returns:
            tuple -- a tuple of image paths and associated integer class labels
        """
        # initialize the list of image paths and class labels
        paths = []
        labels = []
        # load the contents of the file that lists the partial validation image filenames
        val_filenames = open(self.config.VAL_LIST).read()
        val_filenames = val_filenames.strip().split("\n")
        # load the contents of the file that contains the *actual*
        # ground-truth integer class labels for the validation set
        val_labels = open(self.config.VAL_LABELS).read()
        val_labels = val_labels.strip().split("\n")
        # loop over the validation data
        for (row, label) in zip(val_filenames, val_labels):
            # break the row into the partial path and image number
            (partial_path, image_num) = row.strip().split(" ")
            # if the image number is in the blacklist set then we
            # should ignore this validation image
            if image_num in self.val_blacklist:
                continue
            # construct the full path to the validation image, then
            # update the respective paths and labels lists
            path = os.path.sep.join([self.config.IMAGES_PATH, "val",
                                     "{}.JPEG".format(partial_path)])
            paths.append(path)
            labels.append(int(label) - 1)
        # return a tuple of image paths and associated integer class labels
        return (np.array(paths), np.array(labels))
