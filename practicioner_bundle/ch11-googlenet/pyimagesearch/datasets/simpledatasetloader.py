# -*- coding: utf-8 -*-
"""Image loader.

Attributes:
    preprocessors (list, optional):
        A list of image preprocessors that can be sequentially applied to a given input image
"""
import os
import cv2
import numpy as np


class SimpleDatasetLoader:
    """Initialize the image loader.

    Returns:
        tuple -- returns a tuple of the data and labels
    """
    def __init__(self, preprocessors=None):
        # specify the image preprocessors in a list. Each preprocessor will be a separate module
        # store the image preprocessor
        self.preprocessors = preprocessors
        # if the preprocessors are None, empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        """[summary]

        Arguments:
            image_paths {list]} -- Paths to the images in our dataset residing on disk

        Keyword Arguments:
            verbose {int} -- “verbosity level” can be used to print updates to a console
                             (default: {-1})

        Returns:
            tuple -- tuple of the data and labels
        """
        # initialize the list of features and labels
        data = []
        labels = []
        # loop over the input images
        for (i, image_path) in enumerate(image_paths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
