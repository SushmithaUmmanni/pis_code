# -*- coding: utf-8 -*-
"""Image loader."""
import os
import cv2
import numpy as np


class SimpleDatasetLoader:
    """
    Initialize the image loader.

    Attributes:
        preprocessors (list): List of image preprocessors that can be sequentially applied to
                              a given input image.
    """

    def __init__(self, preprocessors=None):
        """
        Initialize the simple dataset loader.

        Args:
            preprocessors (list, optional): List of image preprocessors that can be sequentially
                                  applied to a given input image.
        """
        # specify the image preprocessors in a list. Each preprocessor will be a separate module
        # store the image preprocessor
        self.preprocessors = preprocessors
        # if the preprocessors are None, empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        """
        Load images and labels.

        Args:
            image_paths (list): Paths to the images in our dataset residing on disk.
            verbose (int, optional): verbosity level can be used to print updates to a console.
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
