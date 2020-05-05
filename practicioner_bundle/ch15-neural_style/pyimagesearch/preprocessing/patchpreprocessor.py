# -*- coding: utf-8 -*-
"""Patch Preprocessor

The preprocessor randomly samples M x N regions of an image during the training process. This
method helps reduce regularization, since the network always receives different input image
(except, when the random cropping is the same).


Attributes:
    target_width (int):
        width of the image
    target_height (int):
        height of the image
"""
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    """Run the path preprocessor
    """

    def __init__(self, width, height):
        """Initialize the preprocessor

        Arguments:
            width {int} -- target width of the image
            height {int} -- target height of the image
        """
        self.target_width = width
        self.target_height = height

    def preprocess(self, image):
        """Extract a random crop from the image with the target width and height

        Arguments:
            image {array} -- image to be processed

        Returns:
            [array] -- processed image
        """
        return extract_patches_2d(image, (self.target_height, self.target_width), max_patches=1)[0]
