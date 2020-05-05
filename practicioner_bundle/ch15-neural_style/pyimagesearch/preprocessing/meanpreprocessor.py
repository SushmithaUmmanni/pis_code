# -*- coding: utf-8 -*-
"""Mean Preprocessor.

The preprocessor substracts the mean RGB valus computed across the entire
dataset from the image to be processed.

Attributes:
    r_mean (float):
        mean R value computed across the entire dataset
    g_mean (float):
        mean G value computed across the entire dataset
    b_mean (float):
        mean B value computed across the entire dataset
"""
import cv2


class MeanPreprocessor:
    """Run the mean preprocessor
    """

    def __init__(self, r_mean, g_mean, b_mean):
        """Initialize the mean preprocessor

        Arguments:
            r_mean {float} -- mean R value computed across the entire dataset
            g_mean {float} -- mean G value computed across the entire dataset
            b_mean {float} -- mean B value computed across the entire dataset
        """
        # store the Red, Green, and Blue channel averages across a training set
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        """Substract the means for each channel

        Arguments:
            image {array} -- image to be processed

        Returns:
            array -- processed image
        """
        # split the image into its respective Red, Green, and Blue channels
        (B, G, R) = cv2.split(image.astype("float32"))
        # subtract the means for each channel
        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean
        # merge the channels back together and return the image
        return cv2.merge([B, G, R])
