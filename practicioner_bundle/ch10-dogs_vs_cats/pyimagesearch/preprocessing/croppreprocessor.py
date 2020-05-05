# -*- coding: utf-8 -*-
"""Crop Preprocessor

An over-sampling pre-processor used at testing time to sample five regions of an input image
(the four corners + center area) along with their corresponding horizontal flips (for a total
of 10 crops).

Attributes:
    width (int):
        width of the image
    height (int):
        height of the image
    horizontal_flip ()

    interpolation
"""
import numpy as np
import cv2


class CropPreprocessor:
    """Run the preprocessor
    """

    def __init__(self, width, height, horizontal_flip=True, inter=cv2.INTER_AREA):
        """Initialize the preprocessor

        Arguments:
            width {int} -- target width of the image
            height {int} -- target height of the image

        Keyword Arguments:
            horizontal_flip {bool} -- enables horizontal flipping (default: {True})
            inter {func} -- interpolation algorithm for image resizing (default: {cv2.INTER_AREA})
        """
        # store the target image width, height, whether or not horizontal flips should
        # be included, along with the interpolation method used when resizing
        self.target_width = width
        self.target_height = height
        self.horizontal_flip = horizontal_flip
        self.interpolation = inter

    def preprocess(self, image):
        """Sample five regions of an input image along with their corresponding horizontal flips

        Arguments:
            image {array} -- image to be processed

        Returns:
            [array] -- processed image
        """
        # initialize the list of crops
        crops = []
        # grab the width and height of the image then use these
        # dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.target_width, self.target_height],
            [w - self.target_width, 0, w, self.target_height],
            [w - self.target_width, h - self.target_height, w, h],
            [0, h - self.target_height, self.target_width, h],
        ]
        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.target_width))
        dH = int(0.5 * (h - self.target_height))
        coords.append([dW, dH, w - dW, h - dH])

        # loop over the coordinates, extract each of the crops,
        # and resize each of them to a fixed size
        for (start_x, start_y, end_x, end_y) in coords:
            crop = image[start_y:end_y, start_x:end_x]
            crop = cv2.resize(crop, (self.target_width, self.target_height), interpolation=self.interpolation)
            crops.append(crop)
        # check to see if the horizontal flips should be taken
        if self.horizontal_flip:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
        # return the set of crops
        return np.array(crops)
