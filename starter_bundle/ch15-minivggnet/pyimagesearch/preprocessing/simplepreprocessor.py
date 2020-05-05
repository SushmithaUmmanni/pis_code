# -*- coding: utf-8 -*-
"""Image preprocessor that resizes the image while ignoring the aspect ratio.

Attributes:
    width (int):
        Target width of our input image after resizing.
    height (int):
        Target height of our input image after resizing.
    interpolation (int, optional):
        Type of interpolation algorithm used for resizing.
"""
import cv2


class SimplePreprocessor:
    """Simple image preprocessor

    The preprocessor can resize the image while keeping the same acpect ratio.

    Returns:
        array: resized image
    """

    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """Initialize preprocessor

        Args:
            width (int): Target width of our input image after resizing.
            height (int): Target height of our input image after resizing.
            interpolation (int, optional): Type of interpolation algorithm used for resizing.
                                           Defaults to cv2.INTER_AREA.
        """
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        """Resize the image to a fixed size, ignoring the aspect ratio.

        Args:
            image (array): Input image to be resized

        Returns:
            array: resized image
        """
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
