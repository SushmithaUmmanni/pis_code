# -*- coding: utf-8 -*-
"""Aspect-aware Preprocessing.

Resize to a fixed size, but maintain the aspect ratio

Attributes:
    width (int):
        Image width
    height (int):
        Image height
    inter (str):
        Interpolation method used for resizing the image
"""
import imutils
import cv2


class AspectAwarePreprocessor:
    """Preprocessor resizing an image to a fixed size, but maintaining the aspect ratio.

    Returns:
        array -- resized image with unchanged aspect ratio
    """

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """Initialize preprocessor.

        Arguments:
            width {int} -- image width
            height {int} -- image height

        Keyword Arguments:
            inter {str} -- Interpolation method used for resizing the image
                           (default: {cv2.INTER_AREA})
        """
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """Resize an image to a fixed size, while maintaining the aspect ratio

        Step #1: Determine the shortest dimension and resize along it.
        Step #2: Crop the image along the largest dimension to obtain the target width and height.

        Arguments:
            image {array} -- image to be preprocessed

        Returns:
            array -- resized image with unchanged aspect ratio
        """
        # grab the dimensions of the image and then initialize the deltas to use when cropping
        (height, width) = image.shape[:2]
        width_delta = 0
        height_delta = 0

        # if the width is smaller than the height, then resize along the width (i.e., the smaller
        # dimension) and then update the deltas to crop the height to the desired dimension
        if width < height:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            height_delta = int((image.shape[0] - self.height) / 2.0)

        # otherwise, the height is smaller than the width so resize along the height and then
        # update the deltas to crop along the width
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            width_delta = int((image.shape[1] - self.width) / 2.0)

        # now that our images have been resized, we need to re-grab the
        # width and height, followed by performing the crop
        (height, width) = image.shape[:2]
        image = image[height_delta : height - height_delta, width_delta : width - width_delta]
        # finally, resize the image to the provided spatial dimensions
        # to ensure our output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
