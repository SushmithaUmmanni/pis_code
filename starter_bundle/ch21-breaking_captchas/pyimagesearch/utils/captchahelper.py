# -*- coding: utf-8 -*-
"""Preprocessing the digits.
"""
import imutils
import cv2


def preprocess(image, width, height):
    """Pad and resize the input images to a fixed size without distorting their aspect ratio.

    Arguments:
        image {array} -- input image that will be padded and resized
        width {[type]} -- target output width of the image
        height {[type]} -- target output height of the image

    Returns:
        array -- pre-processed image
    """
    # grab the dimensions of the image, then initialize the padding values
    (height, width) = image.shape[:2]
    # if the width is greater than the height then resize along the width
    if width > height:
        image = imutils.resize(image, width=width)
    # otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)
    # determine the padding values for the width and height toobtain the target dimensions
    pad_width = int((width - image.shape[1]) / 2.0)
    pad_height = int((height - image.shape[0]) / 2.0)
    # pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    # return the pre-processed image
    return image
