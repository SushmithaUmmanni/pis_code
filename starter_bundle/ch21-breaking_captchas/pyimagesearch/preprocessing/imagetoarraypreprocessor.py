# -*- coding: utf-8 -*-
"""Image to array preprocessor.

Attributes:
    data_format (str):
        The path to the output plot used to visualize loss and accuracy over time.
        {None} indicates that the setting inside keras.json should be used.
"""
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    """Image to array preprocessor

    Returns:
        array -- A 3D Numpy array
    """

    def __init__(self, data_format=None):
        """Initialize preprocessor

        Keyword Arguments:
            data_format {string} -- Image data format, either "channels_first" or "channels_last".
                                    (default: {None})
        """
        # store the image data format
        self.data_format = data_format

    def preprocess(self, image):
        """apply the Keras utility function that correctly rearranges the dimensions of the image

        Arguments:
            image {array} -- image to be processed

        Returns:
           array -- A 3D Numpy array
        """
        return img_to_array(image, data_format=self.data_format)
