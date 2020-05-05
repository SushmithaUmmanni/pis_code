# -*- coding: utf-8 -*-
"""
HDF5 Dataset Generators.

The generator class is responsible for yielding batches of images and labels from our HDF5 database.
"""

from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    """
    HDF5 Dataset Generator.

    Atrributes:
        batch_size (int): Size of mini-batches when training the network.
        preprocessors (list): List of image preprocessors.
        augmentation (bool): Augment data in HDF5DatasetGenerator.
        binarize (bool): Labels will be encoded as one-hot vector.
        classes (int): Number of unique class labels in our database.
        database (dict): Loaded HDF5 dataset.
        num_images (int): Number of images.
    """

    def __init__(self, dataset_path, batch_size, preprocessors=None, augmentation=None,
                 binarize=True, classes=2):
        """
        Initialize the database generator.

        Args:
            dataset_path (str): Path to the HDF5 database storing our images and corresponding
                                class labels.
            batch_size (int): Size of mini-batches to yield when training our network.
            preprocessors (list, optional): List of image preprocessors we are going to apply.
            augmentation (bool, optional): If True, then a Keras ImageDataGenerator will be
                                           supplied to augment the data directly inside our HDF5DatasetGenerator.
            binarize (bool, optional): If True, then the labels will be binarized as one-hot
                                       encoded vector.
            classes (int, optional): Number of unique class labels in our database.
        """
        # store the batch size, preprocessors, and data augmentor, whether or
        # not the labels should be binarized, along with the total number of classes
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.augmentation = augmentation
        self.binarize = binarize
        self.classes = classes
        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.database = h5py.File(dataset_path)
        self.num_images = self.database["labels"].shape[0]

    def generator(self, passes=np.inf):
        """
        Yield batches of images and class labels to the Keras .fit_generator function when training
        a network.

        Args:
            passes (int, optional): Value representing the total number of epochs.
        """
        # initialize the epoch count
        epochs = 0
        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 database
            for i in np.arange(0, self.num_images, self.batch_size):
                # extract the images and labels from the HDF database
                images = self.database["images"][i : i + self.batch_size]
                labels = self.database["labels"][i : i + self.batch_size]
                # check to see if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    processed_images = []
                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each to the image
                        for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)
                        # update the list of processed images
                        processed_images.append(image)
                    # update the images array to be the processed images
                    images = np.array(processed_images)
                # if the data augmenator exists, apply it
                if self.augmentation is not None:
                    (images, labels) = next(self.augmentation.flow(images, labels,
                                       batch_size=self.batch_size))
                # yield a tuple of images and labels
                yield (images, labels)
            # increment the total number of epochs
            epochs += 1

    def close(self):
        """Close the database connection."""
        # close the database
        self.database.close()
