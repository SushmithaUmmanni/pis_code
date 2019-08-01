# -*- coding: utf-8 -*-
"""Writing Features to an HDF5 Dataset.

This class is responsible for taking an input set of NumPy arrays
(whether features, raw images, etc.) and writing them to HDF5 format.

Examples:
    $ python extract_features.py --dataset ../datasets/animals/images
                                 --output ../datasets/animals/hdf5/features.hdf5

    $ python extract_features.py --dataset ../datasets/caltech-101/images
                                 --output ../datasets/caltech-101/hdf5/features.hdf5

    $ python extract_features.py --dataset ../datasets/flowers17/images
                                 --output ../datasets/flowers17/hdf5/features.hdf5

Attributes:
    dataset (str):
        path to input dataset
    output (str):
        path to output HDF5 file
    batch_size (int, optional):
        batch size of images to be passed through network (default = 32)
    buffer_size (int, optional):
        size of feature extraction buffer (default = 1000)
"""
import os
import random
import argparse
import numpy as np
import progressbar
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from pyimagesearch.io import HDF5DatasetWriter


def main():
    """Run feature extraction.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args.add_argument("-o", "--output", required=True,
                      help="path to output HDF5 file")
    args.add_argument("-b", "--batch-size", type=int, default=32,
                      help="batch size of images to be passed through network")
    args.add_argument("-s", "--buffer-size", type=int, default=1000,
                      help="size of feature extraction buffer")
    args = vars(args.parse_args())

    # store the batch size in a convenience variable
    batch_size = args["batch_size"]
    # grab the list of images that we'll be describing then randomly
    # shuffle them to allow for easy training and testing splits via
    # array slicing during training time
    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))
    random.shuffle(image_paths)
    # extract the class labels from the image paths then encode the
    # labels
    labels = [p.split(os.path.sep)[-2] for p in image_paths]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # load the VGG16 network
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet", include_top=False)
    # initialize the HDF5 dataset writer, then store the class label
    # names in the dataset
    dataset = HDF5DatasetWriter((len(image_paths), 512 * 7 * 7),
                                args["output"],
                                data_key="features",
                                buffer_size=args["buffer_size"])
    dataset.store_class_labels(label_encoder.classes_)
    # initialize the progress bar
    widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()
    # loop over the images in batches
    for i in np.arange(0, len(image_paths), batch_size):
        # extract the batch of images and labels, then initialize the
        # list of actual images that will be passed through the network
        # for feature extraction
        batch_paths = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_images = []
        # loop over the images and labels in the current batch
        for (_, image_path) in enumerate(batch_paths):
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            # preprocess the image by (1) expanding the dimensions and
            # (2) subtracting the mean RGB pixel intensity from the
            # ImageNet dataset
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            # add the image to the batch
            batch_images.append(image)
        # pass the images through the network and use the outputs as our actual features
        batch_images = np.vstack(batch_images)
        features = model.predict(batch_images, batch_size=batch_size)
        # reshape the features so that each image is represented by
        # a flattened feature vector of the `MaxPooling2D` outputs
        features = features.reshape((features.shape[0], 512 * 7 * 7))
        # add the features and labels to our HDF5 dataset
        dataset.add(features, batch_labels)
        pbar.update(i)

    # close the dataset
    dataset.close()
    pbar.finish()


if __name__ == "__main__":
    main()
