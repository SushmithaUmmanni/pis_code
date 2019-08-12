# -*- coding: utf-8 -*-
"""Building Tiny Imagenet Dataset

Example:
    $ python build_tiny_imagenet.py
"""
import os
import json
import numpy as np
import progressbar
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from pyimagesearch.io import HDF5DatasetWriter
from config import tiny_imagenet_config as config


def main():
    """Serialize the dataset
    """
    # grab the paths to the training images, then extract the training class labels and encode them
    train_paths = list(paths.list_images(config.TRAIN_IMAGES))
    train_labels = [p.split(os.path.sep)[-3] for p in train_paths]
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)

    # perform stratified sampling from the training set to construct a testing set
    split = train_test_split(train_paths,
                             train_labels,
                             test_size=config.NUM_TEST_IMAGES,
                             stratify=train_labels,
                             random_state=42)
    (train_paths, test_paths, train_labels, test_labels) = split

    # load the validation filename => class from file and then use these
    # mappings to build the validation paths and label lists
    mapping = open(config.VAL_MAPPINGS).read().strip().split("\n")
    mapping = [r.split("\t")[:2] for r in mapping]
    val_paths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in mapping]
    val_labels = label_encoder.transform([m[1] for m in mapping])

    # construct a list pairing the training, validation, and testing image paths
    # along with their corresponding labels and output HDF5 files
    datasets = [
        ("train", train_paths, train_labels, config.TRAIN_HDF5),
        ("val", val_paths, val_labels, config.VAL_HDF5),
        ("test", test_paths, test_labels, config.TEST_HDF5)
    ]

    # initialize the lists of RGB channel averages
    (R, G, B) = ([], [], [])

    # loop over the dataset tuples
    for (dataset_type, image_paths, labels, output_path) in datasets:
        # create HDF5 writer
        print("[INFO] building {}...".format(output_path))
        writer = HDF5DatasetWriter((len(image_paths), 64, 64, 3), output_path)
        # initialize the progress bar
        widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
                   progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

        # loop over the image paths
        for (i, (path, label)) in enumerate(zip(image_paths, labels)):
            # load the image from disk
            image = cv2.imread(path)

            # if we are building the training dataset, then compute the mean of each
            # channel in the image, then update the respective lists
            if dataset_type == "train":
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

            # add the image and label to the HDF5 dataset
            writer.add([image], [label])
            pbar.update(i)

        # close the HDF5 writer
        pbar.finish()
        writer.close()

    # construct a dictionary of averages, then serialize the means to a JSON file
    print("[INFO] serializing means...")
    rgb_dict = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    f = open(config.DATASET_MEAN, "w")
    f.write(json.dumps(rgb_dict))
    f.close()


if __name__ == "__main__":
    main()
