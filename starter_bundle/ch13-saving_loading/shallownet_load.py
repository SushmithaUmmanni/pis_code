# -*- coding: utf-8 -*-
"""Loading a pre-trained model from disk.

Learn how to load a pre-trained model from disk. We will classify individual images from the
Animals dataset and then display the classified images to our screen.

Example:
    $ python shallownet_load.py --dataset ../datasets/animals --model shallownet_weights.hdf5

Attributes:
    dataset (str):
        The path to where our input image dataset resides on disk.
    model (str):
        The path to the pre-trained model.
"""
import argparse
import cv2
import numpy as np
from keras.models import load_model
from imutils import paths
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader


def main():
    """Load pre-trained model from disk
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    args = vars(args.parse_args())

    # initialize the class labels
    class_labels = ["cat", "dog", "panda"]

    # grab the list of images in the dataset then randomly sample indexes into the image paths list
    print("[INFO] sampling images...")
    image_paths = np.array(list(paths.list_images(args["dataset"])))
    idxs = np.random.randint(0, len(image_paths), size=(10,))
    image_paths = image_paths[idxs]

    # initialize the image preprocessors
    simple_preprocessor = SimplePreprocessor(32, 32)
    image_to_array_preprocessor = ImageToArrayPreprocessor()

    # load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
    dataset_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor, image_to_array_preprocessor])
    (data, _) = dataset_loader.load(image_paths)
    data = data.astype("float") / 255.0

    # load the pre-trained network
    print("[INFO] loading pre-trained network...")
    model = load_model(args["model"])

    # make predictions on the images
    print("[INFO] predicting...")
    preds = model.predict(data, batch_size=32).argmax(axis=1)
    # loop over the sample images
    for (i, image_path) in enumerate(image_paths):
        # load the example image, draw the prediction, and display it to our screen
        image = cv2.imread(image_path)
        cv2.putText(
            image, "Label: {}".format(class_labels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow("Image", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
