# -*- coding: utf-8 -*-
"""Demonstration of regularization techniques.

Compare l1 and l2 regularizations.

Example:
    $ python regularization.py --dataset ../datasets/animals

Attributes:
    dataset (str):
        The path to where our input image dataset resides on disk.
"""
import argparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader


def main():
    """Run various regularization techniques.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args = vars(args.parse_args())

    # grab the list of image paths
    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))

    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    preprocessor = SimplePreprocessor(32, 32)
    loader = SimpleDatasetLoader(preprocessors=[preprocessor])
    (data, labels) = loader.load(image_paths, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    # encode the labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                          test_size=0.25,
                                                          random_state=5)

    # loop over our set of regularizers
    for regularizer in (None, "l1", "l2"):
        # train a SGD classifier using a softmax loss function and the
        # specified regularization function for 10 epochs
        print("[INFO] training model with `{}` penalty".format(regularizer))
        model = SGDClassifier(loss="log", penalty=regularizer, max_iter=10,
                              learning_rate="constant", tol=1e-3, eta0=0.01, random_state=42)
        model.fit(train_x, train_y)
        # evaluate the classifier
        acc = model.score(test_x, test_y)
        print("[INFO] `{}` penalty accuracy: {:.2f}%".format(regularizer, acc * 100))


if __name__ == '__main__':
    main()
