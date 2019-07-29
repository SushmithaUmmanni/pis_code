# -*- coding: utf-8 -*-
"""Train your first image classifier.

Train a k-NN classifier on the raw pixel intensities of the Animals dataset and use it to classify
unknown animal images.

Example:
    $ python knn.py --dataset ../datasets/animals

Attributes:
    dataset (str):
        The path to where our input image dataset resides on disk.
    neighbors (int, optional):
        The number of neighbors k to apply when using the k-NN algorithm.
    jobs (int, optional):
        the number of concurrent jobs to run when computing the distance
        between an input data point and the training set. A value of -1 will use all available
        cores on the processor.
"""
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader


def main():
    """Train a k-NN classifier.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args.add_argument("-k", "--neighbors", type=int, default=1,
                      help="# of nearest neighbors for classification")
    args.add_argument("-j", "--jobs", type=int, default=-1,
                      help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(args.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))

    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    preprocessor = SimplePreprocessor(32, 32)
    loader = SimpleDatasetLoader(preprocessors=[preprocessor])
    (data, labels) = loader.load(image_paths, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    # show some information on memory consumption of the images
    print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

    # encode the labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                          test_size=0.25,
                                                          random_state=42)

    # train and evaluate a k-NN classifier on the raw pixel intensities
    print("[INFO] evaluating k-NN classifier...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x), target_names=label_encoder.classes_))


if __name__ == '__main__':
    main()
