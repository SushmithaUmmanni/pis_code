# -*- coding: utf-8 -*-
"""Evaluating an Ensemble

First, the best possible model has to be found. Then, usually 5-10 CNN models ara
used for an ensemble. Based on Jensen’s Inequality the performance of an ensemble
cannot be worse than a single model

Example:
    $ python test_ensemble.py --models models

Attributes:
    models (str):
        ppath to where our serialized network weights are stored on disk
"""
import os
import argparse
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10


def main():
    """Evaluate ensemble
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--models", required=True, help="path to models directory")
    args = vars(args.parse_args())

    # load the testing data, then scale it into the range [0, 1]
    (test_x, test_y) = cifar10.load_data()[1]
    test_x = test_x.astype("float") / 255.0

    # initialize the label names for the CIFAR-10 dataset
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    test_y = label_binarizer.fit_transform(test_y)

    # construct the path used to collect the models then initialize the models list
    model_paths = os.path.sep.join([args["models"], "*.model"])
    model_paths = list(glob.glob(model_paths))
    models = []

    # loop over the model paths, loading the model, and adding it to the list of models
    for (i, model_path) in enumerate(model_paths):
        print("[INFO] loading model {}/{}".format(i + 1, len(model_paths)))
        models.append(load_model(model_path))

    # initialize the list of predictions
    print("[INFO] evaluating ensemble...")
    predictions = []
    # loop over the models
    for model in models:
        # use the current model to make predictions on the testing data,
        # then store these predictions in the aggregate predictions list
        predictions.append(model.predict(test_x, batch_size=64))

    # average the probabilities across all model predictions, then show a classification report
    predictions = np.average(predictions, axis=0)
    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))


if __name__ == "__main__":
    main()
