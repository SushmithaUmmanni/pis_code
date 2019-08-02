
# -*- coding: utf-8 -*-
"""Implementing Ranked Accuracy

Demonstrate how to compute rank-1 and rank-5 accuracy for a dataset.

Examples:
    $ python rank_accuracy.py --db ../datasets/flowers17/hdf5/features.hdf5
                              --model ../chapter03-feature_extraction/flowers17.cpickle

    $ python rank_accuracy.py --db ../datasets/caltech-101/hdf5/features.hdf5
                              --model ../chapter03-feature_extraction/caltech101.cpickle

Attributes:
    db (str):
        Path HDF5 database
    model (path)
        Path to the pre-trained scikit-learn classifier residing on disk
"""
import argparse
import pickle
import h5py
from pyimagesearch.utils.ranked import rank5_accuracy


def main():
    """Display rank-1 and rank-5 accuracies
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--db", required=True,
                      help="path HDF5 database")
    args.add_argument("-m", "--model", required=True,
                      help="path to pre-trained model")
    args = vars(args.parse_args())

    # load the pre-trained model
    print("[INFO] loading pre-trained model...")
    model = pickle.loads(open(args["model"], "rb").read())
    # open the HDF5 database for reading then determine the index of the training and testing split,
    # provided that this data was already shuffled *prior* to writing it to disk
    db = h5py.File(args["db"], "r")
    i = int(db["labels"].shape[0] * 0.75)
    # make predictions on the testing set then compute the rank-1 and rank-5 accuracies
    print("[INFO] predicting...")
    preds = model.predict_proba(db["features"][i:])
    (rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])
    # display the rank-1 and rank-5 accuracies
    print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
    print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
    # close the database
    db.close()


if __name__ == "__main__":
    main()
