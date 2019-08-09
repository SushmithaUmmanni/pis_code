# -*- coding: utf-8 -*-
"""Training a Logistic Regression Classifier

Use features extracted from the Cats vs. Dogs dataset to train a logistic regresstion classifier.

Examples:
    $ python train_model.py --database ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 \
                            --model dogs_vs_cats.pickle

Attributes:
    database (str):
        path to HDF5 database
    model (str):
        path to the logistic regression output model
    jobs (int, optional):
        # of jobs to run when tuning hyperparameters (default = -1)
"""
import argparse
import pickle
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def main():
    """Run the logistic regression classifier
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--database", required=True,
                      help="path HDF5 database")
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args.add_argument("-j", "--jobs", type=int, default=-1,
                      help="# of jobs to run when tuning hyperparameters")
    args = vars(args.parse_args())

    # open the HDF5 database for reading then determine the index of the training and testing
    # split, provided that this data was already shuffled *prior* to writing it to disk
    database = h5py.File(args["database"], "r")
    i = int(database["labels"].shape[0] * 0.75)
    # define the set of parameters that we want to tune then start a
    # grid search where we evaluate our model for each value of C
    print("[INFO] tuning hyperparameters...")
    params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
    model = GridSearchCV(LogisticRegression(solver="lbfgs", multi_class="auto"),
                         params, cv=3, n_jobs=args["jobs"])
    model.fit(database["features"][:i], database["labels"][:i])
    print("[INFO] best hyperparameters: {}".format(model.best_params_))
    # generate a classification report for the model
    print("[INFO] evaluating...")
    preds = model.predict(database["features"][i:])
    print(classification_report(database["labels"][i:],
                                preds,
                                target_names=database["label_names"]))
    # compute the raw accuracy with extra precision
    acc = accuracy_score(database["labels"][i:], preds)
    print("[INFO] score: {}".format(acc))
    # serialize the model to disk
    print("[INFO] saving model...")
    f = open(args["model"], "wb")
    f.write(pickle.dumps(model.best_estimator_))
    f.close()
    # close the database
    database.close()


if __name__ == "__main__":
    main()
