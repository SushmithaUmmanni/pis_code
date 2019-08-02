# -*- coding: utf-8 -*-
"""Implementing Ranked Accuracy

Demonstrate how to compute rank-1 and rank-5 accuracy for a dataset.

Examples:
    $ python inspect_model.py

Attributes:
    include-top (int, optional):
        Specify whether or not to include the top of a CNN.
"""
import argparse
from keras.applications import VGG16


def main():
    """Print indexes and layers.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--include-top", type=int, default=1,
                      help="whether or not to include top of CNN")
    args = vars(args.parse_args())

    # load the VGG16 network
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet", include_top=args["include_top"] > 0)
    print("[INFO] showing layers...")

    # loop over the layers in the network and display them to the console
    for (i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}".format(i, layer.__class__.__name__))


if __name__ == "__main__":
    main()
