# -*- coding: utf-8 -*-
"""Build ImageNet Dataset

1. Build the training set.
2. Build the validation set.
3. Construct the testing set by sampling the training set.
4. Loop over each of the sets.
5. Write the image path + corresponding class label to disk.

Examples:
    $ time python build_imagenet.py
"""
import json
import cv2
import numpy as np
import progressbar
from sklearn.model_selection import train_test_split
from config import imagenet_alexnet_config as config
from pyimagesearch.utils import ImageNetHelper


def main():
    """Build ImageNet dataset
    """
    # initialize the ImageNet helper and use it to construct the set of training and testing data
    print("[INFO] loading image paths...")
    imagenet_helper = ImageNetHelper(config)
    (train_paths, train_labels) = imagenet_helper.buildTrainingSet()
    (val_paths, val_labels) = imagenet_helper.buildValidationSet()

    # perform stratified sampling from the training set to construct a testing set
    print("[INFO] constructing splits...")
    split = train_test_split(train_paths,
                             train_labels,
                             test_size=config.NUM_TEST_IMAGES,
                             stratify=train_labels,
                             random_state=42)
    (train_paths, test_paths, train_labels, test_labels) = split

    # construct a list pairing the training, validation, and testing image paths
    # along with their corresponding labels and output list files
    datasets = [
        ("train", train_paths, train_labels, config.TRAIN_MX_LIST),
        ("val", val_paths, val_labels, config.VAL_MX_LIST),
        ("test", test_paths, test_labels, config.TEST_MX_LIST)]

    # initialize the list of Red, Green, and Blue channel averages
    (R, G, B) = ([], [], [])

    # loop over the dataset tuples
    for (data_type, paths, labels, output_path) in datasets:
        # open the output file for writing
        print("[INFO] building {}...".format(output_path))
        f = open(output_path, "w")

        # initialize the progress bar
        widgets = ["Building List: ",
                   progressbar.Percentage(), " ",
                   progressbar.Bar(), " ",
                   progressbar.ETA()]
        progress_bar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

        # loop over each of the individual images + labels
        for (i, (path, label)) in enumerate(zip(paths, labels)):
            # write the image index, label, and output path to file
            row = "\t".join([str(i), str(label), path])
            f.write("{}\n".format(row))

            # if we are building the training dataset, then compute the mean
            # of each channel in the image, then update the respective lists
            if data_type == "train":
                image = cv2.imread(path)
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

            # update the progress bar
            progress_bar.update(i)

        # close the output file
        progress_bar.finish()
        f.close()

    # construct a dictionary of averages, then serialize the means to a JSON file
    print("[INFO] serializing means...")
    dict_ave = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    f = open(config.DATASET_MEAN, "w")
    f.write(json.dumps(dict_ave))
    f.close()


if __name__ == "__main__":
    main()
