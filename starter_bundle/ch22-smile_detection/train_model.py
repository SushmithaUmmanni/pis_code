# -*- coding: utf-8 -*-
"""Training the Smile CNN.

Train a CNN on the SMILES dataset to distinguish between a face that is smiling versus not smiling.

Example:
    $ python train_model.py --dataset ../datasets/SMILEsmileD --model output/lenet.hdf5

Attributes:
    dataset (str):
        Path to the SMILES dataset
    model (str):
        Path to where the serialized LeNet weights will be saved after training.
"""
import os
import argparse
import cv2
import imutils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv import LeNet


def main():
    """Train Smile CNN.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset of faces")
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args = vars(args.parse_args())
    # initialize the list of data and labels
    data = []
    labels = []

    # loop over the input images
    for image_path in sorted(list(paths.list_images(args["dataset"]))):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=28)
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = image_path.split(os.path.sep)[-3]
        label = "smiling" if label == "positives" else "not_smiling"
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    label_encoder = LabelEncoder().fit(labels)
    labels = np_utils.to_categorical(label_encoder.transform(labels), 2)

    # account for skew in the labeled data
    class_totals = labels.sum(axis=0)
    class_weight = class_totals.max() / class_totals

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data,
                                                          labels,
                                                          test_size=0.20,
                                                          stratify=labels,
                                                          random_state=42)
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=1, classes=2)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    model_fit = model.fit(train_x,
                          train_y,
                          validation_data=(test_x, test_y),
                          class_weight=class_weight,
                          batch_size=64,
                          epochs=15,
                          verbose=1)
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=64)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_encoder.classes_))

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 15), model_fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 15), model_fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 15), model_fit.history["acc"], label="acc")
    plt.plot(np.arange(0, 15), model_fit.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
