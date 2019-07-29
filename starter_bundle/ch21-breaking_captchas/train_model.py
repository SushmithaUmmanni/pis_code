# -*- coding: utf-8 -*-
"""Training the Captcha Breaker

Train LeNet model on the image captcha dataset.

Example:
    $ python test_model.py --input downloads --model output/lenet.hdf5

Attributes:
    input (str):
        The path to the input captcha images.
    model (str):
        The path to the serialized weights residing on disk.
"""
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from imutils import paths
from pyimagesearch.nn.conv import LeNet
from pyimagesearch.utils.captchahelper import preprocess


def main():
    """Train the Captcha breaker.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args = vars(args.parse_args())

    # initialize the data and labels
    data = []
    labels = []
    # loop over the input images
    for image_path in paths.list_images(args["dataset"]):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = preprocess(image, 28, 28)
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the
        # labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)
        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                              test_size=0.25,
                                                              random_state=42)
        # convert the labels from integers to vectors
        label_binarizer = LabelBinarizer().fit(train_y)
        train_y = label_binarizer.transform(train_y)
        test_y = label_binarizer.transform(test_y)
        # initialize the model
        print("[INFO] compiling model...")
        model = LeNet.build(width=28, height=28, depth=1, classes=9)
        opt = SGD(lr=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the network
        print("[INFO] training network...")
        model_fit = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                              batch_size=32, epochs=15, verbose=1)
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(test_x, batch_size=32)
        print(classification_report(test_y.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                    target_names=label_binarizer.classes_))
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
