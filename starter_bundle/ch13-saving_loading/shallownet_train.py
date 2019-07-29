# -*- coding: utf-8 -*-
"""Serialize model to disk.

Train and serialize a custom keras model to disk. The model will be used to classify images from
the Animals dataset.

Example:
    $ python shallownet_train.py --dataset ../datasets/animals --model shallownet_weights.hdf5

Attributes:
    dataset (str):
        The path to where our input image dataset resides on disk.
    model (str):
        The path to the pre-trained model.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet


def main():
    """Serialize a trained model to disk.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args = vars(args.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessors
    simple_preprocessor = SimplePreprocessor(32, 32)
    image_to_array_preprocessor = ImageToArrayPreprocessor()
    # load the dataset from disk then scale the raw pixel intensities
    # to the range [0, 1]
    dataset_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor,
                                                        image_to_array_preprocessor])
    (data, labels) = dataset_loader.load(image_paths, verbose=500)
    data = data.astype("float") / 255.0
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                          test_size=0.25, random_state=42)
    # convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.005)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train the network
    print("[INFO] training network...")
    model_fit = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          batch_size=32, epochs=100, verbose=1)
    # save the network to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=["cat", "dog", "panda"]))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 100), model_fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), model_fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), model_fit.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 100), model_fit.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
