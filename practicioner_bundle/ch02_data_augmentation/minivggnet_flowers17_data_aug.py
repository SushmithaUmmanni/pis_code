# -*- coding: utf-8 -*-
"""Flowers-17: With Data Augmentation.

Investigate the effect of aumentation: Classify Flowers-17 dataset with augmentation.

Example:
    $ python minivggnet_flowers17_data_aug.py --dataset ../datasets/flowers17/images

Attributes:
    dataset (str):
        Path to the Flowers-17 dataset
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import MiniVGGNet


def main():
    """Run image classification
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args = vars(args.parse_args())

    # grab the list of images that we'll be describing, then extract
    # the class label names from the image paths
    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))
    class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
    class_names = [str(x) for x in np.unique(class_names)]

    # initialize the image preprocessors
    aspect_aware_preprocessor = AspectAwarePreprocessor(64, 64)
    image_to_array_preprocessor = ImageToArrayPreprocessor()

    # load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[aspect_aware_preprocessor,
                                             image_to_array_preprocessor])
    (data, labels) = sdl.load(image_paths, verbose=500)
    data = data.astype("float") / 255.0

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data,
                                                          labels,
                                                          test_size=0.25,
                                                          random_state=42)

    # convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.05)
    model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    model_fit = model.fit_generator(aug.flow(train_x, train_y, batch_size=32),
                                    validation_data=(test_x, test_y),
                                    steps_per_epoch=len(train_x) // 32,
                                    epochs=100,
                                    verbose=1)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=class_names))

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


if __name__ == "__main__":
    main()
