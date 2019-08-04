# -*- coding: utf-8 -*-
"""Fine-tuning from Start to Finish.

Example:
    $ python finetune_flowers17.py --dataset ../datasets/flowers17/images --model flowers17.model

Attributes:
    dataset (str):
        Path to the input dataset
    model (str):
        Path to our output serialized weights after training
"""
import os
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import FCHeadNet


def main():
    """Fine tune VGG16
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset")
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args = vars(args.parse_args())

    # construct the image generator for data augmentation
    augmentation = ImageDataGenerator(rotation_range=30,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode="nearest")

    # grab the list of images that we'll be describing, then extract
    # the class label names from the image paths
    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))
    class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
    class_names = [str(x) for x in np.unique(class_names)]

    # initialize the image preprocessors
    aspect_aware_preprocessor = AspectAwarePreprocessor(224, 224)
    image_to_array_preprocessor = ImageToArrayPreprocessor()

    # load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
    simple_dataset_loader = SimpleDatasetLoader(preprocessors=[aspect_aware_preprocessor,
                                                               image_to_array_preprocessor])
    (data, labels) = simple_dataset_loader.load(image_paths, verbose=500)
    data = data.astype("float") / 255.0

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data,
                                                          labels,
                                                          test_size=0.25,
                                                          random_state=42)
    # convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().transform(test_y)

    # load the VGG16 network, ensuring the head FC layer sets are left off
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)))

    # initialize the new head of the network, a set of FC layers followed by a softmax classifier
    head_model = FCHeadNet.build(base_model, len(class_names), 256)

    # place the head FC model on top of the base model -- this will
    # become the actual model we will train
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they
    # will *not* be updated during the training process
    for layer in base_model.layers:
        layer.trainable = False

    # compile our model (this needs to be done after our setting our layers to being non-trainable
    print("[INFO] compiling model...")
    opt = RMSprop(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the head of the network for a few epochs (all other  layers are frozen) -- this will
    # allow the new FC layers to start to become initialized with actual "learned" values
    # versus pure random
    print("[INFO] training head...")
    model.fit_generator(augmentation.flow(train_x, train_y, batch_size=32),
                        validation_data=(test_x, test_y), epochs=25,
                        steps_per_epoch=len(train_x) // 32, verbose=1)

    # evaluate the network after initialization
    print("[INFO] evaluating after initialization...")
    predictions = model.predict(test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=class_names))

    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    for layer in base_model.layers[15:]:
        layer.trainable = True

    # for the changes to the model to take affect we need to recompile
    # the model, this time using SGD with a *very* small learning rate
    print("[INFO] re-compiling model...")
    opt = SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the model again, this time fine-tuning *both* the final set
    # of CONV layers along with our set of FC layers
    print("[INFO] fine-tuning model...")
    model.fit_generator(augmentation.flow(train_x, train_y, batch_size=32),
                        validation_data=(test_x, test_y), epochs=100,
                        steps_per_epoch=len(train_x) // 32, verbose=1
                        )
    # evaluate the network on the fine-tuned model
    print("[INFO] evaluating after fine-tuning...")
    predictions = model.predict(test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=class_names))

    # save the model to disk
    print("[INFO] serializing model...")
    model.save(args["model"])


if __name__ == "__main__":
    pass
