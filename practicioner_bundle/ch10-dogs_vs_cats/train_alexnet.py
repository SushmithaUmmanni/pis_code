# -*- coding: utf-8 -*-
"""Training AlexNet on Kaggle: Dogs vs. Cats

Example:
    $ python train_alexnet.py
"""
import os
import json
import matplotlib
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import AlexNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from config import dogs_vs_cats_config as config
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


def main():
    """Train AlexNet on Dogs vs Cats
    """
    # construct the training image generator for data augmentation
    augmentation = ImageDataGenerator(rotation_range=20,
                                      zoom_range=0.15,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.15,
                                      horizontal_flip=True,
                                      fill_mode="nearest")

    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())
    # initialize the image preprocessors
    simple_preprocessor = SimplePreprocessor(227, 227)
    patch_preprocessor = PatchPreprocessor(227, 227)
    mean_preprocessor = MeanPreprocessor(means["R"], means["G"], means["B"])
    image_to_array_preprocessor = ImageToArrayPreprocessor()

    # initialize the training and validation dataset generators
    train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5,
                                     128,
                                     augmentation=augmentation,
                                     preprocessors=[patch_preprocessor,
                                                    mean_preprocessor,
                                                    image_to_array_preprocessor],
                                     classes=2)

    val_gen = HDF5DatasetGenerator(config.VAL_HDF5,
                                   128,
                                   preprocessors=[simple_preprocessor,
                                                  mean_preprocessor,
                                                  image_to_array_preprocessor],
                                   classes=2)
    # initialize the optimizer
    print("[INFO] compiling model...")
    opt = Adam(lr=1e-3)
    model = AlexNet.build(width=227, height=227, depth=3, classes=2, regularization=0.0002)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # construct the set of callbacks
    path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
    callbacks = [TrainingMonitor(path)]

    # train the network
    model.fit_generator(train_gen.generator(),
                        steps_per_epoch=train_gen.num_images // 128,
                        validation_data=val_gen.generator(),
                        validation_steps=val_gen.num_images // 128,
                        epochs=75,
                        max_queue_size=10,
                        callbacks=callbacks, verbose=1)

    # save the model to file
    print("[INFO] serializing model...")
    model.save(config.MODEL_PATH, overwrite=True)

    # close the HDF5 datasets
    train_gen.close()
    val_gen.close()


if __name__ == "__main__":
    main()
