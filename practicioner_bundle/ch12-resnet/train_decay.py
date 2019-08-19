# -*- coding: utf-8 -*-
"""Train ResNet with Learning Rate Decay

Example:
    $ python train_decay.py --model output/resnet_tinyimagenet_decay.hdf5 \
                            --output output

Attributes:
     model (str):
        the path to the output file where ResNet will be serialized after training
    output (str):
        path to our output dir where we will store any plots, logs, etc
"""
import os
import sys
import argparse
import json
import matplotlib
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from config import tiny_imagenet_config as config
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")
# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# define the total number of epochs to train for along with the initial learning rate
NUM_EPOCHS = 75
INIT_LR = 1e-1


def poly_decay(epoch):
    """Implement custom learning rate schedule

    Arguments:
        epoch {int} -- current epoch

    Returns:
        float -- updated learning rate
    """
    # initialize the maximum number of epochs, base learning rate, and power of the polynomial
    max_epochs = NUM_EPOCHS
    base_learning_rate = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = base_learning_rate * (1 - (epoch / float(max_epochs))) ** power
    # return the new learning rate
    return alpha


def main():
    """Train Resnet on TinyImageNet dataset
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", required=True,
                      help="path to output model")
    args.add_argument("-o", "--output", required=True,
                      help="path to output directory (logs, plots, etc.)")
    args = vars(args.parse_args())

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=18,
                             zoom_range=0.15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())

    # initialize the image preprocessors
    simple_preprocessor = SimplePreprocessor(64, 64)
    mean_preprocessor = MeanPreprocessor(means["R"], means["G"], means["B"])
    image_to_array_preprocessor = ImageToArrayPreprocessor()

    # initialize the training and validation dataset generators
    train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5,
                                     64,
                                     augmentation=aug,
                                     preprocessors=[
                                         simple_preprocessor,
                                         mean_preprocessor,
                                         image_to_array_preprocessor],
                                     classes=config.NUM_CLASSES)

    val_gen = HDF5DatasetGenerator(config.VAL_HDF5,
                                   64,
                                   preprocessors=[
                                       simple_preprocessor,
                                       mean_preprocessor,
                                       image_to_array_preprocessor],
                                   classes=config.NUM_CLASSES)

    # construct the set of callbacks
    fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
    json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
    callbacks = [
        TrainingMonitor(fig_path, json_path=json_path),
        LearningRateScheduler(poly_decay)
        ]

    # initialize the optimizer and model (ResNet-56)
    print("[INFO] compiling model...")
    model = ResNet.build(64, 64, 3, config.NUM_CLASSES, (3, 4, 6),
                         (64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
    opt = SGD(lr=INIT_LR, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    model.fit_generator(train_gen.generator(),
                        steps_per_epoch=train_gen.num_images // 64,
                        validation_data=val_gen.generator(),
                        validation_steps=val_gen.num_images // 64,
                        epochs=NUM_EPOCHS,
                        max_queue_size=10,
                        callbacks=callbacks, verbose=1)

    # save the network to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    # close the databases
    train_gen.close()
    val_gen.close()


if __name__ == "__main__":
    main()
