# -*- coding: utf-8 -*-
"""Train DeeperGoogLeNet

Example:
    $ python train.py --checkpoints output/checkpoints

    $ python train.py --checkpoints output/checkpoints \
                      --model output/checkpoints/epoch_25.hdf5 \
                      --start-epoch 25

    $ python train.py --checkpoints output/checkpoints \
                      --model output/checkpoints/epoch_35.hdf5 \
                      --start-epoch 35

Attributes:
    checkpoints (str):
        path to the output directory storing individual checkpoints for the DeeperGoogLeNet model
    model (str, optional):
        path to *specific* model checkpoint to load when restarting training
    start-epoch (int, optional)
        starting epoch number used when restarting from the model (default: 0)
"""
import json
import argparse
import matplotlib
import keras.backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import DeeperGoogLeNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from config import tiny_imagenet_config as config
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


def main():
    """Train DeeperGoogLeNet
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--checkpoints", required=True,
                      help="path to output checkpoint directory")
    args.add_argument("-m", "--model", type=str,
                      help="path to *specific* model checkpoint to load")
    args.add_argument("-s", "--start-epoch", type=int, default=0,
                      help="epoch to restart training at")
    args = vars(args.parse_args())

    # construct the training image generator for data augmentation
    augmentation = ImageDataGenerator(rotation_range=18,
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
                                     augmentation=augmentation,
                                     preprocessors=[simple_preprocessor,
                                                    mean_preprocessor,
                                                    image_to_array_preprocessor],
                                     classes=config.NUM_CLASSES)

    val_gen = HDF5DatasetGenerator(config.VAL_HDF5,
                                   64,
                                   preprocessors=[simple_preprocessor,
                                                  mean_preprocessor,
                                                  image_to_array_preprocessor],
                                   classes=config.NUM_CLASSES)

    # if there is no specific model checkpoint supplied,
    # then initialize the network and compile the model
    if args["model"] is None:
        print("[INFO] compiling model...")
        model = DeeperGoogLeNet.build(width=64,
                                      height=64,
                                      depth=3,
                                      classes=config.NUM_CLASSES,
                                      reg=0.0002)
        opt = Adam(1e-3)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # otherwise, load the checkpoint from disk
    else:
        print("[INFO] loading {}...".format(args["model"]))
        model = load_model(args["model"])
        # update the learning rate
        print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
        K.set_value(model.optimizer.lr, 1e-5)
        print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

    # construct the set of callbacks
    callbacks = [
        EpochCheckpoint(args["checkpoints"], every=5, start_at=args["start_epoch"]),
        TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH, start_at=args["start_epoch"])
        ]

    # train the network
    model.fit_generator(
        train_gen.generator(),
        steps_per_epoch=train_gen.num_images // 64,
        validation_data=val_gen.generator(),
        validation_steps=val_gen.num_images // 64,
        epochs=10,
        max_queue_size=10,
        callbacks=callbacks, verbose=1)

    # close the databases
    train_gen.close()
    val_gen.close()


if __name__ == "__main__":
    main()
