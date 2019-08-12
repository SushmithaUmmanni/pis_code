
import json
import argparse
import keras.backend as K
import matplotlib
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
    aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())


    # initialize the image preprocessors
    sp = SimplePreprocessor(64, 64)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])
    iap = ImageToArrayPreprocessor()

    # initialize the training and validation dataset generators
    trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
                                    preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
    valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
                                preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)


    # if there is no specific model checkpoint supplied, then initialize
    # the network and compile the model
    if args["model"] is None:
        print("[INFO] compiling model...")
        model = DeeperGoogLeNet.build(width=64, height=64, depth=3,
                                    classes=config.NUM_CLASSES, reg=0.0002)
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


if __name__ == "__main__":
    main()
