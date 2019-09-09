# $ python train_alexnet.py --checkpoints checkpoints --prefix alexnet
# $ python train_alexnet.py - -checkpoints checkpoints - -prefix alexnet \
#     - -start-epoch 50

# import the necessary packages
from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv import MxAlexNet
import mxnet as mx
import argparse
import logging
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
                    filename="training_{}.log".format(args["start_epoch"]),
                    filemode="w")

# load the RGB means for the training set, then determine the batch
# size
means = json.loads(open(config.DATASET_MEAN).read())
batchSize = config.BATCH_SIZE * config.NUM_DEVICES

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    rand_crop=True,
