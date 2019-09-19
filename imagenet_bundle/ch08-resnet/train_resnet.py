# -*- coding: utf-8 -*-
"""Train ResNet with MXNet

This script can adapted to any other network architecture. You just need to change the following:
1. Change imports to properly set the configuration file and the network architecture.
2. Update the data_shape in the train_iter and val_iter (only if the network requires
   different input image spatial dimensions).
3. Update the SGD optimizer
4. Change the name of the model being initialized

Examples:
    $ python train_resnet.py --checkpoints checkpoints --prefix resnet
    $ python train_resnet.py --checkpoints checkpoints --prefix resnet --start-epoch 20
    $ python train_resnet.py --checkpoints checkpoints --prefix resnet --start-epoch 30

Attributes:
    checkpoints (int):
        name of the network
    prefix (str)
        name of the dataset
    start-epoch
        resume training from a specific previous epoch that has been serialized to disk
"""
import os
import argparse
import json
import logging
import mxnet as mx
from config import imagenet_resnet_config as config
from pyimagesearch.nn.mxconv import MxResNet


def main():
    """Train AlexNet with MXNet
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--checkpoints", required=True,
                      help="path to output checkpoint directory")
    args.add_argument("-p", "--prefix", required=True,
                      help="name of model prefix")
    args.add_argument("-s", "--start-epoch", type=int, default=0,
                      help="epoch to restart training at")
    args = vars(args.parse_args())

    # set the logging level and output file
    logging.basicConfig(level=logging.DEBUG,
                        filename="training_{}.log".format(args["start_epoch"]),
                        filemode="w")

    # load the RGB means for the training set, then determine the batch size
    means = json.loads(open(config.DATASET_MEAN).read())
    batch_size = config.BATCH_SIZE * config.NUM_DEVICES

    # construct the training image iterator
    train_iter = mx.io.ImageRecordIter(
        path_imgrec=config.TRAIN_MX_REC,
        data_shape=(3, 224, 224),
        batch_size=batch_size,
        rand_crop=True,
        rand_mirror=True,
        rotate=15,
        max_shear_ratio=0.1,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"],
        preprocess_threads=config.NUM_DEVICES * 2)

    # construct the validation image iterator
    val_iter = mx.io.ImageRecordIter(
        path_imgrec=config.VAL_MX_REC,
        data_shape=(3, 224, 224),
        batch_size=batch_size,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"])

    # initialize the optimizer
    opt = mx.optimizer.SGD(learning_rate=1e-1,
                           momentum=0.9,
                           wd=0.0001,
                           rescale_grad=1.0 / batch_size)

    # construct the checkpoints path, initialize the model argument and auxiliary parameters
    checkpoints_path = os.path.sep.join([args["checkpoints"], args["prefix"]])
    arg_params = None
    aux_params = None

    # if there is no specific model starting epoch supplied, then initialize the network
    if args["start_epoch"] <= 0:
        # build the LeNet architecture
        print("[INFO] building network...")
        model = MxResNet.build(config.NUM_CLASSES, (3, 4, 6, 3), (64, 256, 512, 1024, 2048))
        # otherwise, a specific checkpoint was supplied
    else:
        # load the checkpoint from disk
        print("[INFO] loading epoch {}...".format(args["start_epoch"]))
        model = mx.model.FeedForward.load(checkpoints_path, args["start_epoch"])
        # update the model and parameters
        arg_params = model.arg_params
        aux_params = model.aux_params
        model = model.symbol

    # compile the model
    model = mx.model.FeedForward(
        ctx=[mx.gpu(i) for i in range(0, config.NUM_DEVICES)],
        symbol=model,
        initializer=mx.initializer.MSRAPrelu(),
        arg_params=arg_params,
        aux_params=aux_params,
        optimizer=opt,
        num_epoch=100,
        begin_epoch=args["start_epoch"])

    # initialize the callbacks and evaluation metrics
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, 250)]
    epoch_end_callback = [mx.callback.do_checkpoint(checkpoints_path)]
    metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

    # train the network
    print("[INFO] training network...")
    model.fit(
        X=train_iter,
        eval_data=val_iter,
        eval_metric=metrics,
        batch_end_callback=batch_end_callbacks,
        epoch_end_callback=epoch_end_callback)


if __name__ == "__main__":
    main()
