# -*- coding: utf-8 -*-
"""Evaluate AlexNet with MXNet

Examples:
    $ python test_vggnet.py --checkpoints checkpoints --prefix vggnet --epoch 80

Attributes:
    checkpoints (int):
        path to our output checkpoints directory during the training process
    prefix (str):
        name of our actual CNN
    epoch (int):
        epoch of our network that we wish to use for evaluation. For example, if we stopped our
        training after epoch 100, then we would use the 100th epoch for evaluating our network
        on the testing data.
"""
import os
import argparse
import json
import mxnet as mx
from config import imagenet_alexnet_config as config


def main():
    """Evaluate AlexNet with MXNet
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--checkpoints", required=True,
                      help="path to output checkpoint directory")
    args.add_argument("-p", "--prefix", required=True,
                      help="name of model prefix")
    args.add_argument("-e", "--epoch", type=int, required=True,
                      help="epoch # to load")
    args = vars(args.parse_args())

    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())

    # construct the testing image iterator
    test_iter = mx.io.ImageRecordIter(
        path_imgrec=config.TEST_MX_REC,
        data_shape=(3, 227, 227),
        batch_size=config.BATCH_SIZE,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"])

    # load the checkpoint from disk
    print("[INFO] loading model...")
    checkpoints_path = os.path.sep.join([args["checkpoints"], args["prefix"]])
    model = mx.model.FeedForward.load(checkpoints_path, args["epoch"])

    # compile the model
    model = mx.model.FeedForward(
        ctx=[mx.gpu(0)],
        symbol=model.symbol,
        arg_params=model.arg_params,
        aux_params=model.aux_params)

    # make predictions on the testing data
    print("[INFO] predicting on test data...")
    metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
    (rank1, rank5) = model.score(test_iter, eval_metric=metrics)

    # display the rank-1 and rank-5 accuracies
    print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
    print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))


if __name__ == "__main__":
    main()
