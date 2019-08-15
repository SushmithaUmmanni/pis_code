# -*- coding: utf-8 -*-
"""Evaluate model

Example:
    $ python rank_accuracy.py
"""
import json
from keras.models import load_model
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io import HDF5DatasetGenerator
from config import tiny_imagenet_config as config


def main():
    """Compute rank1 and rank5 accuracies
    """
    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())
    # initialize the image preprocessors
    simple_preprocessor = SimplePreprocessor(64, 64)
    mean_preprocessor = MeanPreprocessor(means["R"], means["G"], means["B"])
    image_to_array_preprocessor = ImageToArrayPreprocessor()
    # initialize the testing dataset generator
    test_gen = HDF5DatasetGenerator(config.TEST_HDF5,
                                    64,
                                    preprocessors=[simple_preprocessor,
                                                   mean_preprocessor,
                                                   image_to_array_preprocessor],
                                    classes=config.NUM_CLASSES)

    # load the pre-trained network
    print("[INFO] loading model...")
    model = load_model(config.MODEL_PATH)

    # make predictions on the testing data
    print("[INFO] predicting on test data...")
    predictions = model.predict_generator(test_gen.generator(),
                                          steps=test_gen.num_images // 64,
                                          max_queue_size=10)
    # compute the rank-1 and rank-5 accuracies
    (rank1, rank5) = rank5_accuracy(predictions, test_gen.database["labels"])
    print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
    print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
    # close the database
    test_gen.close()


if __name__ == "__main__":
    main()
