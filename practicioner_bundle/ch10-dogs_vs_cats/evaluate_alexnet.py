# -*- coding: utf-8 -*-
"""Evaluate AlexNet on Kaggle: Dogs vs. Cats

To evaluate AlexNet on the testing set using both our standard method and over-sampling technique:
The over-sampling pre-processor is used at testing time to sample five regions of an input image
(the four corners + center area) along with their corresponding horizontal flips (for a total of
10 crops).

Example:
    $ python crop_accuracy.py
"""
import json
import numpy as np
import progressbar
from keras.models import load_model
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from config import dogs_vs_cats_config as config


def main():
    """Evaluate AlexNet on Cats vs. Dogs
    """
    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())
    # initialize the image preprocessors
    simple_preprocessor = SimplePreprocessor(227, 227)
    mean_preprocessor = MeanPreprocessor(means["R"], means["G"], means["B"])
    crop_preprocessor = CropPreprocessor(227, 227)
    image_to_array_preprocessor = ImageToArrayPreprocessor()

    # load the pretrained network
    print("[INFO] loading model...")
    model = load_model(config.MODEL_PATH)
    # initialize the testing dataset generator, then make predictions on the testing data
    print("[INFO] predicting on test data (no crops)...")
    test_gen = HDF5DatasetGenerator(
        config.TEST_HDF5,
        64,
        preprocessors=[simple_preprocessor, mean_preprocessor, image_to_array_preprocessor],
        classes=2,
    )

    predictions = model.predict_generator(test_gen.generator(), steps=test_gen.num_images // 64, max_queue_size=10)
    # compute the rank-1 and rank-5 accuracies
    (rank1, _) = rank5_accuracy(predictions, test_gen.database["labels"])
    print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
    test_gen.close()

    # re-initialize the testing set generator, this time excluding the `SimplePreprocessor`
    test_gen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[mean_preprocessor], classes=2)
    predictions = []
    # initialize the progress bar
    widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    progress_bar = progressbar.ProgressBar(maxval=test_gen.num_images // 64, widgets=widgets).start()
    # loop over a single pass of the test data
    # passes=1 to indicate the testing data only needs to be looped over once
    for (i, (images, _)) in enumerate(test_gen.generator(passes=1)):
        # loop over each of the individual images
        for image in images:
            # apply the crop preprocessor to the image to generate 10
            # separate crops, then convert them from images to arrays
            crops = crop_preprocessor.preprocess(image)
            crops = np.array([image_to_array_preprocessor.preprocess(c) for c in crops], dtype="float32")
            # make predictions on the crops and then average them
            # together to obtain the final prediction
            pred = model.predict(crops)
            predictions.append(pred.mean(axis=0))
        # update the progress bar
        progress_bar.update(i)
    # compute the rank-1 accuracy
    progress_bar.finish()
    print("[INFO] predicting on test data (with crops)...")
    (rank1, _) = rank5_accuracy(predictions, test_gen.database["labels"])
    print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
    test_gen.close()


if __name__ == "__main__":
    main()
