# -*- coding: utf-8 -*-
"""Simple Object Detection With a CNN

Example:
    $ python simple_detection.py - -image beagle.png - -confidence 0.75

Attributes:
    image (str):
        path to the input image
    confidende (float, default: 0.5):
        minimum probability to filter weak detections
"""
import argparse
import time
import numpy as np
import cv2
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimagesearch.utils.simple_obj_det import image_pyramid
from pyimagesearch.utils.simple_obj_det import sliding_window
from pyimagesearch.utils.simple_obj_det import classify_batch

# These are the width and height of our input --image. Our image is resized,
# ignoring aspect ratio, to INPUT_SCALE prior to being fed through our neural network.
INPUT_SIZE = (350, 350)
# The scale of our image pyramid. A smaller scale corresponds to more layers
# generated while a larger scale implies fewer layers.
PYR_SCALE = 1.5
# The step of our sliding window. The smaller the step, the more windows will
# be evaluated, and consequently the slower our detector will run. The larger the step, fewer
# windows will be evaluated and our detector will run faster. There is a tradeoff between
# window size, speed, and accuracy. If your step is too large you may miss detections. If your
# step is too small, your detector will take a long time to run.
WIN_STEP = 16
# The input ROI size to our CNN as if we were performing classification.
ROI_SIZE = (224, 224)
# The size of the batch to be built and passed through the CNN.
BATCH_SIZE = 64


def main():
    """Run simple object detection
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="path to the input image")
    args.add_argument(
        "-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections"
    )
    args = vars(args.parse_args())

    # load our the network weights from disk
    print("[INFO] loading network...")
    model = ResNet50(weights="imagenet", include_top=True)
    # initialize the object detection dictionary which maps class labels
    # to their predicted bounding boxes and associated probability
    labels = {}
    # load the input image from disk and grab its dimensions
    orig = cv2.imread(args["image"])
    # resize the input image to be a square
    resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    # initialize the batch ROIs and (x, y)-coordinates
    batch_rois = None
    batch_locs = []
    # start the timer
    print("[INFO] detecting objects...")
    start = time.time()
    # loop over the image pyramid
    for image in image_pyramid(resized, scale=PYR_SCALE, min_size=ROI_SIZE):
        # loop over the sliding window locations
        for (x, y, roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # take the ROI and pre-process it so we can later classify the region with Keras
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = imagenet_utils.preprocess_input(roi)
            # if the batch is None, initialize it
            if batch_rois is None:
                batch_rois = roi
            # otherwise, add the ROI to the bottom of the batch
            else:
                batch_rois = np.vstack([batch_rois, roi])
            # add the (x, y)-coordinates of the sliding window to the batch
            batch_locs.append((x, y))
            # check to see if our batch is full
            if len(batch_rois) == BATCH_SIZE:
                # classify the batch, then reset the batch ROIs and (x, y)-coordinates
                labels = classify_batch(model, batch_rois, batch_locs, labels, min_probability=args["confidence"])
                # reset the batch ROIs and (x, y)-coordinates
                batch_rois = None
                batch_locs = []

    # check to see if there are any remaining ROIs that still need to be classified
    if batch_rois is not None:
        labels = classify_batch(model, batch_rois, batch_locs, labels, min_probability=args["confidence"])
    # show how long the detection process took
    end = time.time()
    print("[INFO] detections took {:.4f} seconds".format(end - start))
    # loop over the labels for each of detected objects in the image
    for k in labels.keys():
        # clone the input image so we can draw on it
        clone = resized.copy()
        # loop over all bounding boxes for the label and draw them on the image
        for (box, _) in labels[k]:
            (x, y, w, h) = box
            cv2.rectangle(clone, (x, y), (w, h), (0, 255, 0), 2)

        # show the image *without* apply non-maxima suppression
        cv2.imshow("Without NMS", clone)
        clone = resized.copy()
        # grab the bounding boxes and associated probabilities for each detection, then apply
        # non-maxima suppression to suppress weaker, overlapping detections
        boxes = np.array([p[0] for p in labels[k]])
        probability = np.array([p[1] for p in labels[k]])
        boxes = non_max_suppression(boxes, probability)
        # loop over the bounding boxes again, this time only drawing the
        # ones that were *not* suppressed
        for (x, y, w, h) in boxes:
            cv2.rectangle(clone, (x, y), (w, h), (0, 0, 255), 2)
        # show the output image
        print("[INFO] {}: {}".format(k, len(boxes)))
        cv2.imshow("With NMS", clone)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
