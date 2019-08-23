# -*- coding: utf-8 -*-
"""Supporting Functions for Simple Object Detection Pipeline
"""
from keras.applications import imagenet_utils
import imutils


def sliding_window(image, step, roi_size):
    """Slide a window across the image

    Arguments:
        image {array} -- image to be processed
        step {int} -- step size (in px) for the sliding window
        roi_size {tuple} -- width and height (in px) of ROI, which will be extracted for
                            classification
    """
    for y in range(0, image.shape[0] - roi_size[1], step):
        for x in range(0, image.shape[1] - roi_size[0], step):
            # yield the current window
            yield (x, y, image[y:y + roi_size[1], x:x + roi_size[0]])


def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    """Yield multi-scale representation of an image

    Arguments:
        image {array} -- image to be processed

    Keyword Arguments:
        scale {float} -- controls how the image is resized at each layer (default: {1.5})
        min_size {tuple} -- minimum required width and height of the layer. (default: {(224, 224)})
    """
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        # yield the next image in the pyramid
        yield image


def classify_batch(model, batch_rois, batch_locs, labels,
                   min_probability=0.5, top=10, dims=(224, 224)):
    """Classify ROIs extracted from sliding window

    Arguments:
        model {obj} -- Keras model that we will be using for classification
        batch_rois {array} -- batch of ROIs, which will be classified
        batch_locs {list} -- (x;y)-coordinates of each ROI in batchROIs
        labels {dict} -- class labels dictionary

    Keyword Arguments:
        min_probability {float} -- minimum required probability for a classification
                                   to be considered a valid detection (default: {0.5})
        top {int} -- number of top-K predictions to be returned by the network (default: {10})
        dims {tuple} -- Spatial dimensions of the bounding box (default: {(224, 224)})
    """
    # pass our batch ROIs through our network and decode the predictions
    predictions_ = model.predict(batch_rois)
    predictions = imagenet_utils.decode_predictions(predictions_, top=top)
    # loop over the decoded predictions
    for i, _ in enumerate(predictions):
        for (_, label, prob) in predictions[i]:
            # filter out weak detections by ensuring the predicted probability is greater
            # than the minimum probability
            if prob > min_probability:
                # grab the coordinates of the sliding window for
                # the prediction and construct the bounding box
                (x, y) = batch_locs[i]
                box = (x, y, x + dims[0], y + dims[1])
                # grab the list of predictions for the label and
                # add the bounding box + probability to the list
                object_info = labels.get(label, [])
                object_info.append((box, prob))
                labels[label] = object_info
                # return the labels dictionary
                return labels
