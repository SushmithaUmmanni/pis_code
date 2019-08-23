# import the necessary packages
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy import ndimage
import numpy as np
import argparse
import cv2


def preprocess(p):
    """Preprocess image for inception network architecture

    Arguments:
        p {str} -- Path to an input image

    Returns:
        array -- return the preprocessed image
    """
    # load the input image, convert it to a Keras-compatible array,
    # expand the dimensions so we can pass it through the model, and
    # then finally preprocess it for input to the Inception network
    image = load_img(p)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def deprocess(image):
    """De-process a processed image

    Arguments:
        image {array} -- image, which was previous pre-processed to fit inception network

    Returns:
        array -- return the deprocessed image
    """
    # we are using "channels last" ordering so ensure the RGB
    # channels are the last dimension in the matrix
    image = image.reshape((image.shape[1], image.shape[2], 3))
    # "undo" the preprocessing done for Inception to bring the image
    # back into the range [0, 255]
    image /= 2.0
    image += 0.5
    image *= 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    # we have been processing images in RGB order; however, OpenCV
    # assumes images are in BGR order
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def resize_image(image, size):
    """Resize the image

    Arguments:
        image {array} -- image to be resized
        size {tuple} -- target size (w x h) of the image

    Returns:
        array -- resized image
    """
    resized = np.copy(image)
    zoom = (1, float(size[0]) / resized.shape[1], float(size[1]) / resized.shape[2], 1)
    resized = ndimage.zoom(resized, zoom, order=1)
    return resized


def eval_loss_and_gradients(X):


    # fetch the loss and gradients given the input
    output = fetchLossGrads([X])
    (loss, G) = (output[0], output[1])
    # return a tuple of the loss and gradients
    return (loss, G)


def gradient_ascent(X, iters, alpha, maxLoss=-np.inf):
    """Compute gradient ascent

    This function is responsible for generating our actual dream:

    The consists of the following steps:
    1. We first loop over a number of iterations
    2. We compute the loss and gradients for our input
    3. And then finally apply the actual gradient ascent step

    Arguments:
        X {tensor} -- This is our input tensor (i.e., the input image)
        iters {int} -- total number of iterations to run for
        alpha {float} -- step size/learning rate when applying gradient descent

    Keyword Arguments:
        maxLoss {float} -- If our loss exceeds maxLoss we terminate the gradient ascent process
                            early, preventing us from generating artifacts in our output image.
                            (default: {-np.inf})

    Returns:
        Tensor -- output of gradient ascent
    """
    # loop over our number of iterations
    for i in range(0, iters):
        # compute the loss and gradient
        (loss, G) = eval_loss_and_gradients(X)

        # if the loss is greater than the max loss, break from the
        # loop early to prevent strange effects
        if loss > maxLoss:
        break

        # take a step
        print("[INFO] Loss at {}: {}".format(i, loss))
        X += alpha * G

    # return the output of gradient ascent
    return X
