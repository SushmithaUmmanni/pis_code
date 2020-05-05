# -*- coding: utf-8 -*-
"""Deep Dream implementation

Example:
    $ python deep_dream.py --image jp.jpg --output dream.png

Attributes:
    image (str):
        path to the input image
    output (str):
        path to the output dreamed image
"""
import argparse
import numpy as np
import cv2
from scipy import ndimage
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K

# define the dictionary that includes (1) the layers we are going
# to use for the dream and (2) their respective weights (i.e., the
# larger the weight, the more the layer contributes to the dream)
LAYERS = {
    "mixed2": 2.0,
    "mixed3": 0.5,
}

# define the number of octaves, octave scale, alpha (step for  gradient ascent) number
# of iterations, and max loss -- tweaking these values will produce different dreams
NUM_OCTAVE = 3  # number of octaves (resolutions) to be generated
OCTAVE_SCALE = 1.4  # defines the size of each successive octave
ALPHA = 0.001  # step size for gradient ascent
NUM_ITER = 50  # total number of gradient ascent operations
MAX_LOSS = 10.0  # early stopping criteria


def preprocess(image_path):
    """Preprocess image for inception network architecture

    Arguments:
        p {str} -- Path to an input image

    Returns:
        array -- return the preprocessed image
    """
    # load the input image, convert it to a Keras-compatible array,
    # expand the dimensions so we can pass it through the model, and
    # then finally preprocess it for input to the Inception network
    image = load_img(image_path)
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
    # "undo" the preprocessing done for Inception to bring the image back into the range [0, 255]
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


def eval_loss_and_gradients(X, model, loss, grads):
    """Fetch the loss and gradients given the input

    Arguments:
        X {tensor} -- This is our input tensor (i.e., the input image)

    Returns:
        tuple -- loss and gradients
    """
    # we now need to define a function that can retrieve the value of the
    # loss and gradients given an input image
    outputs = [loss, grads]
    fetch_loss_grads = K.function([model], outputs)

    # Fetch the loss and gradients given the input
    output = fetch_loss_grads([X])
    (loss, gradient) = (output[0], output[1])
    # return a tuple of the loss and gradients
    return (loss, gradient)


# todo: what is model => input tensor of the inception network. Is it a model or a tensor?


def gradient_ascent(X, loss, grads, model, iters, alpha, max_loss=-np.inf):
    """Compute gradient ascent

    This function is responsible for generating our actual dream:

    The consists of the following steps:
    1. We first loop over a number of iterations
    2. We compute the loss and gradients for our input
    3. And then finally apply the actual gradient ascent step

    Arguments:
        X {tensor} -- This is our input tensor (i.e., the input image)
        loss {}
        grads {}
        model {obj}
        iters {int} -- total number of iterations to run for
        alpha {float} -- step size/learning rate when applying gradient descent

    Keyword Arguments:
        max_loss {float} -- If our loss exceeds max_loss we terminate the gradient ascent process
                           early, preventing us from generating artifacts in our output image.
                           (default: {-np.inf})

    Returns:
        Tensor -- output of gradient ascent
    """
    # loop over our number of iterations
    for i in range(0, iters):
        # compute the loss and gradient
        (loss, gradient) = eval_loss_and_gradients(X, model, loss, grads)

        # if the loss is greater than the max loss, break from the
        # loop early to prevent strange effects
        if loss > max_loss:
            break

        # take a step
        print("[INFO] Loss at {}: {}".format(i, loss))
        X += alpha * gradient

    # return the output of gradient ascent
    return X


def main():
    """Run deep dream
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="path to input image")
    args.add_argument("-o", "--output", required=True, help="path to output dreamed image")
    args = vars(args.parse_args())

    # indicate that Keras *should not* be update the weights of any layer during the deep dream
    K.set_learning_phase(0)
    # load the (pre-trained) Inception model from disk, then grab a reference variable to the
    # input tensor of the model (which we'll then be using to perform our CNN hallucination)
    print("[INFO] loading inception network...")
    model = InceptionV3(weights="imagenet", include_top=False)
    dream = model.input

    # define our loss value, then build a dictionary that maps the *name* of each
    # layer inside of Inception to the actual *layer* object itself -- we'll need
    # this mapping when building the loss of the dream
    loss = K.variable(0.0)
    layer_map = {layer.name: layer for layer in model.layers}
    # loop over the layers that will be utilized in the dream
    for layer_name in LAYERS:
        # grab the output of the layer we will use for dreaming, then add the L2-norm
        # of the features to the layer to the loss (we use array slicing here to avoid
        # border artifacts caused by border pixels)
        x = layer_map[layer_name].output
        coeff = LAYERS[layer_name]
        scaling = K.prod(K.cast(K.shape(x), "float32"))
        loss += coeff * K.sum(K.square(x[:, 2:-2, 2:-2, :])) / scaling

    # compute the gradients of the dream with respect to loss and then normalize
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

    # load and preprocess the input image, then grab the (original) input height and width
    image = preprocess(args["image"])
    dims = image.shape[1:3]

    # in order to perform deep dreaming we need to build multiple scales of the input image
    # (i.e., set of images at lower and lower resolutions) -- this list stores the spatial
    # dimensions that we will be resizing our input image to
    octave_dims = [dims]
    # here we loop over the number of octaves (resolutions) we are going to generate
    for i in range(1, NUM_OCTAVE):
        # compute the spatial dimensions (i.e., width and height) for the
        # current octave, then update the dimensions list
        size = [int(d / (OCTAVE_SCALE ** i)) for d in dims]
        octave_dims.append(size)

    # reverse the octave dimensions list so that the *smallest*
    # dimensions are at the *front* of the list
    octave_dims = octave_dims[::-1]

    # clone the original image and then create a resized input image that
    # matches the smallest dimensions
    orig = np.copy(image)
    shrunk = resize_image(image, octave_dims[0])

    # loop over the octave dimensions from smallest to largest
    for (octave, size) in enumerate(octave_dims):
        # resize the image and then apply gradient ascent
        print("[INFO] starting octave {}...".format(octave))
        image = resize_image(image, size)
        image = gradient_ascent(image, loss, grads, model=dream, iters=NUM_ITER, alpha=ALPHA, max_loss=MAX_LOSS)

        # to compute the lost detail we need two images: (1) the shrunk
        # image that has been upscaled to the current octave and (2) the
        # original image that has been downscaled to the current octave
        upscaled = resize_image(shrunk, size)
        downscaled = resize_image(orig, size)
        # the lost detail is computed via a simple subtraction which we
        # immediately back in to the image we applied gradient ascent to
        lost = downscaled - upscaled
        image += lost
        # make the original image be the new shrunk image so we can repeat the process
        shrunk = resize_image(orig, size)

    # deprocess our dream and save it to disk
    image = deprocess(image)
    cv2.imwrite(args["output"], image)
