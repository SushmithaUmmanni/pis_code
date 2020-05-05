# -*- coding: utf-8 -*-
"""Classifying Images with Pre-trained ImageNet CNNs.

Let’s learn how to classify images with pre-trained Convolutional Neural Networks
using the Keras library.

Example:
    $ python imagenet_pretrained.py --image example_images/example_01.jpg --model vgg16
    $ python imagenet_pretrained.py --image example_images/example_02.jpg --model vgg19
    $ python imagenet_pretrained.py --image example_images/example_03.jpg --model inception
    $ python imagenet_pretrained.py --image example_images/example_04.jpg --model xception
    $ python imagenet_pretrained.py --image example_images/example_05.jpg --model resnet

Attributes:
    image (str):
        The path to our input image.
    model (str, optional):
        The name of the pre-trained network to use.
"""
import argparse
import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception  # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# define a dictionary that maps model names to their classes inside Keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,  # TensorFlow ONLY
    "resnet": ResNet50,
}


def main():
    """Classify images with a pre-trained neural network.

    Raises:
        AssertionError: The --model command line argument should be a key in the `MODELS` dictionary
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="path to the input image")
    args.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
    args = vars(args.parse_args())

    # ensure a valid model name was supplied via command line argument
    if args["model"] not in MODELS.keys():
        raise AssertionError("The --model command line argument should " "be a key in the `MODELS` dictionary")

    # initialize the input image shape (224x224 pixels) along with the pre-processing function
    # (this might need to be changed based on which model we use to classify our image)
    input_shape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    # if we are using the InceptionV3 or Xception networks, then we need to set the input shape
    # to (299x299) [rather than (224x224)] and use a different image processing function
    if args["model"] in ("inception", "xception"):
        input_shape = (299, 299)
        preprocess = preprocess_input

    # load the network weights from disk (NOTE: if this is the first time you are running this
    # script for a given network, the weights will need to be downloaded first -- depending on
    # which network you are using, the weights can be 90-575MB, so be patient; the weights
    # will be cached and subsequent runs of this script will be *much* faster)
    print("[INFO] loading {}...".format(args["model"]))
    network = MODELS[args["model"]]
    model = network(weights="imagenet")
    # load the input image using the Keras helper utility while ensuring the image is resized
    # to `input_shape`, the required input dimensions for the ImageNet pre-trained network
    print("[INFO] loading and pre-processing image...")
    image = load_img(args["image"], target_size=input_shape)
    image = img_to_array(image)
    # our input image is now represented as a NumPy array of shape
    # (input_shape[0], input_shape[1], 3) however we need to expand the
    # dimension by making the shape (1, input_shape[0], input_shape[1], 3)
    # so we can pass it through the network. The 1 has to be specified
    # so we can train in batches.
    image = np.expand_dims(image, axis=0)
    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
    image = preprocess(image)
    # classify the image
    print("[INFO] classifying image with '{}'...".format(args["model"]))
    predictions = model.predict(image)
    prediction = imagenet_utils.decode_predictions(predictions)
    # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
    for (i, (_, label, prob)) in enumerate(prediction[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        # load the image via OpenCV, draw the top prediction on the image,
        # and display the image to our screen
        orig = cv2.imread(args["image"])
        (_, label, prob) = prediction[0][0]
        cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
