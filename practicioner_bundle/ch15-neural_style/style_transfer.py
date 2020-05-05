# -*- coding: utf-8 -*-
"""Neural Style Transfer Driver Script

Example:
    $ python style_transfer.py
"""
from keras.applications import VGG19
from pyimagesearch.nn.conv import NeuralStyle

# initialize the settings dictionary
SETTINGS = {
    # initialize the path to the input (i.e., content) image,
    # style image, and path to the output directory
    #
    # "input_path": "inputs/pablo_picasso.jpg",
    # "style_path": "inputs/popova_air_man_space.jpg",
    "input_path": "inputs/jp.jpg",
    "style_path": "inputs/mcescher.jpg",
    "output_path": "output",
    # define the CNN to be used for style transfer, along with
    # the set of content layer and style layers, respectively
    "net": VGG19,
    "content_layer": "block4_conv2",
    "style_layers": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
    # store the content, style, and total variation weights, respectively
    #
    # "content_weight": 1.0,
    # "style_weight": 100.0,
    # "tv_weight": 100.0,
    "content_weight": 1.0,
    "style_weight": 100.0,
    "tv_weight": 10.0,
    # number of iterations
    "iterations": 50,
}


def main():
    """Perform neural style transfer
    """
    neural_style = NeuralStyle(SETTINGS)
    neural_style.transfer()


if __name__ == "__main__":
    main()
