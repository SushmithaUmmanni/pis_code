# -*- coding: utf-8 -*-
"""Multiple Neural Style Transfer Experiments

Example:
    $ python generate_examples.py
"""
import os
import json
from imutils import paths
from keras.applications import VGG19
from pyimagesearch.nn.conv import NeuralStyle

# initialize the set of parameters/filenames
PARAMS = [
    "cw_1.0-sw_100.0-tvw_10.0",
    "cw_1.0-sw_1000.0-tvw_10.0",
    "cw_1.0-sw_100.0-tvw_100.0",
    "cw_1.0-sw_1000.0-tvw_1000.0",
    "cw_10.0-sw_100.0-tvw_10.0",
    "cw_10.0-sw_10.0-tvw_1000.0",
    "cw_10.0-sw_1000.0-tvw_1000.0",
    "cw_50.0-sw_10000.0-tvw_100.0",
    "cw_100.0-sw_1000.0-tvw_100.0"
]

# initialize the base dictionary
SETTINGS = {
    # initialize the path to the input (i.e., content) image,
    # style image, and path to the output directory
    "input_path": None,
    "style_path": None,
    "output_path": None,

    # define the CNN to be used style transfer, along with the
    # set of content layer and style layers, respectively
    "net": VGG19,
    "content_layer": "block4_conv2",
    "style_layers": ["block1_conv1", "block2_conv1",
                     "block3_conv1", "block4_conv1", "block5_conv1"],

    # store the content, style, and total variation weights, respectively
    "content_weight": None,
    "style_weight": None,
    "tv_weight": None,

    # number of iterations
    "iterations": 50,
}


def main():
    """Run neural style transfer for multiple parameters
    """
    # initialize the dictionary of completed runs
    completed = {}

    # if the completed dictionary exists, load it
    if os.path.exists("completed.json"):
        completed = json.loads(open("completed.json", "r").read())

    # grab the set of example images
    image_paths = list(paths.list_images("inputs"))

    # loop over the input images
    for input_path in image_paths:
        for style_path in image_paths:
            # if the two paths are equal, ignore them
            if input_path == style_path:
                continue

            # loop over the parameters
            for param in PARAMS:
                # parse out the content weight, style weight, and total
                # variation weight from the string
                params = param.split("-")
                grid = {
                    "content_weight": float(params[0].replace("cw_", "")),
                    "style_weight": float(params[1].replace("sw_", "")),
                    "tv_weight": float(params[2].replace("tvw_", "")),
                }

                # parse the filenames
                input_filename = input_path[input_path.rfind("/") + 1:]
                input_filename = input_filename[:input_filename.rfind(".")]
                style_filename = style_path[style_path.rfind("/") + 1:]
                style_filename = style_filename[:style_filename.rfind(".")]

                # construct the path to the output file
                output_filepath = "_".join([input_filename, style_filename, param])
                output_filepath = "{}.png".format(output_filepath)
                output_filepath = os.path.sep.join(["outputs", output_filepath])

                # update the settings dictionary
                SETTINGS["input_path"] = input_path
                SETTINGS["style_path"] = style_path
                SETTINGS["output_path"] = output_filepath
                SETTINGS["content_weight"] = grid["content_weight"]
                SETTINGS["style_weight"] = grid["style_weight"]
                SETTINGS["tv_weight"] = grid["tv_weight"]

                # build the key to the completed dictionary
                key = "{}_{}_{}".format(input_filename, style_filename, param)

                # if we have already performed this experiment, skip it
                if key in completed.keys():
                    print("[INFO] skipping: {}".format(key))
                    continue

                # perform neural style transfer with the current settings
                print("[INFO] starting: {}".format(key))
                neural_style = NeuralStyle(SETTINGS)
                neural_style.transfer()

                # indicate that the transfer completed successfully
                completed[key] = True

                # write the dictionary back out to disk
                print("[INFO] finished: {}".format(key))
                f = open("completed.json", "w")
                f.write(json.dumps(completed))
                f.close()


if __name__ == "__main__":
    main()
