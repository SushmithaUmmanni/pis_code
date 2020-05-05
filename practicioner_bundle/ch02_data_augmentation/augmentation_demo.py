# -*- coding: utf-8 -*-
"""Visualizing Data Augmentation.

Example:
    $ python augmentation_demo.py --image jemma.png --output output

Attributes:
    image (str):
        Path to the input image
    output (str):
        Path to the output directory
    prefix (str, optional):
        A string that will be prepended to the output image filename
"""
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


def main():
    """Run keras image augmentation
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="path to the input image")
    args.add_argument("-o", "--output", required=True, help="path to output directory to store augmentation examples")
    args.add_argument("-p", "--prefix", type=str, default="image", help="output filename prefix")
    args = vars(args.parse_args())

    # load the input image, convert it to a NumPy array,
    # and then reshape it to have an extra dimension
    print("[INFO] loading example image...")
    image = load_img(args["image"])
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    total = 0

    # construct the actual Python generator
    print("[INFO] generating images...")
    image_gen = aug.flow(image, batch_size=1, save_to_dir=args["output"], save_prefix=args["prefix"], save_format="jpg")
    # loop over examples from our image data augmentation generator
    for image in image_gen:
        # increment our counter
        total += 1
        # if we have reached 10 examples, break from the loop
        if total == 10:
            break


if __name__ == "__main__":
    main()
