# -*- coding: utf-8 -*-
"""Automatically Downloading Example Images

Downloads the captcha images from a URL.

Example:
    $ python download_images.py --output downloads

Attributes:
    output (str):
        Path to output directory of images
    num-images (int, optional):
        Number of images to download
"""
import time
import os
import argparse
import requests


def main():
    """Download images
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", required=True, help="path to output directory of images")
    args.add_argument("-n", "--num-images", type=int, default=500, help="# of images to download")
    args = vars(args.parse_args())

    # initialize the URL that contains the captcha images that we will be downloading along with
    # the total number of images downloaded thus far
    url = "https://www.e-zpassny.com/vector/jcaptcha.do"
    total = 0

    # loop over the number of images to download
    for _ in range(0, args["num_images"]):
        try:
            # try to grab a new captcha image
            request = requests.get(url, timeout=60)
            # save the image to disk
            path = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
            f = open(path, "wb")
            f.write(request.content)
            f.close()
            # update the counter
            print("[INFO] downloaded: {}".format(path))
            total += 1
            # handle if any exceptions are thrown during the download process
        except BaseException:  # pylint: disable=broad-except
            print("[INFO] error downloading image...")

        # insert a small sleep to be courteous to the server
        time.sleep(0.1)


if __name__ == "__main__":
    main()
