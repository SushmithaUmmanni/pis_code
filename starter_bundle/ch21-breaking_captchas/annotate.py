# -*- coding: utf-8 -*-
"""Annotating and Creating Our Dataset

Annotate downloaded captcha images manually using image processing with openCV. Each number will be
extracted from the captcha, padded to a square, and based on the keystroke (label) saved to the
respective directory.

Example:
    $ python annotate.py --input downloads --annot dataset

Attributes:
    input (str):
        Path to input directory of images
    annot (str):
        Path to output directory of annotations
"""
import os
import argparse
import cv2
import imutils
from imutils import paths


def main():
    """Annotate images
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", required=True,
                      help="path to input directory of images")
    args.add_argument("-a", "--annot", required=True,
                      help="path to output directory of annotations")
    args = vars(args.parse_args())

    # grab the image paths then initialize the dictionary of character counts
    image_paths = list(paths.list_images(args["input"]))
    counts = {}

    # loop over the image paths
    for (i, image_paths) in enumerate(image_paths):
        # display an update to the user
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
        try:
            # load the image and convert it to grayscale, then pad the image to ensure
            # digits caught on the border of the image are retained
            image = cv2.imread(image_paths)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
            # threshold the image to reveal the digits
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # find contours in the image, keeping only the four largest ones
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
            # loop over the contours
            for contour in cnts:
                # compute the bounding box for the contour then extract the digit
                (x, y, w, h) = cv2.boundingRect(contour)
                roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
                # display the character, making it large enough for us to see,
                # then wait for a keypress
                cv2.imshow("ROI", imutils.resize(roi, width=28))
                key = cv2.waitKey(0)
                # if the '`' key is pressed, then ignore the character
                if key == ord("`"):
                    print("[INFO] ignoring character")
                    continue
                # grab the key that was pressed and construct the path the output directory
                key = chr(key).upper()
                dir_path = os.path.sep.join([args["annot"], key])
                # if the output directory does not exist, create it
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                # write the labeled character to file
                count = counts.get(key, 1)
                path = os.path.sep.join([dir_path, "{}.png".format(str(count).zfill(6))])
                cv2.imwrite(path, roi)
                # increment the count for the current key
                counts[key] = count + 1
        # we are trying to control-contour out of the script, so break from the loop (you still
        # need to press a key for the active window to trigger this)
        except KeyboardInterrupt:
            print("[INFO] manually leaving script")
            break
        # an unknown error has occurred for this particular image
        except BaseException:  # pylint: disable=broad-except
            print("[INFO] skipping image...")


if __name__ == '__main__':
    main()
