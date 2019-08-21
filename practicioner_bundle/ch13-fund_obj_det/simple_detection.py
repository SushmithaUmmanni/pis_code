"""INPUT_SIZE: These are the width and height of our input --image. Our image is resized,
ignoring aspect ratio, to INPUT_SCALE prior to being fed through our neural network.
PYR_SCALE: The scale of our image pyramid. A smaller scale corresponds to more layers
generated while a larger scale implies fewer layers.
WIN_STEP: The step of our sliding window. The smaller the step, the more windows will
be evaluated, and consequently the slower our detector will run. The larger the step, fewer
windows will be evaluated and our detector will run faster. There is a tradeoff between
window size, speed, and accuracy. If your step is too large you may miss detections. If your
step is too small, your detector will take a long time to run.
ROI_SIZE: The input ROI size to our CNN as if we were performing classification.
BATCH_SIZE: The size of the batch to be built and passed through the CNN.
"""

# import the necessary packages
from pyimagesearch.utils.simple_obj_det import image_pyramid
from pyimagesearch.utils.simple_obj_det import sliding_window
from pyimagesearch.utils.simple_obj_det import classify_batch
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# initialize variables used for the object detection procedure
INPUT_SIZE = (350, 350)
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (224, 224)
BATCH_SIZE = 64


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


