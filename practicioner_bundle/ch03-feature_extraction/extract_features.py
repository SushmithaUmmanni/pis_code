# import the necessary packages
import os
import argparse
import numpy as np
import progressbar
import random
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from .pyimagesearch.io import HDF5DatasetWriter
from imutils import paths


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args["batch_size"]
# grab the list of images that we'll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
# extract the class labels from the image paths then encode the
# labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
# initialize the HDF5 dataset writer, then store the class label
# names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),
                            args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
                               widgets=widgets).start()
# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
