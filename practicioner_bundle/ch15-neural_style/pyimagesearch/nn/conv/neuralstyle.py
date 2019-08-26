# -*- coding: utf-8 -*-
"""Multiple Neural Style Transfer Experiments

Based on Gatys et.al., 2015, https://arxiv.org/abs/1508.06576

Example:
    $ python generate_examples.py

Attributes:
    settings (dict):
        setting dictionary
"""
import os
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b


class NeuralStyle:
    """Implementation of neural style architecture
    """
    def __init__(self, settings):
        # store the settings dictionary
        self.settings = settings

        # grab the dimensions of the input image
        (w, h) = load_img(self.settings["input_path"]).size
        self.dims = (h, w)

        # load content image and style images, forcing the dimensions of our input image
        self.content = self.preprocess(settings["input_path"])
        self.style = self.preprocess(settings["style_path"])
        self.content = K.variable(self.content)
        self.style = K.variable(self.style)

        # allocate memory of our output image, then combine the content, style, and output into
        # a single tensor so they can be fed through the network
        self.output = K.placeholder((1, self.dims[0], self.dims[1], 3))
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        # load our model from disk
        print("[INFO] loading network...")
        self.model = self.settings["net"](weights="imagenet",
                                          include_top=False,
                                          input_tensor=self.input)

        # build a dictionary that maps the *name* of each layer
        # inside the network to the actual layer *output*
        layer_map = {l.name: l.output for l in self.model.layers}

        # extract features from the content layer, then extract the activations from the style
        # image (index 0) and the output image (index 2) -- these will serve as our style
        # features and output features from the *content* layer
        content_features = layer_map[self.settings["content_layer"]]
        style_features = content_features[0, :, :, :]
        output_features = content_features[2, :, :, :]

        # compute the feature reconstruction loss, weighting it appropriately
        content_loss = self.featureReconLoss(style_features, output_features)
        content_loss *= self.settings["content_weight"]

        # initialize our style loss along with the value used to weight each
        # style layer (in proportion to the total number of style layers
        style_loss = K.variable(0.0)
        weight = 1.0 / len(self.settings["style_layers"])
        # loop over the style layers
        for layer in self.settings["style_layers"]:
            # grab the current style layer and use it to extract the style
            # features and output features from the *style layer*
            style_output = layer_map[layer]
            style_features = style_output[1, :, :, :]
            output_features = style_output[2, :, :, :]

            # compute the style reconstruction loss as we go
            T = self.styleReconLoss(style_features, output_features)
            style_loss += (weight * T)

        # finish computing the style loss, compute the total  variational loss,
        # and then compute the total loss that combines all three
        style_loss *= self.settings["style_weight"]
        tv_loss = self.settings["tv_weight"] * self.tv_loss(self.output)
        total_loss = content_loss + style_loss + tv_loss

        # compute the gradients out of the output image with respect to loss
        grads = K.gradients(total_loss, self.output)
        outputs = [total_loss]
        outputs += grads

        # the implementation of L-BFGS we will be using requires that our loss and
        # gradients be *two separate functions* so here we create a Keras function
        # that can compute both the loss and gradients together and then return each
        # separately using two different class methods
        self.loss_and_grads = K.function([self.output], outputs)

    def preprocess(self, image_path):
        """Pre-process the input image

        Arguments:
            image_path {str} -- path to the input image

        Returns:
            array -- processed image
        """
        # load the input image (while resizing it to the desired dimensions) and preprocess it
        image = load_img(image_path, target_size=self.dims)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        # return the preprocessed image
        return image

    def deprocess(self, image):
        """De-process the output of our neural style transfer algorithm

        The image will pre de-processed based on the reverse of the pre-processing used for VGG16
        and VGG19. If you’re not using either VGG16 or VGG19 you’ll need to modify the deprocess
        function to perform the inverse of the pre-processing required by your network.

        Arguments:
            image {array} -- image to be de-processed

        Returns:
            array -- de-processed image
        """
        # reshape the image, then reverse the zero-centering by *adding*
        # back in the mean values across the ImageNet training set
        image = image.reshape((self.dims[0], self.dims[1], 3))
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.680
        # clip any values falling outside the range [0, 255] and
        # convert the image to an unsigned 8-bit integer
        image = np.clip(image, 0, 255).astype("uint8")
        # return the deprocessed image
        return image

    # todo: what type is the gram matrix

    def compute_gram_matrix(self, X):
        """Computer gram matrix

        The Gram matrix is the dot product between the input vectors and their transpose

        Arguments:
            X {Tensor} -- [description]

        Returns:
            Tensor -- [description]
        """
        # the gram matrix is the dot product between the input
        # vectors and their respective transpose
        features = K.permute_dimensions(X, (2, 0, 1))
        features = K.batch_flatten(features)
        features = K.dot(features, K.transpose(features))
        # return the gram matrix
        return features

    def featureReconLoss(self, style_features, output_features):
        """[summary]

        The content-loss is the L2 norm (sum of squared differences) between the features of our
        input image and the features of the target, output image. By minimizing the L2 norm of
        these activations we can force our output image to have similar structural content
        (but not necessarily similar style) as our input image.

        Arguments:
            style_features {[type]} -- [description]
            output_features {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        # the feature reconstruction loss is the squared error
        # between the style features and output output features
        return K.sum(K.square(output_features - style_features))

    def styleReconLoss(self, style_features, output_features):
        # compute the style reconstruction loss where A is the gram
        # matrix for the style image and G is the gram matrix for the
        # generated image
        A = self.compute_gram_matrix(style_features)
        G = self.compute_gram_matrix(output_features)
        # compute the scaling factor of the style loss, then finish
        # computing the style reconstruction loss
        scale = 1.0 / float((2 * 3 * self.dims[0] * self.dims[1]) ** 2)
        loss = scale * K.sum(K.square(G - A))
        # return the style reconstruction loss
        return loss

    def tv_loss(self, X):
        # the total variational loss encourages spatial smoothness in
        # the output page -- here we avoid border pixels to avoid
        # artifacts
        (h, w) = self.dims
        A = K.square(X[:, :h - 1, :w - 1, :] - X[:, 1:, :w - 1, :])
        B = K.square(X[:, :h - 1, :w - 1, :] - X[:, :h - 1, 1:, :])
        loss = K.sum(K.pow(A + B, 1.25))
        # return the total variational loss
        return loss

    def transfer(self, maxEvals=20):
        # generate a random noise image that will serve as a placeholder array,
        # slowly modified as we run L-BFGS toapply style transfer
        X = np.random.uniform(0, 255, (1, self.dims[0], self.dims[1], 3)) - 128
        # start looping over the desired number of iterations
        for i in range(0, self.settings["iterations"]):
            # run L-BFGS over the pixels in our generated image to minimize the neural style loss
            print("[INFO] starting iteration {} of {}...".format(
                i + 1, self.settings["iterations"]))
            (X, loss, _) = fmin_l_bfgs_b(self.loss,
                                         X.flatten(),
                                         fprime=self.grads,
                                         maxfun=maxEvals)
            print("[INFO] end of iteration {}, loss: {:.4e}".format(i + 1, loss))

            # deprocess the generated image and write it to disk
            image = self.deprocess(X.copy())
            image_path = os.path.sep.join([self.settings["output_path"], "iter_{}.png".format(i)])
            cv2.imwrite(image_path, image)

    def loss(self, X):
        # extract the loss value
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        loss_value = self.loss_and_grads([X])[0]
        # return the loss
        return loss_value

    def grads(self, X):
        # compute the loss and gradients
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        output = self.loss_and_grads([X])
        # extract and return the gradient values
        return output[1].flatten().astype("float64")
