# -*- coding: utf-8 -*-
"""Implementation of basic gradient descent.

Manually created linearly separable dataset will be used to test the algorithm.

Example:
    $ python gradient_descent.py

Attributes:
    epochs (int, optional):
        Number of epochs.
    alpha (float, optional):
        Learning rate.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs


def sigmoid_activation(x):
    """Commpute the sigmoid activation value for a given input.

    Args:
        x (array): input data point

    Returns:
        float: sigmoid activation value
    """
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    """Compute derivate of the sigmoid function.

    Assumption:
    The input 'x' has already been passed through the sigmoid activation function.

    Args:
        x (array): sigmoid activation value

    Returns:
        float: derivative of the sigmoid activation value
    """
    return x * (1 - x)


def predict(X, W):
    """Predict the state of the input data points and weights.

    The function applies our sigmoid activation function and then thresholds it based on
    Predict whether the neuron is firing (1) or not (0).

    Args:
        X (array): input data points
        W (array): weights

    Returns:
        int: 0 (not activated) or 1 (activated)
    """
    # take the dot product between our features and weight matrix
    print("X", type(X.dot(W)))
    preds = sigmoid_activation(X.dot(W))
    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    # return the predictions
    return preds


def main():
    """ Run vanilla gradient descent algorithm.
    """
    # construct the argument parse and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-e", "--epochs", type=int, default=100,
                      help="# of epochs")
    args.add_argument("-a", "--alpha", type=float, default=0.01,
                      help="learning rate")
    args = vars(args.parse_args())

    # generate a 2-class classification problem with 1,000 data points,
    # where each data point is a 2D feature vector
    (X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                        cluster_std=1.5, random_state=1)
    y = y.reshape((y.shape[0], 1))
    # insert a column of 1's as the last entry in the feature
    # matrix - - this little trick allows us to treat the bias
    # as a trainable parameter within the weight matrix
    X = np.c_[X, np.ones((X.shape[0]))]
    # partition the data into training and testing splits using 50% of
    # the data for training and the remaining 50% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(X, y, test_size=0.5, random_state=42)

    # initialize our weight matrix and list of losses
    print("[INFO] training...")
    weights = np.random.randn(X.shape[1], 1)
    losses = []

    # loop over the desired number of epochs
    for epoch in np.arange(0, args["epochs"]):
        # take the dot product between our features `X` and the weight
        # matrix `W`, then pass this value through our sigmoid activation
        # function, thereby giving us our predictions on the dataset
        preds = sigmoid_activation(train_x.dot(weights))
        # now that we have our predictions, we need to determine the
        # `error`, which is the difference between our predictions and
        # the true values
        error = preds - train_y
        loss = np.sum(error ** 2)
        losses.append(loss)
        # the gradient descent update is the dot product between our
        # (1) features and (2) the error of the sigmoid derivative of
        # our predictions
        print('deriv', type(preds))
        derivate_error = error * sigmoid_deriv(preds)
        gradient = train_x.T.dot(derivate_error)
        # in the update stage, all we need to do is "nudge" the weight
        # matrix in the negative direction of the gradient (hence the
        # term "gradient descent" by taking a small step towards a set
        # of "more optimal" parameters
        weights += -args["alpha"] * gradient
        # check to see if an update should be displayed
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

    # evaluate our model
    print("[INFO] evaluating...")
    preds = predict(test_x, weights)
    print(classification_report(test_y, preds))

    # plot the (testing) classification data
    plt.style.use("ggplot")
    plt.figure()
    plt.title("Data")
    plt.scatter(test_x[:, 0], test_x[:, 1], marker="o", c=test_y[:, 0], s=30)

    # construct a figure that plots the loss over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args["epochs"]), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
