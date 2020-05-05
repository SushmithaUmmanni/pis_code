# -*- coding: utf-8 -*-
"""Implementation of backpropagation algorithm.

Attributes:
    W (list):
        List of weights.
    layers (list):
        A list of integers which represents the actual architecture of the feedforward network.
        For example, a value of [2, 2, 1] would imply that our first input layer has two nodes,
        our hidden layer has two nodes, and our final output layer has one node.
    alpha (int, optional):
        Learning rate value
"""
import numpy as np


class NeuralNetwork:
    """Implementation of backpropagation algorithm.
    """

    def __init__(self, layers, alpha=0.1):
        """Initialize the neural network.

        Args:
            layers (list): A list of integers which represents the actual architecture of the
                           feedforward network. For example, a value of [2, 2, 1] would imply that
                           our first input layer has two nodes, our hidden layer has two nodes, and
                           our final output layer has one node.
            alpha (float, optional): Learning rate value. Defaults to 0.1.
        """
        # initialize the list of weights matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha
        # start looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the number of nodes in each respective
            # layer together, adding an extra node for the bias
            weights = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(weights / np.sqrt(layers[i]))
            # the last two layers are a special case where the input
            # connections need a bias term but the output does not
            weights = np.random.randn(layers[-2] + 1, layers[-1])
            self.W.append(weights / np.sqrt(layers[-2]))

    def __repr__(self):
        """Construct and return a string that represents the network architecture

        Returns:
            str: network architecture
        """
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        """Commpute the sigmoid activation value for a given input.

        Args:
            x (array): input data point

        Returns:
            float: sigmoid activation value
        """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        """Compute derivate of the sigmoid function.

        Assumption:
        The input 'x' has already been passed through the sigmoid activation function.

        Args:
            x (array): sigmoid activation value

        Returns:
            float: derivative of the sigmoid activation value
        """
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        """Train the neural network

        Arguments:
            X {array} -- training data
            y {array} -- labels for each entry in X

        Keyword Arguments:
            epochs {int} -- number for epochs (default: {1000})
            display_update {int} -- number specifying how many N epochs taining progress will be
                                    printed to terminal (default: {100})
        """
        # insert a column of 1's as the last entry in the feature matrix -- this little trick
        # allows us to treat the bias as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
                # check to see if we should display a training update
                if epoch == 0 or (epoch + 1) % display_update == 0:
                    loss = self.calculate_loss(X, y)
                    print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        """Implement backpropagation algorithm.

        Arguments:
            x {float} -- An individual data point from our design matrix
            y {float} -- The corresponding class label
        """
        # construct our list of output activations for each layer as our data point flows through
        # the network; the first activation is a special case -- it's just the input
        # feature vector itself
        activations = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by taking the dot product between the
            # activation and the weight matrix -- this is called the "net input"
            # to the current layer
            net = activations[layer].dot(self.W[layer])
            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.sigmoid(net)
            # once we have the net output, add it to our list of activations
            activations.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the difference between our *prediction*
        # (the final output activation in the activations list) and the true target value
        error = activations[-1] - y
        # from here, we need to apply the chain rule and build our list of deltas `D`; the first
        # entry in the deltas is simply the error of the output layer times the derivative
        # of our activation function for the output value
        deltas = [error * self.sigmoid_deriv(activations[-1])]
        # once you understand the chain rule it becomes super easy to implement with a `for` loop
        # -- simply loop over the layers in reverse order (ignoring the last two since we
        # already have taken them into account)
        for layer in np.arange(len(activations) - 2, 0, -1):
            # the delta for the current layer is equal to the delta of the *previous layer* dotted
            # with the weight matrix of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = deltas[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(activations[layer])
            deltas.append(delta)
        # since we looped over our layers in reverse order we need to reverse the deltas
        deltas = deltas[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer activations with their
            # respective deltas, then multiplying this value by some small learning rate and adding
            # to our weight matrix -- this is where the actual "learning" takes place
            self.W[layer] += -self.alpha * activations[layer].T.dot(deltas[layer])

    def predict(self, X, add_bias=True):
        """Predict the label for given input data.

        Arguments:
            X {array} -- The data points we’ll be predicting class labels for.

        Keyword Arguments:
            add_bias {bool} -- A boolean indicating whether we need to add a column of 1’s to X to
                               perform the bias trick. (default: {True})

        Returns:
            float -- final class label prediction
        """
        # initialize the output prediction as the input features -- this value will be (forward)
        # propagated through the network to obtain the final prediction
        preds = np.atleast_2d(X)
        # check to see if the bias column should be added
        if add_bias:
            # insert a column of 1's as the last entry in the feature matrix (bias)
            preds = np.c_[preds, np.ones((preds.shape[0]))]
        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking the dot product between the
            # current activation value `preds` and the weight matrix associated with the current
            # layer, then passing this value through a nonlinear activation function
            preds = self.sigmoid(np.dot(preds, self.W[layer]))
        # return the predicted value
        return preds

    def calculate_loss(self, X, targets):
        """Compute loss based on calculated input data predictions

        Args:
            X (list): data points
            targets (list): ground-truth labels

        Returns:
            flaot: loss
        """
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        # return the loss
        return loss
