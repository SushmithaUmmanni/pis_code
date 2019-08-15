"""Visualize network architecture.

Visualize architecture of the LeNet model. In order to create the graphics you need to install the
following packages:

$ sudo apt-get install graphviz
$ pip install graphviz==0.5.2
$ pip install pydot==1.3.0

Example:
    $ python visualize_architecture.py
"""
from keras.utils import plot_model
from pyimagesearch.nn.conv import LeNet


def main():
    """Visualize network architecture.
    """
    # initialize LeNet and then write the network architecture visualization graph to disk
    model = LeNet.build(28, 28, 1, 10)
    plot_model(model, to_file="lenet.png", show_shapes=True)


if __name__ == '__main__':
    main()
