"""Visualize network architecture.

Example:
    $ python visualize_architecture.py
"""
from keras.utils import plot_model
from pyimagesearch.nn.conv import LeNet


def main():
    """Visualize network architecture.
    """
    # initialize LeNet and then write the network architecture
    # visualization graph to disk
    model = LeNet.build(28, 28, 1, 10)
    plot_model(model, to_file="lenet.png", show_shapes=True)


if __name__ == '__main__':
    main()
