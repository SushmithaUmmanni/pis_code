# -*- coding: utf-8 -*-
"""Train and evaluate LeNet on MNIST dataset.

1. Loading the MNIST dataset from disk.
2. Instantiating the LeNet architecture.
3. Training LeNet.
4. Evaluating network performance.

Example:
    $ python lenet_mnist.py --dataset ../datasets/animals
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import LeNet


def main():
    """Train and evaluate LeNet on MNIST dataset.
    """
    # grab the MNIST dataset (if this is your first time using this
    # dataset then the 11MB download may take a minute)
    print("[INFO] accessing MNIST...")
    ((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()

    # if we are using "channels first" ordering, then reshape the
    # design matrix such that the matrix is:
    # num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
        train_data = train_data.reshape((train_data.shape[0], 1, 28, 28))
        test_data = test_data.reshape((test_data.shape[0], 1, 28, 28))
        # otherwise, we are using "channels last" ordering, so the design
        # matrix shape should be: num_samples x rows x columns x depth
    else:
        train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
        test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

    # scale data to the range of [0, 1]
    train_data = train_data.astype("float32") / 255.0
    test_data = test_data.astype("float32") / 255.0
    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_labels = label_binarizer.fit_transform(train_labels)
    test_labels = label_binarizer.transform(test_labels)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train the network
    print("[INFO] training network...")
    model_fit = model.fit(
        train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128, epochs=20, verbose=1
    )

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_data, batch_size=128)
    print(
        classification_report(
            test_labels.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=[str(x) for x in label_binarizer.classes_],
        )
    )
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 20), model_fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 20), model_fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 20), model_fit.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 20), model_fit.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
