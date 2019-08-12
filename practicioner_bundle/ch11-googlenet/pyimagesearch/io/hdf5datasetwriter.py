# -*- coding: utf-8 -*-
"""Writing Features to an HDF5 Dataset.

This class is responsible for taking an input set of NumPy arrays
(whether features, raw images, etc.) and writing them to HDF5 format.

Attributes:
    dims (tuple):
        dims parameter controls the dimension or shape of the data we will be storing
        in the dataset.
    output_path (str):
        path to where our output HDF5 file will be stored on disk.
    data_key (str, optional):
        name of the dataset (default = "images")
    buffer_size (int, optional):
        buffer_size controls the size of our in-memory buffer (default = 1000)
"""
import os
import h5py


class HDF5DatasetWriter:
    """Write data to to HDF5 format.
    """
    def __init__(self, dims, output_path, data_key="images", buffer_size=1000):
        """Initialize HDF5 dataset writer

        Arguments:
            dims {tuple} -- dimension or shape of the data that will be stored
            output_path {str} -- Path to where our HDF5 file will be stored on disk

        Keyword Arguments:
            data_key {str} -- name of the dataset (default: {"images"})
            buffer_size {int} -- size of the in-memory buffer (default: {1000})

        Raises:
            ValueError: supplied `output_path` already exists and cannot be overwritten
        """
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_path):
            raise ValueError("The supplied `output_path` already exists and cannot be overwritten."
                             "Manually delete the file before continuing.", output_path)
        # open the HDF5 database for writing and create two datasets: one to store the
        # images/features and another to store the class labels
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.buffer_size = buffer_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        """Add the rows and labels to the buffer

        Arguments:
            rows {array} -- data arrays / feature vectors
            labels {list} -- labels to the corresponding feature vectors
        """
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        """write the buffers to disk then reset the buffer
        """
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels):
        """Store the class labels to a separate file.

        Arguments:
            class_labels {list} -- list of class labels
        """
        # create a dataset to store the actual class label names, then store the class labels
        data_type = h5py.special_dtype(vlen='unicode')
        label_set = self.db.create_dataset("label_names", (len(class_labels),), dtype=data_type)
        label_set[:] = class_labels

    def close(self):
        """write any data left in the buffers to HDF5 and close the dataset
        """
        # check to see if there are any other entries in the buffer that need to be flushed to disk
        if self.buffer["data"]:
            self.flush()
        # close the dataset
        self.db.close()
