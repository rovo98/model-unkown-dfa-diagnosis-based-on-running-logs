# Loading dataset (training and testing) for models.
# rovo98

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

from raw_data_processing import GENERATED_LOGS_LOC
from utils.load_and_save import load_sparse_csr


def load_dataset_group(filename, num_of_classes, num=1):
    """Loads np.arrays (data) from the specified location.

    :type filename: str
    :type num: int
    :type num_of_classes: int
    :rtype numpy array, numpy array

    Args
        filename: name of the file that saves the data, or prefix of a bunch of files saving the data.
        num: number of files to be loaded. default 1

    Returns
        dataset represented in np.array, features and labels
    """

    if num < 2:
        path = GENERATED_LOGS_LOC + os.sep + filename + '.npz'
        print('File to be loading: {}'.format(path))
        data = load_sparse_csr(path)
        data = data.todense()
        split = np.hsplit(data, (data.shape[1] - 1,))
        features, labels = split[0], split[1]
        return features, to_categorical(labels, num_of_classes)
    else:
        result_features, result_labels = None, None
        for i in range(num):
            path = GENERATED_LOGS_LOC + os.sep + filename + str(i) + '.npz'
            print('file to be loading: {}'.format(path))
            data = load_sparse_csr(path)
            data = data.todense()
            split = np.hsplit(data, (data.shape[1] - 1,))
            features, labels = split[0], split[1]

            if result_features is None:
                result_features = features
                result_labels = labels
            else:
                result_features = np.vstack((result_features, features))
                result_labels = np.vstack((result_labels, labels))
        return result_features, to_categorical(result_labels, num_of_classes)


def preprocess_dataset(
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_of_faulty_type,
        batch_size=32,
        shuffle_buffer_size=100):
    """using tf.data.Dataset.from_tensor_slices() method to consume numpy array data.

    :type train_features: numpy array
    :type train_labels: numpy array
    :type test_features: numpy array
    :type test_labels: numpy array
    :type num_of_faulty_type: int
    :type batch_size: int
    :type shuffle_buffer_size: int

    Args
        train_features:
        train_labels:
        test_features:
        test_labels:
        num_of_fault: the number of the faulty mode (depends on generated DFA)
        batch_size:
        shuffle_buffer_size:
    Returns
        batched training and testing dataset (tf.data.Dataset)
    """

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, tf.one_hot(train_labels, depth=num_of_faulty_type)))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features, tf.one_hot(test_labels, depth=num_of_faulty_type)))

    train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    test_dataset.batch(batch_size)

    return train_dataset, test_dataset
