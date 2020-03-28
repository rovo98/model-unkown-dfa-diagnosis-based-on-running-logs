# Loading dataset (training and testing) for models.
# rovo98

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_preprocessing.utils.load_and_save import load_sparse_csr

GENERATED_LOGS_LOC = '../dataset'


# FIXME: APIs design may needed to be refactored current design is less elegant and complex.

def load_processed_dataset(filename,
                           num_of_faulty_type,
                           num=1,
                           location=GENERATED_LOGS_LOC,
                           for_rnn=False,
                           test_size_ratio=0.2):
    """Loading the encoded running logs and do some operations before filling them to
    the specified model.
    1. training and testing dataset splitting.
    2. reshape the shape of the original dataset to meet model's requirements.

    :type filename: str
    :type num: int
    :type num_of_faulty_type: int
    :type location: str
    :type for_rnn: bool
    :type test_size_ratio: float
    :rtype numpy array, numpy array

    :param filename: name of the file that saves the data, or prefix of a bunch of files saving the data.
    :param num: number of files to be loaded. default 1
    :param location: location of the files to be loaded. A default value is set in the module
    :param for_rnn: which type of encoded running logs to be loaded.(False, cnn by default)
    :param test_size_ratio: the ratio of the testing dataset after splitting. default 0.2
    :return: dataset represented in np.array, features and labels
    """
    print('>>> loading data from files...')

    # using simple mapping encoding test.
    train_x, train_y = __load_dataset_group(filename, num_of_faulty_type, num, location, for_rnn)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=test_size_ratio)

    print('size of training set: {}, size of testing set: {}'.format(len(train_x), len(test_x)))
    n_features = train_x.shape[1]
    print('n_features: {}'.format(n_features))
    # reshape the dataset.
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)
    print('dataset shape: {}'.format(train_x.shape))
    print('>>> processed running logs loaded successfully!')

    return train_x, train_y, test_x, test_y


def __load_dataset_group(filename, num_of_classes, num=1,
                         location=GENERATED_LOGS_LOC,
                         for_rnn=False):
    """Loads np.arrays (data) from the specified location.

    :type filename: str
    :type num: int
    :type num_of_classes: int
    :type location: str
    :type for_rnn: bool
    :rtype numpy array, numpy array

    :param filename: name of the file that saves the data, or prefix of a bunch of files saving the data.
    :param num: number of files to be loaded. default 1
    :param num_of_classes: number of the classes of the labels.
    :param location: location of the files to be loaded.
    :param for_rnn: which type of encoded running logs to be loaded.(False, cnn by default)
    :return: dataset represented in np.array, features and labels
    """

    if num < 2:
        path = location + os.sep + filename + '.npz'
        print('File to be loading: {}'.format(path))
        features, labels = __load_and_split_feature_label(path, for_rnn)
        return features, to_categorical(labels, num_of_classes)
    else:
        result_features, result_labels = None, None
        for i in range(num):
            path = location + os.sep + filename + str(i) + '.npz'
            print('file to be loading: {}'.format(path))
            features, labels = __load_and_split_feature_label(path, for_rnn)

            if result_features is None:
                result_features = features
                result_labels = labels
            else:
                result_features = np.vstack((result_features, features))
                result_labels = np.vstack((result_labels, labels))
        return result_features, to_categorical(result_labels, num_of_classes)


def __load_and_split_feature_label(path, for_rnn):
    """helper function of load_dataset_group"""
    if for_rnn:
        data = np.load(path, allow_pickle=True)
        data = data['data']
    else:
        data = load_sparse_csr(path)
        data = data.todense()
    split = np.hsplit(data, (data.shape[1] - 1,))
    features, labels = split[0], split[1]
    return features, labels


def preprocess_dataset(
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_of_faulty_type,
        batch_size=32,
        shuffle_buffer_size=100):
    """using tf.data.Dataset.from_tensor_slices() method to consume numpy array data.
    NOTICE : Deprecated.

    :type train_features: numpy array
    :type train_labels: numpy array
    :type test_features: numpy array
    :type test_labels: numpy array
    :type num_of_faulty_type: int
    :type batch_size: int
    :type shuffle_buffer_size: int

    :param train_features: training features in ndarray representation.
    :param train_labels:    training labels in ndarray representation.
    :param test_features: testing features in ndarray representation.
    :param test_labels: testing labels in ndarray representation.
    :param num_of_faulty_type: the number of the faulty mode (depends on generated DFA)
    :param batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number
            of consecutive elements of this dataset to combine in a single batch.
    :param shuffle_buffer_size: A `tf.int64` scalar `tf.Tensor`, representing
            the number of elements from this dataset from which the new dataset will sample.
    :return: batched training and testing dataset (tf.data.Dataset)
    """

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, tf.one_hot(train_labels, depth=num_of_faulty_type)))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features, tf.one_hot(test_labels, depth=num_of_faulty_type)))

    train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    test_dataset.batch(batch_size)

    return train_dataset, test_dataset
