# Build 1 dimensional CNNs using tf.keras libs.
# author rovo98

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from raw_data_processing import GENERATED_LOGS_LOC
from utils.load_and_save_sparse_matrix import load_sparse_csr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    """

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, tf.one_hot(train_labels, depth=num_of_faulty_type)))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features, tf.one_hot(test_labels, depth=num_of_faulty_type)))

    train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    test_dataset.batch(batch_size)

    return train_dataset, test_dataset


def evaluate_model(
        train_x,
        train_y,
        test_x,
        test_y,
        n_outputs):
    """fit and evaluate a model."""

    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
    epochs, batch_size = 10, 32
    model = Sequential(name='fd1dcovnet')
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # fit network
    model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, verbose=2)
    # evaluate model
    _, accuracy = model.evaluate(x=test_x, y=test_y, batch_size=batch_size, verbose=0)
    return accuracy


def summarize_results(scores):
    """summarize scores"""
    print('scores: ', scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%%(+/-%.3f)' % (m, s))


def run_experiment(repeats=10):
    """run an experiment"""

    num_of_faulty_type = 5

    print('>>> loading data from files...')

    train_x, train_y = load_dataset_group(
        '2019-12-28 00:46:37_czc1OmZzNzphczE1OmZlczQ=_processed_logs', num_of_faulty_type)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)
    # test_x, test_y = load_dataset_group(
    #     '2019-12-26 20:38:06_czEwMjpmczEwOmFzMTc6ZmVzNA==_processed_logs_ep8', num_of_faulty_type)

    print('size of training set: {}, size of testing set: {}'.format(len(train_x), len(test_x)))

    # train_dataset, test_dataset = preprocess_dataset(train_x, train_y, test_x, test_y, 5)

    n_features = train_x.shape[1]
    print('n_features: {}'.format(n_features))
    # reshape the dataset.
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

    print('training dateset shape: {}'.format(train_x.shape))

    print('>>> training and testing dataset loaded successfully!')

    scores = list()
    for r in range(repeats):
        score = evaluate_model(train_x, train_y, test_x, test_y, num_of_faulty_type)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)

    # summarize results.
    summarize_results(scores)


# Driver the program to test the method above.
if __name__ == '__main__':
    run_experiment()
