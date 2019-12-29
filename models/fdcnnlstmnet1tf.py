# Define cnn model using one LSTM (long short term memory) layer
# author rovo98
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.utils import plot_model

from model_data_input import load_dataset_group

# filter warning logs of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable memory growth for every GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware available'
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


def evaluate_cnn_lstm_model(
        train_x,
        train_y,
        test_x,
        test_y,
        n_outputs,
        lstm_output_size,
        save_model=False):
    """fit and evaluate a model.

    modification of the model defined in fd1dconvnettf.py

    Args
        train_x: features of training dataset
        train_y: labels of training dataset
        test_x: features of testing dataset
        test_y: labels of testing dataset
        n_outputs: number of the outputs of the model
        save_model: whether to save model or not, default False.
    Returns
        None
    """

    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
    verbose, epochs, batch_size = 1, 10, 32

    model = Sequential(name='fd1dcovnet_cnn_lstm')
    model.add(Conv1D(filters=64, kernel_size=37, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=37, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # save a plot of the model
    model_plot_name = './model_archs/fdcnnlstmnet1.png'
    plot_model(model, show_shapes=True, to_file=model_plot_name)
    print('model plot saved: {}'.format(model_plot_name))

    # fit network
    model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(x=test_x, y=test_y, batch_size=batch_size, verbose=0)

    # save the model
    if save_model:
        model_name = 'fd1dconvnet_cnn_lstm.h5'
        model.save(model_name)
        print('> model {} saved.'.format(model_name))
    return accuracy


def summarize_results(scores):
    """summarize scores"""
    print('scores: ', scores)
    m, s = np.mean(scores), np.std(scores)
    print('Mean Accuracy: %.3f%%(+/-%.3f)' % (m, s))


def run_experiment(repeats=10):
    """run an experiment"""

    num_of_faulty_type = 5

    print('>>> loading data from files...')

    train_x, train_y = load_dataset_group(
        '2019-12-28 00:46:37_czc1OmZzNzphczE1OmZlczQ=_processed_logs', num_of_faulty_type)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)

    print('size of training set: {}, size of testing set: {}'.format(len(train_x), len(test_x)))

    n_features = train_x.shape[1]
    print('n_features: {}'.format(n_features))
    # reshape the dataset.
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

    print('training dataset shape: {}'.format(train_x.shape))

    print('>>> training and testing dataset loaded successfully!')

    lstm_output_size = 300

    # scores = list()
    # for r in range(repeats):
    #     score = evaluate_cnn_lstm_model(train_x, train_y, test_x, test_y, num_of_faulty_type, lstm_output_size)
    #     score = score * 100.0
    #     print('>#%d: %.3f' % (r + 1, score))
    #     scores.append(score)

    score = evaluate_cnn_lstm_model(train_x, train_y, test_x, test_y, num_of_faulty_type, lstm_output_size)
    score = score * 100.0
    print('>#%s: %.3f' % ('Test Accuracy', score))
    # summarize results.
    # summarize_results(scores)


# Driver the program to test the method above.
if __name__ == '__main__':
    start_time = time.perf_counter()
    run_experiment()
    end_time = time.perf_counter()
    print('Total cost time: {} seconds.'.format(end_time - start_time))
