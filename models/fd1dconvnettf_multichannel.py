# Define 1 dimensional CNN with multi-head
# author rovo98

import time

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

from model_data_input import load_dataset_group

# filter warning logs of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable memory growth for every GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware available'
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


def evaluate_multi_head_model(
        train_x,
        train_y,
        test_x,
        test_y,
        n_outputs,
        save_model=False):
    """fit and evaluate a model.
        multi-head cnn model. In this model, we define three different kernel size headed cnn layers

        e.g. A three-headed model may have three different kernel size of 3, 5, 11, allowing the model to read and
        interpret sequence data at three different resolutions.

        The interpretations from all three heads are then concatenated within the model and interpreted by a
        fully-connected layer before a prediction is made.

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
    verbose, epochs, batch_size = 1, 30, 32

    # head 1
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv1 = Conv1D(filters=64, kernel_size=31, activation='relu')(inputs1)
    conv1_2 = Conv1D(filters=64, kernel_size=31, activation='relu')(conv1)
    drop1 = Dropout(0.5)(conv1_2)
    pool1 = MaxPool1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # head 2
    inputs2 = Input(shape=(n_timesteps, n_features))
    conv2 = Conv1D(filters=64, kernel_size=33, activation='relu')(inputs2)
    conv2_2 = Conv1D(filters=64, kernel_size=33, activation='relu')(conv2)
    drop2 = Dropout(0.5)(conv2_2)
    pool2 = MaxPool1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # head 3
    inputs3 = Input(shape=(n_timesteps, n_features))
    conv3 = Conv1D(filters=64, kernel_size=37, activation='relu')(inputs3)
    conv3_2 = Conv1D(filters=64, kernel_size=37, activation='relu')(conv3)
    drop3 = Dropout(0.5)(conv3_2)
    pool3 = MaxPool1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    # save a plot of the model
    model_plot_name = 'multi_channel_06.png'
    plot_model(model, show_shapes=True, to_file=model_plot_name)
    print('model plot saved: {}'.format(model_plot_name))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit([train_x, train_x, train_x], train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate([test_x, test_x, test_x], test_y, batch_size=batch_size, verbose=0)

    # save the model
    if save_model:
        model_name = 'fd1dcovnet_multichannel_06.h5'
        model.save(model_name)
        print('> model {} saved.'.format(model_name))
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

    # split dataset into training and testing sets.
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)

    print('size of training set: {}, size of testing set: {}'.format(len(train_x), len(test_x)))

    n_features = train_x.shape[1]
    print('n_features: {}'.format(n_features))
    # reshape the dataset.
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

    print('training dataset shape: {}'.format(train_x.shape))

    print('>>> training and testing dataset loaded successfully!')

    # repeat experiment
    # scores = list()
    # for r in range(repeats):
    #     score = evaluate_multi_head_model(train_x, train_y, test_x, test_y, num_of_faulty_type)
    #     score = score * 100.0
    #     print('>#%d: %.3f' % (r + 1, score))
    #     scores.append(score)

    score = evaluate_multi_head_model(train_x, train_y, test_x, test_y, num_of_faulty_type, True)
    score = score * 100.0
    print('>#%d test acc: %.3f' % (1, score))
    # summarize results.
    # summarize_results(scores)


# Driver the program to test the method above.
if __name__ == '__main__':
    start_time = time.perf_counter()
    run_experiment()
    end_time = time.perf_counter()
    print('Total cost time: {} seconds.'.format(end_time - start_time))
