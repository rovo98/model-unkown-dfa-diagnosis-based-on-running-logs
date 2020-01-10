# Define a model using CNNs layers and LSTM layers as two heads
# author rovo98

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

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
    verbose, epochs, batch_size = 1, 15, 32

    # FIXME: relu vanishing gradient problem may needed to be solved.
    # using leaky relu, prelu, or elu (exponential linear unit), or selu (scaled elu)

    # head 1
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv = Conv1D(filters=64, kernel_size=31, activation='selu')(inputs1)
    conv_1 = Conv1D(filters=128, kernel_size=31, activation='selu')(conv)
    conv_2 = Conv1D(filters=64, kernel_size=31, activation='selu')(conv_1)
    pool = GlobalAveragePooling1D()(conv_2)

    # head 2
    inputs2 = Input(shape=(n_timesteps, n_features))
    lstm = LSTM(128)(inputs2)
    dropout = Dropout(0.2)(lstm)

    merged = concatenate([pool, dropout])

    # interpretation
    dense1 = Dense(100, activation='selu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    # save a plot of the model
    model_plot_name = './model_archs/fdcnnlstmnet2_02.png'
    plot_model(model, show_shapes=True, to_file=model_plot_name)
    print('model plot saved: {}'.format(model_plot_name))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit([train_x, train_x], train_y, epochs=epochs, batch_size=batch_size,
              verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate([test_x, test_x], test_y, batch_size=batch_size, verbose=0)

    # plot history data
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    # ax1.set_ylabel('Accuracy')
    # ax1.set_xlabel('Epoch')
    # ax1.plot(history.history['accuracy'], label='train')
    # ax1.plot(history.history['val_accuracy'], label='val')
    # ax2.set_ylabel('Categorical crossentropy')
    # ax2.set_xlabel('Epoch')
    # ax2.plot(history.history['loss'], label='train')
    # ax2.plot(history.history['val_loss'], label='val')
    # ax2.yaxis.tick_right()
    # plt.legend()
    # plt.savefig('./exper_imgs/fdcnnlstmnet2_fig_01.png')

    # save the model
    if save_model:
        model_name = './trained_saved/fdcnnlstmnet2_czc1OmZzNzphczE1OmZlczQ=.h5'
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
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)

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

    score = evaluate_multi_head_model(train_x, train_y, test_x, test_y, num_of_faulty_type)
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
