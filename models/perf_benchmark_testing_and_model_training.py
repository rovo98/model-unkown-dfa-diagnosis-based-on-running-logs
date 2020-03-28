# performance benchmarks testing for the models.
# Training & Evaluating the compiled models.
# author rovo98
import os

import numpy as np
import tensorflow as tf
from models.fdconv1d.model import build_2l_conv1d
from model_data_input import load_processed_dataset

# filter warning logs of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable memory growth for every GPU.
# Using GPU devices to train the models is recommended.
# uncomment the following several lines of code to disable forcing using GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware available'
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


def evaluate_model(
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
    verbose, epochs, batch_size = 1, 10, 32

    model = build_2l_conv1d((n_timesteps, n_features), n_outputs, kernel_size=7, dropout_rate=0.2)

    # plot the model
    # model_plot_name = './model_archs/fd1dconvnet_temporary.png'
    # plot_model(model, show_shapes=True, to_file=model_plot_name)
    # print('model plot saved: {}'.format(model_plot_name))

    # fit network
    model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(x=test_x, y=test_y, batch_size=batch_size, verbose=0)

    # save the model
    if save_model:
        model_name = './trained_saved/fd1dconvnet_czE4OmZzNDphczE2OmZlczI=.h5'
        model.save(model_name)
        print('> model {} saved.'.format(model_name))
    return accuracy


def summarize_results(scores):
    """summarize scores"""
    print('scores: ', scores)
    m, s = np.mean(scores), np.std(scores)
    print('Avg Test Accuracy: %.3f%%(+/-%.3f)' % (m, s))


# def summarize_results(scores, params):
#     """summarize scores"""
#     print('scores: ', scores, 'kernels: ', params)
#     for i in range(len(scores)):
#         m, s = np.mean(scores[i]), np.std(scores[i])
#         print('Params=%d, Accuracy: %.3f%%(+/-%.3f)' % (params[i], m, s))
#
#     # boxplot of scores
#     plt.boxplot(scores, labels=params)
#     plt.savefig('./exper_imgs/exp_cnn_filters__kernel_size=31_02.png')


def run_experiment(repeats=10):
    """run an experiment"""

    num_of_faulty_type = 5

    # print('>>> loading data from files...')
    #
    # train_x, train_y = load_dataset_group(
    #     '2019-12-28 00:46:37_czc1OmZzNzphczE1OmZlczQ=_processed_logs', num_of_faulty_type)
    #
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)
    #
    # print('size of training set: {}, size of testing set: {}'.format(len(train_x), len(test_x)))
    #
    # n_features = train_x.shape[1]
    # print('n_features: {}'.format(n_features))
    # # reshape the dataset.
    # train_x = np.expand_dims(train_x, axis=2)
    # test_x = np.expand_dims(test_x, axis=2)
    #
    # print('training dataset shape: {}'.format(train_x.shape))
    #
    # print('>>> training and testing dataset loaded successfully!')

    train_x, train_y, test_x, test_y = load_processed_dataset(
        '2019-12-28 00:46:37_czc1OmZzNzphczE1OmZlczQ=_processed_logs', num_of_faulty_type, test_size_ratio=0.1)

    # n_params = [96, 128, 192, 256]
    # all_scores = list()

    # for p in n_params:
    # repeat experiment
    scores = list()
    for r in range(repeats):
        # print('>>>\t current it: {}, p: {}'.format(r, p))
        score = evaluate_model(train_x, train_y, test_x, test_y, num_of_faulty_type)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
        # all_scores.append(scores)

    # score = evaluate_model(train_x, train_y, test_x, test_y, num_of_faulty_type, 64)
    # score = score * 100.0
    # print('>#%s: %.3f' % ('Test Accuracy', score))
    # summarize results.
    # summarize_results(all_scores, n_params)

    print('Average Accuracy: %.3f%%(+/-%.3f)' % (np.mean(scores), np.std(scores)))
