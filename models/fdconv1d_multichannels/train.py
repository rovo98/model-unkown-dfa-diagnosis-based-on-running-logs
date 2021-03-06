# training the defined model
# NOTICE: this model currently only tested on dataset processed using compact huffman encoding.
# author rovo98

import os

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.model_selection import train_test_split
from models.fdconv1d_multichannels.model import build_fdconv1d_multichannel
from model_data_input import load_processed_dataset
from models.utils.misc import running_timer
from models.utils.misc import plot_training_history
from models.utils.misc import multichannel_batch_generator

# filter warning logs of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable memory growth for every GPU.
# Using GPU devices to train the models is recommended.
# uncomment the following several lines of code to disable forcing using GPU.
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, 'Not enough GPU hardware available'
# for gpu in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu, True)

# another approach from 'https://github.com/tensorflow/tensorflow/issues/25138#issuecomment-559339162'
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# noinspection DuplicatedCode
@running_timer
def train_model(epochs=10,
                batch_size=32,
                training_verbose=1,
                print_model_summary=False,
                using_validation=False,
                validation_split=0.2,
                plot_history_data=False,
                history_fig_name='default',
                plot_model_arch=False,
                plot_model_name='default',
                save_model=False,
                save_model_name='default'):
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2019-12-28 00:46:37_czc1OmZzNzphczE1OmZlczQ=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')

    # 1. single faulty mode(small state size): short logs (10 - 50)
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 15:55:22_czE4OmZzNDphczE2OmZlczI=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')
    # 2. single faulty mode(small state size): long logs (60 - 100)
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:00:22_czE4OmZzNDphczE2OmZlczI=_processed_logs_b', num_of_faulty_type,
    #     location='../../dataset')

    # 3. single faulty mode(big state size): short logs (10 - 50)
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:16:04_czgwOmZzODphczE4OmZlczQ=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')
    # 4. single faulty mode(big state size): long logs (60 - 100)
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-19 17:09:05_czgwOmZzODphczE4OmZlczQ=_processed_logs_b_rg', num_of_faulty_type,
    #     location='../../dataset')

    # 5. multi faulty mode (small state size): short logs
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:34:50_czE3OmZzNDphczE0OmZlczI=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')

    # 6. multi faulty mode (small state size): long logs
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:36:40_czE3OmZzNDphczE0OmZlczI=_processed_logs_b', num_of_faulty_type,
    #     location='../../dataset')

    # 7. multi faulty mode (big state size): short logs
    # num_of_faulty_type = 16
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:40:03_czgwOmZzODphczIwOmZlczQ=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')

    # 8. multi faulty mode (big state size): long logs
    # num_of_faulty_type = 16
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:41:29_czgwOmZzODphczIwOmZlczQ=_processed_logs_b', num_of_faulty_type,
    #     location='../../dataset')

    # increasing log set size experiment dataset
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:16:49_czE3OmZzNDphczE0OmZlczI=_processed_logs_10k', num_of_faulty_type,
    #     location='../../dataset')
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:17:39_czE3OmZzNDphczE0OmZlczI=_processed_logs_20k', num_of_faulty_type,
    #     location='../../dataset')

    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:18:02_czE3OmZzNDphczE0OmZlczI=_processed_logs_30k', num_of_faulty_type,
    #     location='../../dataset')

    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:18:33_czE3OmZzNDphczE0OmZlczI=_processed_logs_40k', num_of_faulty_type,
    #     location='../../dataset')

    # increasing log length experiment dataset
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:20:12_czE3OmZzNDphczE0OmZlczI=_processed_logs_50L', num_of_faulty_type,
    #     location='../../dataset')
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:20:38_czE3OmZzNDphczE0OmZlczI=_processed_logs_100L', num_of_faulty_type,
    #     location='../../dataset')
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:21:04_czE3OmZzNDphczE0OmZlczI=_processed_logs_150L', num_of_faulty_type,
    #     location='../../dataset')
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-23 13:21:54_czE3OmZzNDphczE0OmZlczI=_processed_logs_200L', num_of_faulty_type,
    #     location='../../dataset')

    # egr system dataset
    num_of_faulty_type = 3
    train_x, train_y, test_x, test_y = load_processed_dataset(
        '2020-04-18 02:32:00_egr-system-logs.txt_processed_logs', num_of_faulty_type,
        location='../../dataset')
    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

    # building the model.
    model = build_fdconv1d_multichannel((n_timesteps, n_features), num_of_faulty_type, lr=5e-5)

    # print out the model summary
    if print_model_summary:
        model.summary()

    # plot and save the model architecture.
    if plot_model_arch:
        plot_model(model, to_file=plot_model_name, show_shapes=True)

    # fit network
    if plot_history_data:
        history = model.fit(x=[train_x, train_x, train_x], y=train_y, epochs=epochs, batch_size=batch_size,
                            verbose=training_verbose, validation_split=validation_split)
        plot_training_history(history, 'fdconv1d-multichannels', history_fig_name, '../exper_imgs')
    elif using_validation:
        es = EarlyStopping('val_loss', 1e-5, 5, 1, 'min')
        history = model.fit(x=[train_x, train_x, train_x], y=train_y, epochs=epochs, batch_size=batch_size,
                            verbose=training_verbose, validation_split=validation_split, callbacks=[es])
        # using generator for training big dataset or complex model
        # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=validation_split)
        # history = model.fit_generator(generator=multichannel_batch_generator(train_x, train_y, batch_size, 3),
        #                               steps_per_epoch=len(train_x) // batch_size,
        #                               epochs=epochs,
        #                               validation_data=(val_x, val_y),
        #                               verbose=training_verbose,
        #                               callbacks=[es],
        #                               use_multiprocessing=True)
        plot_training_history(history, 'fdconv1d-multichannels', history_fig_name, '../exper_imgs')
    else:
        model.fit(x=[train_x, train_x, train_x], y=train_y, epochs=epochs, batch_size=batch_size,
                  verbose=training_verbose)

    _, accuracy = model.evaluate(x=[test_x, test_x, test_x], y=test_y, batch_size=batch_size, verbose=0)

    # saving the model
    if save_model:
        model.save(save_model_name)
        print('>>> model saved: {}'.format(save_model_name))

    print('\n>>> Accuracy on testing given testing dataset: {}'.format(accuracy * 100))


# Driver the program to test the methods above.
if __name__ == '__main__':
    train_model(50,
                print_model_summary=True,
                using_validation=True,
                history_fig_name='fdconv1dmultichannel-egr',
                save_model=True,
                save_model_name='../trained_saved/fdconv1dmultichannel-egr.h5')
