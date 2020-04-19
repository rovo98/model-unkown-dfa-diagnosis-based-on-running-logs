# Training the model, saving model.
# benchmarking the mark, running experiments on the given models.
# author rovo98

import os

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from model_data_input import load_processed_dataset
from models.fdlstm.model import build_fdlstm
from models.utils.misc import running_timer
from models.utils.misc import plot_training_history

# filter warning logs of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable memory growth for every GPU.
# Using GPU devices to train the models is recommended.
# uncomment the following several lines of code to disable forcing using GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware available'
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# another approach from 'https://github.com/tensorflow/tensorflow/issues/25138#issuecomment-559339162'
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


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
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-02-22 20:34:10_czE4OmZzNDphczE2OmZlczI=_processed_logs_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)

    # 1. single faulty mode(small state size): short logs (10 - 50)
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:03:28_czE4OmZzNDphczE2OmZlczI=_processed_logs_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)
    # 2. single faulty mode(small state size): long logs (60 - 100)
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:07:13_czE4OmZzNDphczE2OmZlczI=_processed_logs_b_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)

    # 3. single faulty mode(big state size): short logs (10 - 50)
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:32:12_czgwOmZzODphczE4OmZlczQ=_processed_logs_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)
    # 4. single faulty mode(big state size): long logs (60 - 100)
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:32:44_czgwOmZzODphczE4OmZlczQ=_processed_logs_b_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)

    # 5. multi faulty mode (small state size): short logs
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:37:52_czE3OmZzNDphczE0OmZlczI=_processed_logs_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)

    # 6. multi faulty mode (small state size): long logs
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:38:15_czE3OmZzNDphczE0OmZlczI=_processed_logs_b_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)

    # 7. multi faulty mode (big state size): short logs
    # num_of_faulty_type = 16
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:42:20_czgwOmZzODphczIwOmZlczQ=_processed_logs_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)
    # 8. multi faulty mode (big state size): long logs
    # num_of_faulty_type = 16
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:42:40_czgwOmZzODphczIwOmZlczQ=_processed_logs_b_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)

    num_of_faulty_type = 3
    train_x, train_y, test_x, test_y = load_processed_dataset(
        '2020-04-17 23:29:19_egr-system-logs.txt_processed_logs_rnn', num_of_faulty_type,
        location='../../dataset', for_rnn=True)
    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
    model = build_fdlstm((n_timesteps, n_features), num_of_faulty_type, 300)

    # print out the model summary
    if print_model_summary:
        model.summary()

    # plot and save the model architecture.
    if plot_model_arch:
        plot_model(model, to_file=plot_model_name, show_shapes=True)

    # fit network
    if plot_history_data:
        history = model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, verbose=training_verbose,
                            validation_split=validation_split)
        plot_training_history(history, 'fdlstm', history_fig_name, '../exper_imgs')
    elif using_validation:
        es = EarlyStopping('val_categorical_accuracy', 1e-4, 3, 1, 'max')
        history = model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, verbose=training_verbose,
                            validation_split=validation_split, callbacks=[es])
        plot_training_history(history, 'fdlstm', history_fig_name, '../exper_imgs')
    else:
        model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, verbose=training_verbose)

    _, accuracy = model.evaluate(x=test_x, y=test_y, batch_size=batch_size, verbose=0)

    # saving the model
    if save_model:
        model.save(save_model_name)
        print('>>> model saved: {}'.format(save_model_name))

    print('\n>>> Accuracy on testing given testing dataset: {}'.format(accuracy * 100))


def evaluate_model():
    pass


# running the program to test the methods above.
if __name__ == '__main__':
    # Training the model.
    train_model(100,
                print_model_summary=True,
                using_validation=True,
                history_fig_name='fdlstm-egr',
                save_model=True,
                save_model_name='../trained_saved/fdlstm-egr.h5')
