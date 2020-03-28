# misc tool functions using when building model or training model, etc.
# author rovo98
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps


def running_timer(task):
    """A decorator function as a wrapper of the given experiment to train the specified
    model.

    This will counting the time for the execution of the given experiment running.
    And print out the cost time in human readable format.

    :param task: a task to be running.
    """

    @wraps(task)
    def decorated(*args, **kwargs):
        start_time = time.perf_counter()
        task(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        if duration > 60 * 60:  # in hour format
            duration = duration / (60 * 60)
            print('Total cost time: {} hours.'.format(duration))
        elif duration > 60:
            duration = duration / 60
            print('Total cost time: {} minutes.'.format(duration))
        else:
            print('Total cost time: {} seconds.'.format(duration))

    return decorated


def nn_batch_generator(x_data, y_data, batch_size):
    """Generating a batch of training dataset for fitting the model.
    :param x_data : overall training dataset features.
    :param y_data : overall training dataset labels.
    :param batch_size : hyper-parameter `batch_size`
    """
    samples_per_epoch = x_data.shape[0]
    number_of_batches = samples_per_epoch / batch_size
    counter = 0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size * counter:batch_size * (counter + 1)]
        x_batch = x_data[index_batch]
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(x_batch), y_batch
        if counter > number_of_batches:
            counter = 0


def multichannel_batch_generator(x_data, y_data, batch_size, channel=2):
    """Generating a batch of training dataset for fitting the model.
    :param x_data : overall training dataset features.
    :param y_data : overall training dataset labels.
    :param batch_size : hyper-parameter `batch_size`
    :param channel: num of the channel to generate.
    """
    samples_per_epoch = x_data.shape[0]
    number_of_batches = samples_per_epoch / batch_size
    counter = 0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size * counter:batch_size * (counter + 1)]
        x_batch = x_data[index_batch]
        y_batch = y_data[index_batch]
        counter += 1

        x = []
        for c in range(channel):
            x.append(np.array(x_batch))
        yield x, y_batch
        if counter > number_of_batches:
            counter = 0


def plot_training_history(history, title, filename, location='./exper_imgs'):
    """Plot and save the acc & loss fig to the specified location with the given name.
    By default the location is under the '<curr_project>/models/exper_imgs'.

    :type history: any
    :type title: str
    :type filename: str
    :type location: str

    :param history: historical data after training the model.
    :param filename: the name of the fig to be saved.
    :param title: the suptitle of the figure
    :param location: the folder to save the plotted figs
    """
    # simple checking
    if location.endswith('/'):
        location.rstrip('/')
    print('>>> plotting the given history data...')
    # plot history data
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.plot(history.history['categorical_accuracy'], label='train')
    ax1.plot(history.history['val_categorical_accuracy'], label='val')
    ax2.set_ylabel('Categorical crossentropy')
    ax2.set_xlabel('Epoch')
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.yaxis.tick_right()
    plt.legend()

    plt.savefig('{}/{}.png'.format(location, filename))
    print(">>> plotted fig saved: {}/{}".format(location, filename))
