# Defining the dfconv1d which stands for a model built using several 1 dimensional
# convolutional layer
# author rovo98

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_2l_conv1d(input_shape,  # type: tuple
                    n_outputs,  # type: int
                    filters=64,  # type: int
                    kernel_size=3,  # type: int
                    dropout_rate=0.5,  # type: float
                    max_pool_size=2  # type: int
                    ):
    # type: (...) -> Sequential
    """Build and returns the compiled model.
    This model using two convolution layers with same filters and kernel_size.

    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for Conv1D layer
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param filters: the number of the filters of the conv1d layer
    :param kernel_size: the kernel_size of the two conv1d layers.
    :param dropout_rate: dropout rate of the dropout layer
    :param max_pool_size: pool size of the MaxPool1D layer
    :return: Compiled model (built with tf.keras APIs)
    """
    model = Sequential(name='fd2lcov1dnet')
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))

    return __add_model_tail(model, n_outputs, dropout_rate, max_pool_size)


def build_3l_conv1d(input_shape,  # type: tuple
                    n_outputs,  # type: int
                    kernel_size=3,  # type: int
                    dropout_rate=0.5,  # type: float
                    max_pool_size=2,  # type: int
                    lr=0.001,  # type: float
                    ):
    # type: (...) -> Sequential
    """Building and compiling the fdconv1d model using three 1 dimensonal convolution layers.

    conv1d_1: 128 filters
    conv1d_2: 256 filters
    conv1d_3: 128 filters

    This model using three convolution layers with same filters and kernel_size.

    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for Conv1D layer
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param kernel_size: the kernel_size of the two conv1d layers.
    :param dropout_rate: dropout rate of the dropout layer
    :param max_pool_size: pool size of the MaxPool1D layer
    :param lr: learning rate of the Adam optimizer
    :return: Compiled model (built with tf.keras APIs)
    """
    model = Sequential(name='fd3lconv1dnet')
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))

    return __add_model_tail(model, n_outputs, dropout_rate, max_pool_size, lr)


def __add_model_tail(model,
                     n_outputs,
                     dropout_rate,
                     max_pool_size,
                     lr=0.001):
    """helper function of the methods above.
    Extracting the common codes here.
    """
    model.add(Dropout(dropout_rate))
    model.add(MaxPool1D(pool_size=max_pool_size))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    adam = Adam(lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    return model
