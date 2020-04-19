# Do feature engineering using convolution layers rather than directly using
# plain input sequences.
# author rovo98

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# noinspection DuplicatedCode
def build_fd2conv1dlstm(input_shape,  # type: tuple
                        n_outputs,  # type: int
                        kernel_size=3,  # type: int
                        max_pool_size=2,  # type: int
                        lstm_output_size=100,  # type: int
                        dropout_rate=0.5,  # type: float
                        lr=0.001    # type: float
                        ):
    # type: (...) -> Sequential
    """Building and returning compiled fdcnnlstmnet1 model.
    This model is modification of fd1dconvnet integrating with LSTM layers
    NOTICE: this model's filters are set to 64.

    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param kernel_size: the kernel size of the convolution layers in model.
    :param max_pool_size: the pool size of the MaxPooling layer in model.
    :param lstm_output_size: number of the outputs of the LSTM layer.
    :param dropout_rate: the fraction of the units to be dropout (dropout Layer).
    :param lr: learning rate of the optimizer
    :return: Compiled model (built with tf.keras APIs)
    """

    model = Sequential(name='fdconv1dlstm')
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPool1D(pool_size=max_pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    adam = Adam(lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    return model
