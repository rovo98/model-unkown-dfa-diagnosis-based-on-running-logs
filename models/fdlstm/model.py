# simple LSTM model.
# author : rovo98

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# noinspection DuplicatedCode
def build_fdlstm(input_shape,  # type: tuple
                 n_outputs,  # type: int
                 lstm_output_size=100,  # type: int
                 dropout_rate=0.5  # type: float
                 ):
    # type: (...) -> Sequential
    """Building and returning compiled fdlstmnet model.
    This model only uses LSTM layers.

    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param lstm_output_size: number of the outputs of the LSTM layer.
            default: 300.
    :param dropout_rate: 0-1, the fraction to drop out the units.
    :return: Compiled model (built with tf.keras APIs)
    """

    model = Sequential(name='fdlstmnet')
    model.add(Input(shape=input_shape))
    model.add(LSTM(lstm_output_size))
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model
