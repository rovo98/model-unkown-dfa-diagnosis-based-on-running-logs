# Also a multi-headed model
# first head: three convolution layers with a globalAveragePooling layer
# second head: simple LSTM followed by a dropout layer.
# author rovo98

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense


# noinspection DuplicatedCode
def build_fdconv1d_lstm(input_shape, n_outputs, kernel_size=3, for_rnn=False):
    """Building and returning compiled fdcnnlstmnet2 model.

    This model is a multi-channel model using three convolution layers
    following by a global average pooling layer as head1 and one LSTM layer following
    by a dropout layer.

    NOTICE: except the kernel_size hyper-parameter of the model, others are fixed.

    :type input_shape: tuple
    :type n_outputs: int
    :type kernel_size: int
    :type for_rnn: bool
    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param kernel_size: the hyper-parameter `kernel_size` of every convolution layer, default 3.
    :param for_rnn: using different filters for different dataset encoding approach. (Avoid gpu out of memory exception)
                    default False.
    :return: Compiled model (built with tf.keras APIs)
    """
    # head 1
    inputs1 = Input(shape=input_shape)
    # head 2
    inputs2 = Input(shape=input_shape)
    lstm = LSTM(300)(inputs2)
    dropout = Dropout(0.2)(lstm)
    pool = __header(inputs1, kernel_size, for_rnn)
    merged = concatenate([pool, dropout])

    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='fdconv1d-lstm')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def __header(inputs, kernel_size, for_rnn=False):
    """Helper function of method above.
    Returns a convolution header with three convolution layers
    """
    if for_rnn:
        conv = Conv1D(filters=64, kernel_size=kernel_size, activation='relu')(inputs)
        conv_1 = Conv1D(filters=128, kernel_size=kernel_size, activation='relu')(conv)
        conv_2 = Conv1D(filters=64, kernel_size=kernel_size, activation='relu')(conv_1)
    else:
        conv = Conv1D(filters=128, kernel_size=kernel_size, activation='relu')(inputs)
        conv_1 = Conv1D(filters=256, kernel_size=kernel_size, activation='relu')(conv)
        conv_2 = Conv1D(filters=128, kernel_size=kernel_size, activation='relu')(conv_1)

    pool = GlobalAveragePooling1D()(conv_2)
    return pool
