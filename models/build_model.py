# Define and comiling models for the diagnosis problem (classification task)
# All models are built with tf.keras APIs (tensorflow official maintained)
# author rovo98

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM


# TODO: redesign the parameters of the following apis.


def build_fd1d2convnet(input_shape, n_outputs):
    """Building and compiling the fd1d2convnet model.

    This model using two convolution layers with same filters and kernel_size.

    :type input_shape: tuple
    :type n_outputs: int

    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for Conv1D layer
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :return: Compiled model (built with tf.keras APIs)
    """
    model = Sequential(name='fd1d2covnet')
    model.add(Conv1D(filters=64, kernel_size=31, activation='relu',input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=31, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model


def build_fd1d3convnet(input_shape, n_outputs):
    """Building and compiling the fd1d3convnet model.

    This model using three convolution layers with same filters and kernel_size.

    :type input_shape: tuple
    :type n_outputs: int

    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for Conv1D layer
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :return: Compiled model (built with tf.keras APIs)
    """
    model = Sequential(name='fd1d3covnet')
    model.add(Conv1D(filters=128, kernel_size=31, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=256, kernel_size=31, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=31, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_fd1dconvnet_multichannel(input_shape, n_outputs):
    """Building and returning compiled fd1dconvent_multichannel model

    multi-channel cnn is cnn model using different convolution layers with respective kernel size.

    And concatenate their flatten results before the last layer.

    :type input_shape: tuple
    :type n_outputs: int
    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :return: Compiled model (built with tf.keras APIs)
    """
    pass
    inputs1 = Input(shape=input_shape)
    conv1 = Conv1D(filters=64, kernel_size=31, activation='relu')(inputs1)
    conv1_2 = Conv1D(filters=64, kernel_size=31, activation='relu')(conv1)
    drop1 = Dropout(0.5)(conv1_2)
    pool1 = MaxPool1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # head 2
    inputs2 = Input(shape=input_shape)
    conv2 = Conv1D(filters=64, kernel_size=39, activation='relu')(inputs2)
    conv2_2 = Conv1D(filters=64, kernel_size=39, activation='relu')(conv2)
    drop2 = Dropout(0.5)(conv2_2)
    pool2 = MaxPool1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # head 3
    inputs3 = Input(shape=input_shape)
    conv3 = Conv1D(filters=64, kernel_size=43, activation='relu')(inputs3)
    conv3_2 = Conv1D(filters=64, kernel_size=43, activation='relu')(conv3)
    drop3 = Dropout(0.5)(conv3_2)
    pool3 = MaxPool1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_fdlstmnet(input_shape, n_outputs, lstm_output_size=300):
    """Building and returning compiled fdlstmnet model.
    This model only uses LSTM layers.


    :type input_shape: tuple
    :type n_outputs: int
    :type lstm_output_size: int
    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param lstm_output_size: number of the outputs of the LSTM layer.
    :return: Compiled model (built with tf.keras APIs)
    """

    model = Sequential(name='fdlstmnet')
    model.add(Input(shape=input_shape))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_fdcnnlstmnet1(input_shape, n_outputs, lstm_output_size=300):
    """Building and returning compiled fdcnnlstmnet1 model.
    This model is modification of fd1dconvnet integrating with LSTM layers

    :type input_shape: tuple
    :type n_outputs: int
    :type lstm_output_size: int
    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param lstm_output_size: number of the outputs of the LSTM layer.
    :return: Compiled model (built with tf.keras APIs)
    """

    model = Sequential(name='fd1dcovnet_cnn_lstm')
    model.add(Conv1D(filters=64, kernel_size=31, activation='selu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=31, activation='selu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(100, activation='selu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def build_fdcnnlstmnet2(input_shape, n_outputs):
    """Building and returning compiled fdcnnlstmnet2 model.

    This model is a multi-channel model using three convolution layers
    following by a global average pooling layer as head1 and one LSTM layer following
    by a dropout layer.

    :type input_shape: tuple
    :type n_outputs: int
    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :return: Compiled model (built with tf.keras APIs)
    """
    inputs1 = Input(shape=input_shape)
    conv = Conv1D(filters=128, kernel_size=31, activation='relu')(inputs1)
    conv_1 = Conv1D(filters=256, kernel_size=31, activation='relu')(conv)
    conv_2 = Conv1D(filters=128, kernel_size=31, activation='relu')(conv_1)
    pool = GlobalAveragePooling1D()(conv_2)

    # head 2
    inputs2 = Input(shape=input_shape)
    lstm = LSTM(300)(inputs2)
    dropout = Dropout(0.2)(lstm)

    merged = concatenate([pool, dropout])

    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    return model
