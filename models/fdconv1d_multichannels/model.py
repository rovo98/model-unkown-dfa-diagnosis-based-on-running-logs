# Defining the models using conv1d layers.
# These models is mainly used for those using compact huffman encoding approach.
# Since the used filters and kernel_size are well tested.
# author : rovo98

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


def build_fdconv1d_multichannel(input_shape, n_outputs, lr=1e-3):
    """Building and returning compiled fd1dconvent_multichannel model

    multi-channel cnn is cnn model using different convolution layers with respective kernel size.

    And concatenate their flatten results before the last layer.

    NOTICE: this model aims at those using compact huffman encoding approach.

    :type input_shape: tuple
    :type n_outputs: int
    :type lr: float
    :param input_shape: (timesteps, input_dim) of (samples, timesteps, input_dim) for input layers.
    :param n_outputs: number of the outputs of the this model. (depends on the task.)
    :param lr: learning rate of the optimizer
    :return: Compiled model (built with tf.keras APIs)
    """
    pass
    inputs1 = Input(shape=input_shape)
    conv1 = Conv1D(filters=64, kernel_size=31, activation='relu')(inputs1)
    conv1_2 = Conv1D(filters=128, kernel_size=31, activation='relu')(conv1)
    conv1_3 = Conv1D(filters=64, kernel_size=31, activation='relu')(conv1_2)
    drop1 = Dropout(0.5)(conv1_3)
    pool1 = MaxPool1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # head 2
    inputs2 = Input(shape=input_shape)
    conv2 = Conv1D(filters=64, kernel_size=39, activation='relu')(inputs2)
    conv2_2 = Conv1D(filters=128, kernel_size=39, activation='relu')(conv2)
    conv2_3 = Conv1D(filters=64, kernel_size=39, activation='relu')(conv2_2)
    drop2 = Dropout(0.5)(conv2_3)
    pool2 = MaxPool1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # head 3
    inputs3 = Input(shape=input_shape)
    conv3 = Conv1D(filters=64, kernel_size=43, activation='relu')(inputs3)
    conv3_2 = Conv1D(filters=128, kernel_size=43, activation='relu')(conv3)
    conv3_3 = Conv1D(filters=64, kernel_size=43, activation='relu')(conv3_2)
    drop3 = Dropout(0.5)(conv3_3)
    pool3 = MaxPool1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    adam = Adam(learning_rate=lr)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs, name='fdconv1d-multichannels')
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    return model
