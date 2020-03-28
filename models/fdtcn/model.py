# Defining the TCN model.
# author rovo98

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tcn import TCN


def build_fdtcn(input_shape,  # type: tuple
                n_outputs,  # type: int
                nb_filter=64,  # type: int
                kernel_size=3,  # type: int
                lr=1e-3,  # type: float
                nb_stacks=1,  # type: int
                dilations=(1, 2, 4, 8, 16, 32),  # type: tuple
                ):
    # type: (...) -> Model
    """Build a TCN model, and returns the built compiled model.
    :param input_shape: 3D tuple (batch_size, timesteps, input_dim)
    :param n_outputs: the number of the outputs of this model.
    :param kernel_size: the kernel size of TCN layers.
    :param nb_filter: the number of the filters in TCN layers
    :param lr: learning rate of the Adam optimizer
    :param nb_stacks: num of the residual blocks.
    :param dilations: the list of dilations
    """

    inputs = Input(shape=input_shape)
    tcn = TCN(nb_filters=nb_filter, kernel_size=kernel_size,
              nb_stacks=nb_stacks,
              dilations=dilations)(inputs)
    # d = Dense(100, activation='relu')(tcn)
    outputs = Dense(n_outputs, activation='softmax')(tcn)
    adam = Adam(lr)
    model = Model(inputs, outputs, name='fdtcn')
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def build_fdtcn_2l(input_shape,  # type: tuple
                   n_outputs,  # type: int
                   kernel_size=3,  # type: int
                   ):
    # type: (...) -> Model

    inputs = Input(shape=input_shape)
    tcn1 = TCN(kernel_size=kernel_size, return_sequences=True)(inputs)
    tcn2 = TCN(kernel_size=kernel_size)(tcn1)
    # d = Dense(100, activation='relu')(tcn2)
    outputs = Dense(n_outputs, activation='softmax')(tcn2)
    model = Model(inputs, outputs, name='fdtcn2l')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
