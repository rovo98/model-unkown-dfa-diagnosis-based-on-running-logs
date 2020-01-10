# utils func for encoding running log using compressed numerical vector in ndarray.
# rovo98
# since 2019.12.24


import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.load_and_save import load_object

CHARACTER_ENCODING_MAPPINGS = {}  # events to compressed code mappings of DES
MAX_COLUMN_SIZE = 0  # maximum column size of encoded log representation.
ENCODING_CONFIG_LOC = '../../encoding-configs'  # location to load configurations.


# NOTICE: this func is mainly used for encoding new coming running logs
# which is not in the training dataset.
def encode_log(log, num_of_faulty_type):
    """encoding the given log using compressed vector representation mentioned in README.md
    This func is a modified version of transform_observation in raw_data_processing module.

    :type log: str
    :type num_of_faulty_type: int

    :param log: a raw running log <str>
    :param num_of_faulty_type: the number of the faults of DES.
    :return: encoded log represented using ndarray, features, labels
    """
    # basic check
    if len(CHARACTER_ENCODING_MAPPINGS) == 0:
        raise Exception('CHARACTER_ENCODING_MAPPINGS is empty!')
    if len(log) == 0:
        raise Exception('Invalid log format')

    log_list = log.split('T')
    observation, label = log_list[0], log_list[1]

    encoded_observation = []
    for c in observation:
        encoded_observation.extend(CHARACTER_ENCODING_MAPPINGS.get(c))
    # zero padding.
    padding_len = MAX_COLUMN_SIZE - len(encoded_observation)
    encoded_observation.extend([0] * padding_len)

    # convert str to int
    label = int(label)

    encoded_observation = np.array(encoded_observation)
    label = np.array(label)

    label = to_categorical(label, num_of_faulty_type)

    return encoded_observation, label


def load_config(filename):
    """Loading necessary configuration for encoding running logs.
    HELPER function of above.

    :type filename: str
    :param filename: name of the configuration file to be loaded.(in preset location).
    """

    global CHARACTER_ENCODING_MAPPINGS, MAX_COLUMN_SIZE
    path = ENCODING_CONFIG_LOC + os.sep + filename
    load_list = load_object(path)
    CHARACTER_ENCODING_MAPPINGS, MAX_COLUMN_SIZE = load_list[0], load_list[1]
