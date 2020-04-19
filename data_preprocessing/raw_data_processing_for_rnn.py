# raw running-logs pre-processing for CNNs.
# This is an encoding process.
# This approach is mainly used by RNNs defined in this project.
# Different from previous approach (compact character-level encoding), this is much simpler.
# And it may also be used by CNNs defined in this project (using smaller kernel size).
# author rovo98

# Illustration:
# using simple mapping like 'a' -> 1, 'b' -> 2, ... 'z' -> 26

import os
import re
import random
import time
import numpy as np

import config
from tqdm import tqdm
from data_preprocessing.utils.imbalanced_preprocessing import under_sampling
from data_preprocessing.utils.imbalanced_preprocessing import over_sampling

SIZE_OF_EACH_LOG_TYPE = []
RAW_RUNNING_LOGS = []
MAX_COLUMN_SIZE = 0


# noinspection DuplicatedCode
def load_data(filename, raw_data_path=config.DEFAULT_RAW_DATA_PATH):
    """Loading raw running logs from the specified location.

    NOTICE: this is simple modification of the function in
    raw_data_processing.py with the same name.

    :type filename: str
    :type raw_data_path: str

    :param filename: name of the file containing the raw running logs
    :param raw_data_path: the path of the logs located.
    """

    config.DEFAULT_GENERATED_LOG_FILE_NAME = filename.split('_')[1]

    print('>>> loading raw data, file to be loaded: {}...'.format(filename))

    path = raw_data_path + os.sep + filename
    with open(path, 'r') as f:
        # the first line of the file contains the basic infos.
        base_info = f.readline()
        global SIZE_OF_EACH_LOG_TYPE
        SIZE_OF_EACH_LOG_TYPE = re.findall('log.*?(\\d+)', base_info)
        SIZE_OF_EACH_LOG_TYPE = [int(x) for x in SIZE_OF_EACH_LOG_TYPE]

        # initialize the RAW_RUNNING_LOGS list
        global RAW_RUNNING_LOGS
        log_types = len(SIZE_OF_EACH_LOG_TYPE)
        for _ in range(log_types):
            RAW_RUNNING_LOGS.append([])

        # Reading raw logs from file.
        logs = f.readlines()
        for i in tqdm(range(len(logs)), desc="Loading logs from file"):
            log = logs[i].strip().split('T')
            observation, faulty_mode = log[0], log[1]
            # store logs read from file
            RAW_RUNNING_LOGS[int(faulty_mode)].append([observation, faulty_mode])

        # Do over sampling or under sampling if it is necessary.
        # also ignoring the 0 items
        temp = list(filter(lambda x: x != 0, SIZE_OF_EACH_LOG_TYPE))
        threshold = int(sum(temp[1:]) / (len(temp) - 1))
        print('>>> Do over-sampling or under-sampling..., threshold: {}'.format(threshold))
        its = len(RAW_RUNNING_LOGS)
        for i in tqdm(range(its), desc='over/under sampling'):
            length = len(RAW_RUNNING_LOGS[i])
            # ignoring empty list.
            if length == 0:
                continue
            if length > threshold:
                RAW_RUNNING_LOGS[i] = under_sampling(RAW_RUNNING_LOGS[i], threshold)
            else:
                RAW_RUNNING_LOGS[i] = over_sampling(RAW_RUNNING_LOGS[i], threshold)

        # re-combining the RAW_RUNNING_LOGS into one list and shuffling it.
        temp = []
        for _ in range(its):
            if len(RAW_RUNNING_LOGS[0]) > 0:
                temp.extend(RAW_RUNNING_LOGS[0])
            del RAW_RUNNING_LOGS[0]
        random.shuffle(temp)
        del RAW_RUNNING_LOGS
        RAW_RUNNING_LOGS = temp

        print('>>> After sampling, the number of the overall logs becomes: {}'.format(len(RAW_RUNNING_LOGS)))
        global MAX_COLUMN_SIZE
        for pl in RAW_RUNNING_LOGS:
            length = len(pl[0])
            if MAX_COLUMN_SIZE < length:
                MAX_COLUMN_SIZE = length

        # validation
        if MAX_COLUMN_SIZE != config.OBSERVATION_LENGTH:
            print('>>> given OBSERVATION_LENGTH is error, actual maximum length is used!')
        print('>>> Done.')


def encode_and_save_logs(filename='processed_logs'):
    """encoding running logs in global variable RAW_RUNNING_LOGS
    A modification version.
    Encoding running logs, and then save them to the specified file.

    :type filename: str

    :param filename: name of the file to save logs.
    """

    # basic validation
    if len(RAW_RUNNING_LOGS) == 0:
        raise Exception('RAW_RUNNING_LOGS is empty, load_data() should be called before this func.')

    encoded_logs_list = []

    filename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '_' + \
               config.DEFAULT_GENERATED_LOG_FILE_NAME + '_' + filename + \
               '_rnn'
    # TODO: batch processing or more elegant approach is needed, memory overflow issue
    # batch processing may needed when running logs data is too big to process in one time.
    log_size = len(RAW_RUNNING_LOGS)
    if log_size > 300_000:
        batch_size = 100_000
        epoch = log_size // batch_size
        print('>>> batch processing is needed, batch size: {}, epochs: {}\n'.format(batch_size, epoch))
        start, end = 0, batch_size

        for ep in range(epoch):
            print('>>> current epoch: {}\n'.format(ep + 1))
            batched_logs_list = []
            for i in tqdm(range(start, end), desc='Encoding observations of logs'):
                batched_logs_list.append(__transform_observation(RAW_RUNNING_LOGS[i]))
            start, end = end + 1, min(end + batch_size, log_size)

            # saving current batched running logs to local file is recommended.
            __save_to_file(batched_logs_list, filename + '_ep' + str(ep))
            del batched_logs_list
    else:
        for i in tqdm(range(log_size), desc="Encoding observation of logs"):
            encoded_logs_list.append(__transform_observation(RAW_RUNNING_LOGS[i]))

        print('>>> done!\n')
        __save_to_file(encoded_logs_list, filename)


def __transform_observation(log):
    """helper function for encode_and_save()
    NOTICE: this method is also a modification in raw_data_processing.py with
    the same name.
    :type log: list
    :param log: A running log <list> type contains observation and label [observation, label]
    :return: A encoded running log. <list> and the last element is the label of the given log.
    """

    # basic checking
    if len(log) == 0:
        raise Exception('Invalid log format!')
    if MAX_COLUMN_SIZE == 0:
        raise Exception('load_data() method should be run firstly!')

    # simply maps ascii characters specified number
    # e.g. 'a' -> 1
    offset = 96
    encoded_observation = [(ord(x) - offset) for x in log[0]]

    # 0 padding.
    padding_len = MAX_COLUMN_SIZE - len(encoded_observation)
    encoded_observation.extend([0] * padding_len)

    # convert str to int
    label = int(log[1])

    encoded_observation.append(label)
    return encoded_observation


def __save_to_file(logs, filename):
    """Helper function
    NOTICE: this is a modification version of the function
    in raw_data_processing.py with the same name.

    :param logs: a list of encoded running logs
    :param filename: the name of the file to save the logs.
    :return: None
    """
    # basic checking
    if filename is None or len(filename) == 0:
        raise Exception('Invalid filename is given')
    if len(logs) == 0:
        return
    print('for debug, print out one encoded log: {}'.format(logs[0]))
    path = config.GENERATED_LOGS_LOC + os.sep + filename + '.npz'
    print('>>> saving the logs to specified file...(for rnn)')
    # simply saved into npz file.
    np.savez_compressed(path, data=np.array(logs))
    print('>>> logs saved to {}'.format(path))


# Driver the program to test the methods above.
if __name__ == '__main__':
    # load_data('2020-01-10 21:16:34_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')

    config.OBSERVATION_LENGTH = 50

    # multi-faulty testing.
    # load_data('2020-03-14 15:37:12_czgwOmZzODphczIwOmZlczQ=_running-logs.txt')

    # single faulty mode small state set
    # load_data('2020-03-17 14:32:41_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')  # short logs (10 - 50)
    # load_data('2020-03-17 14:34:49_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')  # long logs (60 - 100)

    # single faulty mode big state set
    # load_data('2020-03-17 14:52:12_czgwOmZzODphczE4OmZlczQ=_running-logs.txt')  # short logs (10 - 50)
    # load_data('2020-03-17 14:52:30_czgwOmZzODphczE4OmZlczQ=_running-logs.txt')  # long logs (60 - 100)

    # multiply faulty mode small state set
    # load_data('2020-03-17 14:52:51_czE3OmZzNDphczE0OmZlczI=_running-logs.txt')  # short logs (10 - 50)
    # load_data('2020-03-17 14:53:13_czE3OmZzNDphczE0OmZlczI=_running-logs.txt')  # long logs (60 - 100)

    # multiply faulty mode big state set
    # load_data('2020-03-17 14:53:32_czgwOmZzODphczIwOmZlczQ=_running-logs.txt')  # short logs (10 - 50)
    # load_data('2020-03-17 14:53:42_czgwOmZzODphczIwOmZlczQ=_running-logs.txt')  # long logs (60 - 100)

    # increasing log set size experiment dataset
    # load_data('2020-03-21 00:38:12_czE3OmZzNDphczE0OmZlczI=_running-logs_10k.txt')
    # load_data('2020-03-21 00:38:56_czE3OmZzNDphczE0OmZlczI=_running-logs_20k.txt')
    # load_data('2020-03-21 00:39:05_czE3OmZzNDphczE0OmZlczI=_running-logs_30k.txt')
    # load_data('2020-03-21 00:39:18_czE3OmZzNDphczE0OmZlczI=_running-logs_40k.txt')
    # increasing log length experiment dataset
    # load_data('2020-03-21 00:41:44_czE3OmZzNDphczE0OmZlczI=_running-logs_50L.txt')
    # load_data('2020-03-21 00:41:54_czE3OmZzNDphczE0OmZlczI=_running-logs_100L.txt')
    # load_data('2020-03-21 00:42:05_czE3OmZzNDphczE0OmZlczI=_running-logs_150L.txt')
    # load_data('2020-03-21 00:42:15_czE3OmZzNDphczE0OmZlczI=_running-logs_200L.txt')

    # extra egr system dataset
    load_data('2020-04-17 23:27:36_egr-system-logs.txt')
    print('size of each log type: ', SIZE_OF_EACH_LOG_TYPE)
    print('Maximum column size: {}'.format(MAX_COLUMN_SIZE))
    encode_and_save_logs()
    # encode_and_save_logs("processed_logs_200L")
