# loading raw data, processing them, and save to a csv file.
# running-logs encoding for 1d-CNN input.
# author rovo98
# since 2019.12.24

import os
import re
import time
from operator import itemgetter
import random

import numpy as np

from scipy.sparse import csr_matrix
from tqdm import tqdm

import config
from data_preprocessing.utils.imbalanced_preprocessing import under_sampling, over_sampling
from data_preprocessing.utils.load_and_save import save_sparse_csr

CHARACTER_FREQ_LIST = []  # stores the frequency of each character in the given overall running logs.
CHARACTER_ENCODING_MAPPINGS = {}  # stores the mappings between raw observation and encoded observation
RAW_RUNNING_LOGS = []  # list for storing un-processed running logs.
MAX_COLUMN_SIZE = 0  # maximum number of columns for processed observation of logs.
SIZE_OF_EACH_LOG_TYPE = []


# noinspection DuplicatedCode
def load_data(filename, raw_data_path=config.DEFAULT_RAW_DATA_PATH):
    """Loads raw running logs. Fill contents to the global parameters.

    :type filename: str
    :type raw_data_path: str

    :param filename: name of the file containing running logs.
    :param raw_data_path: the path of the logs located.
    """

    config.DEFAULT_GENERATED_LOG_FILE_NAME = filename.split('_')[1]

    print('>>> loading raw data, file to be loaded: {}...\n'.format(filename))

    path = raw_data_path + os.sep + filename
    with open(path, 'r') as f:
        # first line containing the basic info about the dfa.
        base_info = f.readline()
        global SIZE_OF_EACH_LOG_TYPE
        SIZE_OF_EACH_LOG_TYPE = re.findall('logs.*?(\\d+)', base_info)
        SIZE_OF_EACH_LOG_TYPE = [int(x) for x in SIZE_OF_EACH_LOG_TYPE]
        observable_events = (base_info.split(':')[-1]
                             .replace('[', '').replace(']', '').strip().split(','))
        # initialize character freq dict
        character_freq_dict = {}
        for e in observable_events:
            character_freq_dict[e] = 0

        global RAW_RUNNING_LOGS
        # initialize the RAW_RUNNING_LOGS list
        log_types = len(SIZE_OF_EACH_LOG_TYPE)
        for _ in range(log_types):
            RAW_RUNNING_LOGS.append([])
        # Reading raw logs from file.
        logs = f.readlines()
        for i in tqdm(range(len(logs)), desc="Loading logs from file"):
            log = logs[i].strip().split('T')
            observation, faulty_mode = log[0], log[1]
            # count frequency of each character
            for c in observation:
                character_freq_dict[c] = character_freq_dict[c] + 1
            # stores logs read from file.
            # RAW_RUNNING_LOGS.append([observation, faulty_mode])
            RAW_RUNNING_LOGS[int(faulty_mode)].append([observation, faulty_mode])

        # Do over sampling & under sampling if it is necessary
        # ignoring the 0 items
        temp = list(filter(lambda x: x != 0, SIZE_OF_EACH_LOG_TYPE))
        print(temp)
        threshold = int(sum(temp[1:]) / (len(temp) - 1))
        print('>>> Do over-sampling or under sampling..., threshold: {}'.format(threshold))
        its = len(RAW_RUNNING_LOGS)
        for i in tqdm(range(its), desc="over/under sampling"):
            length = len(RAW_RUNNING_LOGS[i])
            # ignoring the empty list
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

        print('>>> After sampling, the number of the overall logs becomes {}'.format(len(RAW_RUNNING_LOGS)))

        # sorting the observable events by frequency.
        global CHARACTER_FREQ_LIST
        for k, v in sorted(character_freq_dict.items(), key=itemgetter(1), reverse=True):
            CHARACTER_FREQ_LIST.append((k, v))

        global CHARACTER_ENCODING_MAPPINGS
        n = 0
        for c in CHARACTER_FREQ_LIST:
            numerical_list = [1]
            numerical_list.extend([0] * n)
            numerical_list.append(1)
            CHARACTER_ENCODING_MAPPINGS[c[0]] = numerical_list
            n = n + 1

        # count the maximum size of the features representing vector
        global MAX_COLUMN_SIZE
        MAX_COLUMN_SIZE = len(CHARACTER_ENCODING_MAPPINGS
                              .get(CHARACTER_FREQ_LIST[-1][0])) * config.OBSERVATION_LENGTH
        print('>>> done!\n')


def encode_and_save_logs(filename='processed_logs'):
    """encoding running logs in global variable RAW_RUNNING_LOGS using method mentioned in README.md
    Encoding running logs, and then save them to the specified file.

    :type filename: str

    :param filename: name of the file to save logs.
    """

    # basic validation
    if len(RAW_RUNNING_LOGS) == 0:
        raise Exception('RAW_RUNNING_LOGS is empty, load_data() should be called before this func.')
    if len(CHARACTER_ENCODING_MAPPINGS) == 0:
        raise Exception('CHARACTER_ENCODING_MAPPING is empty! check load_data() func')
    if MAX_COLUMN_SIZE == 0:
        raise Exception('Some initialization needed to be done in load_data() func')

    encoded_logs_list = []

    filename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '_' + \
               config.DEFAULT_GENERATED_LOG_FILE_NAME + '_' + filename
    # TODO: batch processing or more elegant approach is needed, memory overflow issue
    # batch processing may needed when running logs data is too big to process in one time.
    log_size = len(RAW_RUNNING_LOGS)
    if log_size > 300_000:
        batch_size = 100_000
        epoch = log_size // batch_size
        print('>>> batch processing is needed, batch size: {}, epochs: {}\n'.format(batch_size, epoch))
        start, end = 0, batch_size

        for ep in range(epoch):
            print('>>> epoch: {}\n'.format(ep + 1))
            batched_logs_list = []
            for i in tqdm(range(start, end), desc='Encoding observations of logs'):
                batched_logs_list.append(__transform_observation(RAW_RUNNING_LOGS[i]))
            start, end = end + 1, min(end + batch_size, log_size)

            # saving current batched running logs to local file is recommended.
            __save_as_npz(batched_logs_list, filename + '_ep' + str(ep))
            del batched_logs_list
    else:
        for i in tqdm(range(log_size), desc="Encoding observation of logs"):
            encoded_logs_list.append(__transform_observation(RAW_RUNNING_LOGS[i]))

        print('>>> done!\n')
        __save_as_npz(encoded_logs_list, filename)


def __transform_observation(log):
    """helper function for encode_and_save_logs()

    :type log: list

    :param log: A running log <list> type contains observation, label, structure like
            [observation, label]
    :return: A encoded running log. <list> and the last element is the label of the log.
    """

    # basic check
    if len(CHARACTER_ENCODING_MAPPINGS) == 0:
        raise Exception('CHARACTER_ENCODING_MAPPINGS is empty!')
    if len(log) == 0:
        raise Exception('Invalid log format')

    encoded_observation = []
    for c in log[0]:
        encoded_observation.extend(CHARACTER_ENCODING_MAPPINGS.get(c))
    # zero padding.
    # additional checking
    if MAX_COLUMN_SIZE < len(encoded_observation):
        raise Exception('Encoding error, please checking the OBSERVATION_LENGTH.')
    padding_len = MAX_COLUMN_SIZE - len(encoded_observation)
    encoded_observation.extend([0] * padding_len)

    # convert str to int
    label = int(log[1])

    encoded_observation.append(label)
    return encoded_observation


def __save_as_npz(logs, filename):
    """Helper func for encode_and_save_logs() method.

    :type logs: list
    :type filename: str
    :rtype None

    :param: logs: a list of encoded running logs.
    :param filename: the name of the file to save the logs.
    """

    # basic checking
    if filename is None or len(filename) == 0:
        raise Exception('Invalid filename is given!')
    if len(logs) == 0:
        return

    if not os.path.exists(config.GENERATED_LOGS_LOC):
        os.mkdir(config.GENERATED_LOGS_LOC)

    print('>>> saving logs to specified file...')
    # representing logs as np.array and compressing them.
    csr_logs = csr_matrix(np.array(logs))
    path = config.GENERATED_LOGS_LOC + os.sep + filename + '.npz'
    save_sparse_csr(path, csr_logs)
    print('>>> logs saved to {}'.format(path))


def save_encoding_config(filename='config'):
    """Saving configurations for encoding raw logs.
    NOTICE: This method must run after load_data().

    :type filename: str

    :param filename: name of the file to save configuration of current dataset.
    """

    # basic checking.
    if len(CHARACTER_ENCODING_MAPPINGS) == 0 or MAX_COLUMN_SIZE == 0:
        raise Exception('This method must run after load_data()!')

    from utils.load_and_save import save_object

    filename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '_' + \
               config.DEFAULT_GENERATED_LOG_FILE_NAME + '_' + filename
    path = config.GENERATED_ENCODING_CONFIG_LOC + os.sep + filename

    # creates target folder if it not existed.
    if not os.path.exists(config.GENERATED_ENCODING_CONFIG_LOC):
        os.mkdir(config.GENERATED_ENCODING_CONFIG_LOC)

    save_object(path, [CHARACTER_ENCODING_MAPPINGS, MAX_COLUMN_SIZE])
    print('> Encoding configuration saved to {} successfully.'.format(path))


# Driver the program to test the method above.
if __name__ == '__main__':
    # FIXME: Refactoring maybe needed, current configuring method is less elegant.
    # REMARKS: checking all configuration before launch the program.
    # NOTICE: Check OBSERVATION_LENGTH before launch the program.
    config.OBSERVATION_LENGTH = 200
    # load_data('2019-12-28 00:41:55_czc1OmZzNzphczE1OmZlczQ=_running-logs.txt')
    # load_data('2020-01-09 22:53:29_czE4OmZzNDphczE0OmZlczI=_running-logs.txt')
    # load_data('2020-01-09 22:56:13_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')
    # load_data('2020-01-10 21:16:34_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')

    # load_data('2020-03-14 15:37:12_czgwOmZzODphczIwOmZlczQ=_running-logs.txt')

    # single faulty mode small state set
    # load_data('2020-03-17 14:32:41_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')  # short logs (10 - 50)
    # load_data('2020-03-17 14:34:49_czE4OmZzNDphczE2OmZlczI=_running-logs.txt')  # long logs (60 - 100)

    # single faulty mode big state set
    # load_data('2020-03-17 14:52:12_czgwOmZzODphczE4OmZlczQ=_running-logs.txt')  # short logs (10 - 50)
    # load_data('2020-03-17 14:52:30_czgwOmZzODphczE4OmZlczQ=_running-logs.txt')  # long logs (60 - 100)
    # load_data('2020-03-19 17:06:46_czgwOmZzODphczE4OmZlczQ=_running-logs_re-generated.txt')

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
    load_data('2020-03-21 00:42:15_czE3OmZzNDphczE0OmZlczI=_running-logs_200L.txt')
    print('Filled configurations: ')
    print('max column size:', MAX_COLUMN_SIZE)
    print('size of each log type: ', SIZE_OF_EACH_LOG_TYPE)
    # encode_and_save_logs()
    encode_and_save_logs('processed_logs_200L')
    # save configuration of current processed running logs.
    # save_encoding_config()
