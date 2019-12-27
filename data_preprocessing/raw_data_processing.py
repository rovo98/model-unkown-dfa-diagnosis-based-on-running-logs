# loading raw data, processing them, and save to a csv file.
# author rovo98

import os
import re
import time
from operator import itemgetter
import random

import numpy as np

from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils.imbalanced_preprocessing import under_sampling, over_sampling
from utils.load_and_save_sparse_matrix import save_sparse_csr

DEFAULT_RAW_DATA_PATH = '../../generated-logs'

CHARACTER_FREQ_LIST = []  # stores the frequency of each character in the given overall running logs.
CHARACTER_ENCODING_MAPPINGS = {}  # stores the mappings between raw observation and encoded observation

RAW_RUNNING_LOGS = []  # list for storing un-processed running logs.

MAX_COLUMN_SIZE = 0  # maximum number of columns for processed observation of logs.
OBSERVATION_LENGTH = 50  # the maximum length of the observation of logs in training data.
SIZE_OF_EACH_LOG_TYPE = []

GENERATED_LOGS_LOC = '../dataset'  # location to save the processed running logs
DEFAULT_GENERATED_LOG_FILE_NAME = ''


def load_data(filename, raw_data_path=DEFAULT_RAW_DATA_PATH):
    """Loads raw running logs. Fill contents to the global parameters.

    :type filename: str
    :type raw_data_path: str

    Arguments
    ---
    filename: str
        filename of the file containing running logs.
    raw_data_path: str
        the path of the logs located.
    """

    global DEFAULT_GENERATED_LOG_FILE_NAME
    DEFAULT_GENERATED_LOG_FILE_NAME = filename.split('_')[1]

    print('>>> loading raw data...\n')

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
        threshold = int(sum(SIZE_OF_EACH_LOG_TYPE[1:]) / (len(SIZE_OF_EACH_LOG_TYPE) - 1))
        print('>>> Do over-sampling or under sampling..., threshold: {}'.format(threshold))
        its = len(RAW_RUNNING_LOGS)
        for i in tqdm(range(its), desc="over/under sampling"):
            length = len(RAW_RUNNING_LOGS[i])
            if length > threshold:
                RAW_RUNNING_LOGS[i] = under_sampling(RAW_RUNNING_LOGS[i], threshold)
            else:
                RAW_RUNNING_LOGS[i] = over_sampling(RAW_RUNNING_LOGS[i], threshold)

        # re-combining the RAW_RUNNING_LOGS into one list and shuffling it.
        temp = []
        for _ in range(its):
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
                              .get(CHARACTER_FREQ_LIST[-1][0])) * OBSERVATION_LENGTH

        print('>>> done!\n')


def encode_and_save_logs(filename='processed_logs'):
    """encoding running logs in global variable RAW_RUNNING_LOGS using method mentioned in README.md

    :type filename: str

    Arguments
        filename: name of the file to save logs.

    Encoding running logs, and then save them to the specified file.
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
               DEFAULT_GENERATED_LOG_FILE_NAME + '_' + filename
    # TODO: batch processing or more elegant approach, memory overflow issue
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
                batched_logs_list.append(transform_observation(RAW_RUNNING_LOGS[i]))
            start, end = end + 1, min(end + batch_size, log_size)

            # saving current batched running logs to local file is recommended.
            save_as_npz(batched_logs_list, filename + '_ep' + str(ep))
            del batched_logs_list
    else:
        for i in tqdm(range(log_size), desc="Encoding observation of logs"):
            encoded_logs_list.append(transform_observation(RAW_RUNNING_LOGS[i]))

        print('>>> done!\n')
        save_as_npz(encoded_logs_list, filename)


def transform_observation(log):
    """helper function for encode()

    :type log: list

    Arguments
        log: A running log <list> type contains observation, label

    Returns
        A encoded running log. <list> and the last element is the label of the log.
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
    padding_len = MAX_COLUMN_SIZE - len(encoded_observation)
    encoded_observation.extend([0] * padding_len)

    # convert str to int
    label = int(log[1])

    encoded_observation.append(label)
    return encoded_observation


def save_as_npz(logs, filename):
    """Helper func for encode() method.

    :type logs: list
    :type filename: str
    :rtype None

    Arguments
        logs: a list of encoded running logs.
        filename: the name of the file to save the logs.
    """

    # basic checking
    if filename is None or len(filename) == 0:
        raise Exception('Invalid filename is given!')
    if len(logs) == 0:
        return

    if not os.path.exists(GENERATED_LOGS_LOC):
        os.mkdir(GENERATED_LOGS_LOC)

    print('>>> saving logs to specified file...')
    # representing logs as np.array and compressing them.
    csr_logs = csr_matrix(np.array(logs))
    path = GENERATED_LOGS_LOC + os.sep + filename + '.npz'
    save_sparse_csr(path, csr_logs)
    print('>>> logs saved to {}'.format(path))


# Driver the program to test the method above.
if __name__ == '__main__':
    # NOTICE: Check OBSERVATION_LENGTH before launch the program.

    # load_data('2019-12-23 18:33:31_czgxOmZzODphczIwOmZlczQ=_running-logs.txt')
    # load_data('2019-12-24 21:33:29_czEwMjpmczEwOmFzMTc6ZmVzNA==_running-logs.txt')
    # load_data('2019-12-23 18:34:59_czY3OmZzNjphczE2OmZlczM=_running-logs.txt')
    load_data('2019-12-28 00:41:55_czc1OmZzNzphczE1OmZlczQ=_running-logs.txt')
    print('Filled configurations: ')
    print('max column size:', MAX_COLUMN_SIZE)
    print('size of each log type: ', SIZE_OF_EACH_LOG_TYPE)
    encode_and_save_logs()
    # print(RAW_RUNNING_LOGS)
    # print(len(RAW_RUNNING_LOGS))
