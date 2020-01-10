import unittest

import os
from utils.load_and_save import load_sparse_csr

GENERATED_LOGS_LOC = '../../dataset'


class MyTestCase(unittest.TestCase):
    def test_loading_raw_logs(self):
        import raw_data_processing as rdp

        rdp.load_data('2019-12-23 18:33:31_czgxOmZzODphczIwOmZlczQ=_running-logs.txt',
                      '../../../generated-logs')

        print('Filled configurations: ')
        print('max column size:', rdp.MAX_COLUMN_SIZE)
        print('size of each log type: ', rdp.SIZE_OF_EACH_LOG_TYPE)

        self.assertTrue(len(rdp.CHARACTER_FREQ_LIST) > 0)
        print('character frequency list:')
        print(rdp.CHARACTER_FREQ_LIST)

        self.assertTrue(len(rdp.CHARACTER_ENCODING_MAPPINGS) > 0)
        print('character encoding mapping:')
        print(rdp.CHARACTER_ENCODING_MAPPINGS)

    def test_loading_encoded_logs(self):
        path = GENERATED_LOGS_LOC + os.sep
        path = path + '2019-12-31 23:47:12_czEwMjpmczEwOmFzMTc6ZmVzNA==_processed_logs_ep0'
        data = load_sparse_csr(path)
        self.assertIsNotNone(data)
        print(data)
        # print(data.todense())
        densed_data = data.todense()
        print(densed_data.shape)
        # print(densed_data)

        import numpy as np
        split = np.hsplit(densed_data, (densed_data.shape[1] - 1,))
        train_data = split[0]
        labels = split[1]
        print(train_data, labels)
        print(train_data.shape)

        size = len(train_data)
        print('size: {}'.format(size))

        print(train_data.reshape(1, size * 1400).reshape(1, size, 1400)[0])
        print(train_data.reshape(size, 1, 1400).shape)
        print(train_data.reshape(size, 1400, 1).shape)
        print(train_data.shape)
        print(np.expand_dims(train_data, axis=2).shape)

        print(train_data[0].shape)
        print(train_data[0].reshape((1, 1400)))

        from sklearn.model_selection import train_test_split
        from tensorflow.keras.utils import to_categorical
        x_train, x_test, y_train, y_test = train_test_split(train_data, to_categorical(labels, 5), test_size=0.2)
        print(x_train, y_train)
        print(len(x_train), len(y_train))
        print(len(x_test), len(y_test))

        np.random.seed(200)
        np.random.shuffle(x_train)
        np.random.seed(200)
        np.random.shuffle(y_train)

        print(x_train, y_train)
        print(y_train.shape)

        # import tensorflow as tf
        # dataset = tf.data.Dataset.from_tensor_slices((train_data, tf.one_hot(labels, 5)))
        # dataset = dataset.shuffle(100).batch(32)
        # for batches, labels in dataset.take(1):
        #     print(batches.shape)


if __name__ == '__main__':
    unittest.main()
