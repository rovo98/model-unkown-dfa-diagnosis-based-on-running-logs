import unittest

import os
from utils.load_and_save_sparse_matrix import load_sparse_csr

GENERATED_LOGS_LOC = '../../dataset'


class MyTestCase(unittest.TestCase):
    def test_loading_encoded_logs(self):
        path = GENERATED_LOGS_LOC + os.sep
        # path = path + '2019-12-26 20:31:36_czgxOmZzODphczIwOmZlczQ=_processed_logs.npz'
        path = path + '2019-12-26 20:38:06_czEwMjpmczEwOmFzMTc6ZmVzNA==_processed_logs_ep0.npz'
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

        print(train_data.reshape(1, size*1400).reshape(1, size, 1400)[0])
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

        # import tensorflow as tf
        # dataset = tf.data.Dataset.from_tensor_slices((train_data, tf.one_hot(labels, 5)))
        # dataset = dataset.shuffle(100).batch(32)
        # for batches, labels in dataset.take(1):
        #     print(batches.shape)


if __name__ == '__main__':
    unittest.main()
