import unittest


class MyTestCase(unittest.TestCase):
    # noinspection DuplicatedCode
    def test_loading_model_and_predict(self):
        # prepare test data
        test_logs = ['vlwkkpqwdwlT4', 'ksvnbwxxwblbpnnklkbpwllxdwpkbddddlknkwpkbddddlT2',
                     'kdqsvblsxblsvnwkkT4', 'vlvnkswwslwwT1',
                     'vldsplwllxllvqnqkkkklkxT3', 'kslsxbkkwsvxsdssddkbspxkswpbwxxT0',
                     'vldsxvslddvslbbvwblT2']

        import utils.log_encoding as log_encoding
        import numpy as np

        # Encoding the given testing logs.
        log_encoding.load_config('2019-12-28 16:43:36_czc1OmZzNzphczE1OmZlczQ=_config')
        encoded_logs = []
        observations = None
        for rl in test_logs:
            features, label = log_encoding.encode_log(rl, num_of_faulty_type=5)
            encoded_logs.append([features, label])
            if observations is None:
                observations = features
            else:
                observations = np.vstack((observations, features))

        self.assertTrue(len(encoded_logs) == len(test_logs))

        from tensorflow.keras.models import load_model
        import os
        import tensorflow as tf

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Enable memory growth for every GPU physical device.
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, 'Not enough GPU hardware available'
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

        # model = load_model('../trained_saved/fd1dconvnet_czc1OmZzNzphczE1OmZlczQ=.h5')
        # model = load_model('../trained_saved/fd1dconvnet_multichannel_06_czc1OmZzNzphczE1OmZlczQ=.h5')
        model = load_model('../trained_saved/fdcnnlstmnet2_01_czc1OmZzNzphczE1OmZlczQ=.h5')

        print('model loaded successfully.')
        self.assertIsNotNone(model)

        observations = np.expand_dims(observations, axis=2)
        print(observations)

        # do predictions on the given dataset.
        input_x = tf.cast(observations, tf.float16)
        # predictions = model.predict(tf.cast(observations, tf.float16))
        predictions = model.predict([input_x, input_x, input_x])

        print('prediction: ', predictions)

        for p in predictions:
            p = list(p)
            print('predict label:', p, 'faulty type: ', p.index(max(p)))

        for el in encoded_logs:
            ell = list(el[1])
            print('label: {}, faulty type: {}'.format(ell, ell.index(max(ell))))

        # FIXME: compute the accuracy is much better than above.

    def test_trained_model_perf(self):
        from model_data_input import load_dataset_group
        # num_of_classes = 5
        num_of_classes = 3
        # x, y = load_dataset_group(
        #     '2019-12-31 23:47:12_czEwMjpmczEwOmFzMTc6ZmVzNA==_processed_logs_ep1',
        #     num_of_classes, 1, '../../dataset')

        x, y = load_dataset_group(
            '2020-01-10 21:20:48_czE4OmZzNDphczE2OmZlczI=_processed_logs',
            num_of_classes, 1, '../../dataset')

        self.assertIsNotNone(x)
        self.assertIsNotNone(y)

        # reshape the testing dataset
        import numpy as np

        x = np.expand_dims(x, axis=2)

        print('size of test dataset: {}'.format(len(x)))

        # do prediction using loaded model
        from tensorflow.keras.models import load_model
        import os
        import tensorflow as tf

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Enable memory growth for every GPU physical device.
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, 'Not enough GPU hardware available'
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

        # model = load_model('../trained_saved/fd1dconvnet_czEwMjpmczEwOmFzMTc6ZmVzNA==.h5')
        model = load_model('../trained_saved/fd1dconvnet_czE4OmZzNDphczE2OmZlczI=.h5')
        _, accuracy = model.evaluate(x=x, y=y, batch_size=32)

        print('\nTested acc: {}'.format(accuracy))


if __name__ == '__main__':
    unittest.main()
