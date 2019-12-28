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

        model = load_model('../fd1dcovnet.h5')

        print('model loaded successfully.')
        self.assertIsNotNone(model)

        observations = np.expand_dims(observations, axis=2)
        print(observations)

        # do predictions on the given dataset.
        predictions = model.predict(tf.cast(observations, tf.float16))

        print('prediction: ', predictions)

        for p in predictions:
            p = list(p)
            print('predict label:', p, 'faulty type: ', p.index(max(p)))

        for el in encoded_logs:
            ell = list(el[1])
            print('label: {}, faulty type: {}'.format(ell, ell.index(max(ell))))

        # FIXME: compute the accuracy is much better than above.


if __name__ == '__main__':
    unittest.main()
