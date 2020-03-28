import unittest

RAW_DATA_LOC = '../../../generated-logs'


class MyTestCase(unittest.TestCase):
    def test_loading_basic_info(self):
        import os
        import re
        # filename = '2019-12-23 18:33:31_czgxOmZzODphczIwOmZlczQ=_running-logs.txt'
        filename = '2019-12-24 21:33:29_czEwMjpmczEwOmFzMTc6ZmVzNA==_running-logs.txt'
        path = RAW_DATA_LOC + os.sep + filename

        with open(path, 'r') as f:
            first_line = f.readline()
            num_of_each_log_type = re.findall('logs:.*?(\\d+)', first_line)
            self.assertIsNotNone(num_of_each_log_type)
            # convert to int list
            num_of_each_log_type = [int(x) for x in num_of_each_log_type]
            average_num = sum(num_of_each_log_type[1:]) / (len(num_of_each_log_type) - 1)
            print(num_of_each_log_type)
            average_num = int(average_num)
            print('average logs number: {}'.format(average_num))
            new_size = average_num * len(num_of_each_log_type)
            print('after sampling, size of overall logs becomes {}'.format(new_size))

    def test_over_sampling(self):
        test_list = [1, 0, 1, 1] * 1_000
        print(test_list)
        from data_preprocessing.utils.imbalanced_preprocessing import over_sampling
        result = over_sampling(test_list, 10_000)
        print(result)
        self.assertIsNotNone(result)
        self.assertTrue(len(result) == 10_000)

    def test_under_sampling(self):
        from data_preprocessing.utils.imbalanced_preprocessing import under_sampling

        test_list = [1, 0, 1] * 10_000
        print(test_list)
        result = under_sampling(test_list, 5_000)
        print(result)
        self.assertIsNotNone(result)
        self.assertTrue(len(result) == 5_000)

    def test_save_and_load_using_pickle(self):
        from data_preprocessing.utils.load_and_save import save_object
        from data_preprocessing.utils.load_and_save import load_object

        test_dict = {'a': [1, 0, 0, 1], 'b': [1, 1]}
        test_int = 100
        save_object('./test_config', [test_dict, test_int])

        loaded_list = load_object('./test_config')
        self.assertIsNotNone(loaded_list)
        self.assertEquals(loaded_list[1], test_int)
        print(loaded_list[0], loaded_list[1])
        print(type(loaded_list[0]), type(loaded_list[1]))

    def test_encoding_new_log(self):
        import data_preprocessing.utils.log_encoding as log_encoding
       
        test_logs = ['vlwkkpqwdwlT4', 'ksvnbwxxwblbpnnklkbpwllxdwpkbddddlknkwpkbddddlT2',
                     'kdqsvblsxblsvnwkkT4', 'vlvnkswwslwwT1',
                     'vldsplwllxllvqnqkkkklkxT3', 'kslsxbkkwsvxsdssddkbspxkswpbwxxT0',
                     'vldsxvslddvslbbvwblT2']

        # loading encoding configuration
        log_encoding.load_config('2019-12-28 16:43:36_czc1OmZzNzphczE1OmZlczQ=_config')

        print('character frequency mappings:')
        print(log_encoding.CHARACTER_ENCODING_MAPPINGS)
        print('max column size: ')
        print(log_encoding.MAX_COLUMN_SIZE)

        encoded_logs = []
        for rl in test_logs:
            features, label = log_encoding.encode_log(rl, num_of_faulty_type=5)
            encoded_logs.append([features, label])

        self.assertTrue(len(encoded_logs) == len(test_logs))
        # print(encoded_logs)
        for el in encoded_logs:
            print(el[0], len(el[0]), el[1])


if __name__ == '__main__':
    unittest.main()
