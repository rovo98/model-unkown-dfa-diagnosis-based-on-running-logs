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
        from utils.imbalanced_preprocessing import over_sampling
        result = over_sampling(test_list, 10_000)
        print(result)
        self.assertIsNotNone(result)
        self.assertTrue(len(result) == 10_000)

    def test_under_sampling(self):
        test_list = [1, 0, 1] * 10_000
        print(test_list)
        from utils.imbalanced_preprocessing import under_sampling
        result = under_sampling(test_list, 5_000)
        print(result)
        self.assertIsNotNone(result)
        self.assertTrue(len(result) == 5_000)


if __name__ == '__main__':
    unittest.main()
