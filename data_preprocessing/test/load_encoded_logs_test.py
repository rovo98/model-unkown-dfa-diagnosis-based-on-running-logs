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
        print(data.todense())


if __name__ == '__main__':
    unittest.main()
