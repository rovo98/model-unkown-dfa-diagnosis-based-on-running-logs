import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_plotting_accuracies(self):
        import matplotlib.pyplot as plt

        dataset = ['sfm-s-10_50', 'sfm-s-60_100',
                   'sfm-b-10_50', 'sfm-b-60_100',
                   'mfm-s-10_50', 'mfm-s-60_100',
                   'mfm-b-10_50', 'mfm-b-60_100']
        fdconv1d = [96.46, 99.65,
                    96.87, 98.46,
                    96.06, 99.05,
                    99.10, 96.70]
        fdconv1d_m = [97.73, 99.71,
                      95.42, 98.40,
                      95.92, 99.32,
                      90.02, 94.35]
        fdlstm = [97.27, 99.42,
                  93.43, 97.64,
                  95.72, 98.83,
                  91.99, 95.49]
        fdconv1dlstm = [96.56, 99.27,
                        83.71, 95.36,
                        94.80, 98.41,
                        75.86, 87.05]
        fdtcn = [97.85, 99.16,
                 94.36, 97.23,
                 95.88, 99.10,
                 52.32, 42.68]
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.scatter(dataset, fdconv1d, label='fdConv1d', marker='o')
        ax.scatter(dataset, fdconv1d_m, label='fdConv1dMultiChannel', marker='v')
        ax.scatter(dataset, fdlstm, label='fdLSTM', marker='^')
        ax.scatter(dataset, fdconv1dlstm, label='fdConv1dLSTM', marker='d')
        ax.scatter(dataset, fdtcn, label='fdTCN', marker='s')

        ax.plot(dataset, fdconv1d, linestyle='-.')
        ax.plot(dataset, fdconv1d_m, linestyle='-.')
        ax.plot(dataset, fdlstm, linestyle='-.')
        ax.plot(dataset, fdconv1dlstm, linestyle='-.')
        ax.plot(dataset, fdtcn, linestyle='-.')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_yticks(range(40, 100), 5)

        ax.legend()
        fig.suptitle('Testing Accuracy on Each Dataset')
        # fig.set_xlabel('dataset')
        # fig.set_ylabel('accuracy %')
        plt.savefig('testing-accuarcy-on-each-dataset.png')

    def test_plotting_exper_data(self):
        import matplotlib.pyplot as plt

        log_size_range = ['10k', '20k', '30k', '40k']
        log_length_range = ['50', '100', '150', '200']

        ls_fctree = [49.75, 68.00, 66.98, 62.17]
        ll_fctree = [62.99, 85.58, 91.95, 93.03]
        ls_fdconv1d = [93.64, 94.14, 95.48, 96.06]
        ll_fdconv1d = [96.13, 97.54, 98.45, 98.41]
        ls_fdconv1dmc = [91.33, 93.83, 95.08, 94.42]
        ll_fdconv1dmc = [95.59, 97.43, 98.13, 98.79]
        ls_fdconv1dlstm = [77.19, 87.47, 89.98, 89.62]
        ll_fdconv1dlstm = [91.69, 94.48, 95.31, 96.08]
        ls_fdtcn = [50.34, 58.57, 65.92, 69.32]
        ll_fdtcn = [76.65, 89.99, 97.21, 97.35]

        fig, ax = plt.subplots(1)

        # ax.scatter(log_size_range, ls_fctree, label='FCCtT', marker='p')
        # ax.plot(log_size_range, ls_fctree, linestyle='-.')
        # ax.scatter(log_size_range, ls_fdconv1d, label='fdConv1d', marker='o')
        # ax.plot(log_size_range, ls_fdconv1d, linestyle='-.')
        # ax.scatter(log_size_range, ls_fdconv1dmc, label='fdConv1dMultiChannel', marker='v')
        # ax.plot(log_size_range, ls_fdconv1dmc, linestyle='-.')
        # ax.scatter(log_size_range, ls_fdconv1dlstm, label='fdConv1dLSTM', marker='d')
        # ax.plot(log_size_range, ls_fdconv1dlstm, linestyle='-.')
        # ax.scatter(log_size_range, ls_fdtcn, label='fdTCN', marker='s')
        # ax.plot(log_size_range, ls_fdtcn, linestyle='-.')

        ax.scatter(log_length_range, ll_fctree, label='FCCtT', marker='p')
        ax.plot(log_length_range, ll_fctree, linestyle='-.')
        ax.scatter(log_length_range, ll_fdconv1d, label='fdConv1d', marker='o')
        ax.plot(log_length_range, ll_fdconv1d, linestyle='-.')
        ax.scatter(log_length_range, ll_fdconv1dmc, label='fdConv1dMultiChannel', marker='v')
        ax.plot(log_length_range, ll_fdconv1dmc, linestyle='-.')
        ax.scatter(log_length_range, ll_fdconv1dlstm, label='fdConv1dLSTM', marker='d')
        ax.plot(log_length_range, ll_fdconv1dlstm, linestyle='-.')
        ax.scatter(log_length_range, ll_fdtcn, label='fdTCN', marker='s')
        ax.plot(log_length_range, ll_fdtcn, linestyle='-.')

        ax.set_yticks(range(40, 101), 10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel('Log length')
        ax.set_ylabel('Accuracy %')
        ax.legend()

        # plt.show()
        plt.savefig('exper_inc_log_length.png')

    # noinspection PyMethodMayBeStatic
    def test_plotting_egr_results(self):
        import matplotlib.pyplot as plt
        models = ['fdConv1d', 'fdConv1dMultiChannel', 'fdConv1dLSTM', 'fdTCN']
        acc = [69.26, 70.13, 66.99, 69.47]

        fig, ax = plt.subplots(1)

        ax.scatter(models, acc)
        ax.plot(models, acc, linestyle='-.')

        ax.set_yticks(range(65, 72), 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel('Model Name')
        ax.set_ylabel('Accuracy %')
        fig.suptitle('Testing Accuracy on EGR System Dataset')
        # ax.legend()
        plt.savefig('testing-accuracy-on-egr-dataset.png')


if __name__ == '__main__':
    unittest.main()
