import unittest
import numpy as np
from train_model import prepare_single_data_window, prepare_dataset


class TestTrainingPipeline(unittest.TestCase):

    def test_prepare_single_data_window(self):
        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10470, 0.12], [6, 10463, 0.33], [7, 10476, 0.6],
                           [7, 10430, 1.2]])

        x, y = prepare_single_data_window(
            trades, 2, 4, 5, req_target_factor=1.005, req_stop_factor=0.999)

        # no stop but no target hit (tgt_price = 10510.289, stop_price = 10447.542)
        self.assertTrue(np.allclose(x, [2, 10457.11, 1.1,
                                        3, 10458, 0.72]))
        self.assertEqual(y, False)

        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10470, 0.12], [6, 10512, 0.33], [7, 10476, 0.6],
                           [7, 10430, 1.2]])

        x, y = prepare_single_data_window(
            trades, 2, 4, 5, req_target_factor=1.005, req_stop_factor=0.999)

        # target hit (tgt_price = 10510.289, stop_price = 10447.542)
        self.assertTrue(np.allclose(x, [2, 10457.11, 1.1,
                                        3, 10458, 0.72]))
        self.assertEqual(y, True)

        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10442, 0.12], [6, 10512, 0.33], [7, 10476, 0.6],
                           [7, 10430, 1.2]])

        x, y = prepare_single_data_window(
            trades, 2, 4, 5, req_target_factor=1.005, req_stop_factor=0.999)

        # stop hit before target (tgt_price = 10510.289, stop_price = 10447.542)
        self.assertTrue(np.allclose(x, [2, 10457.11, 1.1,
                                        3, 10458, 0.72]))
        self.assertEqual(y, False)

        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10512, 0.12], [6, 10442, 0.33], [7, 10476, 0.6],
                           [7, 10430, 1.2]])

        x, y = prepare_single_data_window(
            trades, 2, 4, 5, req_target_factor=1.005, req_stop_factor=0.999)

        # stop hit after target, so the trade is valid (tgt_price = 10510.289, stop_price = 10447.542)
        self.assertTrue(np.allclose(x, [2, 10457.11, 1.1,
                                        3, 10458, 0.72]))
        self.assertEqual(y, True)

    def test_prepare_dataset(self):
        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10470, 0.12], [6, 10512, 0.33], [7, 10476, 0.6],
                           [7, 10430, 1.2], [7, 10560, 3.4]])

        X, y = prepare_dataset(
            trades, 2, 5, req_target_factor=1.005, req_stop_factor=0.999)

        #         >>> trades[:, 1]*1.005
        # array([10508.481  , 10508.481  , 10509.39555, 10510.29   , 10513.305  ,
        #        10516.32   , 10522.35   , 10564.56   , 10528.38   , 10482.15   ])
        # >>> trades[:, 1]*0.999
        # array([10445.7438 , 10445.7438 , 10446.65289, 10447.542  , 10450.539  ,
        #        10453.536  , 10459.53   , 10501.488  , 10465.524  , 10419.57   ])

        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)

        self.assertTrue(np.allclose(
            X[0], [1, 10456.20, 0.21, 1, 10456.20, 0.03]))
        self.assertTrue(np.allclose(
            X[1], [2, 10457.11, 1.1, 3, 10458, 0.72]))
        self.assertTrue(np.allclose(
            X[2], [4, 10461, 0.21, 5, 10464, 0.03]))

        self.assertEqual(y[0], False)
        self.assertEqual(y[1], True)
        # 10430 should stop us out, even though 10516.32 target is hit at 10560
        self.assertEqual(y[2], False)

        # let's just make sure we have the ranges right
        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10470, 0.12], [6, 10512, 0.33], [7, 10476, 0.6],
                           [7, 10430, 1.2]])

        X, y = prepare_dataset(
            trades, 2, 5, req_target_factor=1.005, req_stop_factor=0.999)

        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)

        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03],
                           [6, 10470, 0.12]])

        X, y = prepare_dataset(
            trades, 2, 5, req_target_factor=1.005, req_stop_factor=0.999)

        self.assertEqual(len(X), 1)
        self.assertEqual(len(y), 1)

        trades = np.array([[1, 10456.20, 0.21], [1, 10456.20, 0.03], [2, 10457.11, 1.1],
                           [3, 10458, 0.72], [4, 10461, 0.21], [5, 10464, 0.03]])

        X, y = prepare_dataset(
            trades, 2, 5, req_target_factor=1.005, req_stop_factor=0.999)

        # we need total 2+5 trades
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)


if __name__ == '__main__':
    unittest.main()
