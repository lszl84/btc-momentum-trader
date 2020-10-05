import unittest
import numpy as np
from train_model import prepare_single_data_window


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
