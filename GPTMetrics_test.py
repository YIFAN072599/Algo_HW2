import unittest
import pandas as pd
import numpy as np
from GPTMetrics import TAQMetrics

class TestTAQMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a sample DataFrame with mock data
        data = {
            'Price': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
            'Volume': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            'midQuote': [100.5, 101.5, 102.5, 101.5, 100.5, 99.5, 100.5, 101.5, 102.5, 103.5]
        }
        cls.df = pd.DataFrame(data)
        cls.df.index = pd.date_range(start='2023-03-27 09:30:00', periods=10, freq='2T')

        cls.taq_metrics = TAQMetrics(cls.df)

    def test_calculate_vwap(self):
        expected_vwap = 100.66666666666667
        actual_vwap = self.taq_metrics.calculate_vwap()
        self.assertAlmostEqual(expected_vwap, actual_vwap, places=5)

    def test_calculate_vwap_sub(self):
        expected_vwap_sub = 101.11111111111111
        actual_vwap_sub = self.taq_metrics.calculate_vwap_sub()
        self.assertAlmostEqual(expected_vwap_sub, actual_vwap_sub, places=5)

    def test_calculate_market_impact(self):
        expected_g = 1.5
        expected_h = -0.8333333333333335
        actual_g, actual_h = self.taq_metrics.calculate_market_impact()
        self.assertAlmostEqual(expected_g, actual_g, places=5)
        self.assertAlmostEqual(expected_h, actual_h, places=5)

    def test_calculate_imbalance(self):
        expected_imbalance = 24000
        actual_imbalance = self.taq_metrics.calculate_imbalance()
        self.assertEqual(expected_imbalance, actual_imbalance)

    def test_calculate_mid_quote_returns_std(self):
        expected_std = 0.014719602977
        actual_std = self.taq_metrics.calculate_mid_quote_returns_std()
        self.assertAlmostEqual(expected_std, actual_std, places=5)

    def test_calculate_total_daily_volume(self):
        expected_volume = 55000
        actual_volume = self.taq_metrics.calculate_total_daily_volume()
        self.assertEqual(expected_volume, actual_volume)

    def test_get_terminal_price(self):
        expected_terminal_price = 102.0
        actual_terminal_price = self.taq_metrics.get_terminal_price()
        self.assertEqual(expected_terminal_price, actual_terminal_price)

    def test_get_arrival_price(self):
        expected_arrival_price = 101.0
        actual_arrival_price = self.taq_metrics.get_arrival_price()
        self.assertEqual(expected_arrival_price, actual_arrival_price)

if __name__ == '__main__':
    unittest.main()