# test_inflation.py

import unittest
import pandas as pd
import numpy as np
import logging
from inflation import adjust_for_inflation

# Suppress yfinance download progress bar
logging.getLogger('yfinance').setLevel(logging.ERROR)

class TestInflationAdjustment(unittest.TestCase):

    def setUp(self):
        print("\nSetting up test data...")
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=5),
            'Close': [100, 102, 101, 103, 104],
            'Open': [99, 101, 100, 102, 103]
        })
        print("Test data created successfully")

    def test_basic_inflation_adjustment(self):
        """Test basic inflation adjustment functionality"""
        print("\nRunning basic inflation adjustment test...")
        result = adjust_for_inflation(self.sample_data)
        self.assertIn('Close_Adj', result.columns)
        self.assertIn('Inflation Factor', result.columns)
        self.assertTrue(result['Close_Adj'].notna().all())
        print("Basic inflation adjustment test passed")

    def test_missing_date_column_raises(self):
        """Test that missing date column raises ValueError"""
        print("\nTesting missing date column...")
        df_no_date = self.sample_data.drop('Date', axis=1)
        with self.assertRaises(ValueError):
            adjust_for_inflation(df_no_date)
        print("Missing date column test passed")

    def test_missing_price_column_raises(self):
        """Test that missing price column raises ValueError"""
        print("\nTesting missing price column...")
        df_no_price = self.sample_data.drop('Close', axis=1)
        with self.assertRaises(ValueError):
            adjust_for_inflation(df_no_price)
        print("Missing price column test passed")

    def test_custom_price_column(self):
        """Test inflation adjustment with custom price column"""
        print("\nTesting custom price column...")
        result = adjust_for_inflation(self.sample_data, price_col='Open')
        self.assertIn('Open_Adj', result.columns)
        self.assertTrue(result['Open_Adj'].notna().all())
        print("Custom price column test passed")

    def test_inflation_factors(self):
        """Test that inflation factors are correctly applied"""
        print("\nTesting inflation factors...")
        result = adjust_for_inflation(self.sample_data)
        # Check that inflation factors are within expected range
        self.assertTrue((result['Inflation Factor'] >= 0.92).all())
        self.assertTrue((result['Inflation Factor'] <= 1.05).all())
        print("Inflation factors test passed")

if __name__ == '__main__':
    print("Starting inflation adjustment tests...")
    unittest.main(verbosity=2)
