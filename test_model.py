# test_model.py

import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TestLinearRegressionPipeline(unittest.TestCase):

    def setUp(self):
        # Fake dataset with perfect linearity
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Day': np.arange(1, 101),
            'Price': np.arange(10, 110) + np.random.normal(0, 1, 100)
        })

    def test_model_training_and_prediction(self):
        X = self.df[['Day']]
        y = self.df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        self.assertLess(mse, 10.0, "MSE should be low for a near-linear dataset")

    def test_model_coefficients(self):
        model = LinearRegression()
        model.fit(self.df[['Day']], self.df['Price'])
        self.assertGreater(model.coef_[0], 0, "Coefficient should be positive")

if __name__ == '__main__':
    unittest.main()
