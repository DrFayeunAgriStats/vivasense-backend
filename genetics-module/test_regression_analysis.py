"""
Unit tests for analysis_regression_routes.py
"""

import base64
import unittest
from unittest.mock import patch
import pandas as pd
from fastapi import HTTPException

from analysis_regression_routes import analysis_regression, RegressionRequest


class TestRegressionEndpoint(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Standard valid dataset
        self.df_valid = pd.DataFrame({
            'X_Var': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            'Y_Var': [2.1, 3.9, 6.2, 8.1, 10.0, 12.1, 14.0, 15.9, 18.2, 20.0, 21.9, 24.1],
            'NonNumeric': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
            'Constant': [5.0] * 12
        })

        # High fit, small N dataset (n=5)
        self.df_small_high_fit = pd.DataFrame({
            'X_Var': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Y_Var': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        
        # Non-significant slope dataset
        self.df_no_relation = pd.DataFrame({
            'X_Var': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Y_Var': [10, 12, 9, 11, 10, 13, 8, 10, 11, 9, 12, 10]
        })

        self.mock_ctx = {
            'base64_content': base64.b64encode(self.df_valid.to_csv(index=False).encode()).decode(),
            'file_type': 'csv'
        }

    def _build_request(self, x, y, model="linear"):
        return RegressionRequest(
            dataset_token='test_token',
            x_variable=x,
            y_variable=y,
            model_type=model
        )

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_successful_linear_regression(self, mock_cache):
        """Test successful regression with valid numeric X/Y"""
        mock_cache.return_value = self.mock_ctx
        request = self._build_request('X_Var', 'Y_Var')

        res = await analysis_regression(request)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.n, 12)
        self.assertTrue(res.slope > 0)
        self.assertGreater(res.r_squared, 0.9)
        self.assertLess(res.p_value_slope, 0.05)
        self.assertEqual(res.direction, "positive")
        self.assertEqual(res.strength_class, "strong")
        self.assertEqual(res.significance_class, "significant")
        self.assertIn("increases by", res.plain_language_effect)
        self.assertIn("captures a meaningful proportion", res.summary_interpretation)
        
        # Plot data present
        self.assertEqual(len(res.plot_data.x), 12)
        self.assertEqual(len(res.plot_data.fitted_y), 12)
        
        # Reliable (sample_size_ok flag)
        self.assertIn("sample_size_ok", res.reliability_flags)

        # Equation format: "Y = a + b × X" — spaces around ×, sign explicit
        self.assertIn(" \u00d7 ", res.equation)          # multiplication sign with spaces
        self.assertIn("Y_Var", res.equation)              # predictor name present
        self.assertTrue(
            res.equation.startswith("Y_Var ="),
            f"Equation should start with 'Y_Var =', got: {res.equation!r}",
        )

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_rejection_x_equals_y(self, mock_cache):
        """Reject when X == Y"""
        mock_cache.return_value = self.mock_ctx
        request = self._build_request('X_Var', 'X_Var')

        with self.assertRaises(HTTPException) as cm:
            await analysis_regression(request)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("must be different", cm.exception.detail)

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_rejection_non_numeric(self, mock_cache):
        """Reject when a variable is non-numeric"""
        mock_cache.return_value = self.mock_ctx
        request = self._build_request('X_Var', 'NonNumeric')

        with self.assertRaises(HTTPException) as cm:
            await analysis_regression(request)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("must be numeric", cm.exception.detail)

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_rejection_zero_variance(self, mock_cache):
        """Reject when X or Y has zero variance"""
        mock_cache.return_value = self.mock_ctx
        request = self._build_request('Constant', 'Y_Var')

        with self.assertRaises(HTTPException) as cm:
            await analysis_regression(request)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("zero variance", cm.exception.detail)

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_rejection_insufficient_n(self, mock_cache):
        """Reject when n < 3"""
        # Dataset with only 2 rows
        df_tiny = pd.DataFrame({'X_Var': [1, 2], 'Y_Var': [1, 2]})
        mock_cache.return_value = {
            'base64_content': base64.b64encode(df_tiny.to_csv(index=False).encode()).decode(),
            'file_type': 'csv'
        }
        request = self._build_request('X_Var', 'Y_Var')

        with self.assertRaises(HTTPException) as cm:
            await analysis_regression(request)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("Minimum 3 required", cm.exception.detail)

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_warnings_small_sample(self, mock_cache):
        """High R² + Small N warning + Small sample warning"""
        mock_cache.return_value = {
            'base64_content': base64.b64encode(self.df_small_high_fit.to_csv(index=False).encode()).decode(),
            'file_type': 'csv'
        }
        request = self._build_request('X_Var', 'Y_Var')
        res = await analysis_regression(request)

        self.assertEqual(res.n, 5)
        self.assertIn("small_sample", res.reliability_flags)
        self.assertIn("high_fit_small_n", res.reliability_flags)
        
        warning_texts = " ".join(res.warnings)
        self.assertIn("may be unstable", warning_texts)
        self.assertIn("may be misleading", warning_texts)

    @patch('analysis_regression_routes.dataset_cache.get_dataset')
    async def test_warnings_non_significant(self, mock_cache):
        """Non-significant slope warning"""
        mock_cache.return_value = {
            'base64_content': base64.b64encode(self.df_no_relation.to_csv(index=False).encode()).decode(),
            'file_type': 'csv'
        }
        request = self._build_request('X_Var', 'Y_Var')
        res = await analysis_regression(request)

        self.assertEqual(res.significance_class, "not_significant")
        self.assertIn("non_significant_slope", res.reliability_flags)
        
        # Confirm rigorous non-significant overrides (Fixes 1, 2, and 3)
        self.assertEqual(res.direction, "no_clear_relationship")
        self.assertEqual(res.strength_class, "negligible_or_unreliable")
        self.assertIn("not be interpreted as meaningful", res.plain_language_effect)
        self.assertIn("does not appear to be a useful linear predictor", res.summary_interpretation)
        
        warning_texts = " ".join(res.warnings)
        self.assertIn("No statistically reliable linear relationship", warning_texts)


if __name__ == '__main__':
    unittest.main()