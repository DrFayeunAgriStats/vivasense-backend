"""
Unit tests for analysis_trait_association_routes.py

Tests the trait association analysis logic including:
- Successful two-trait analysis
- Invalid trait input handling
- Classification functions
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import base64
import pandas as pd
import io

# Import the module under test
from analysis_trait_association_routes import (
    _classify_strength,
    _classify_direction,
    _classify_confidence,
    _classify_selection_relevance,
    _compute_risk_flags,
    _build_matrix_dict,
    analyze_trait_association,
    router,
)


class TestTraitAssociationClassifications(unittest.TestCase):
    """Test the classification helper functions."""

    def test_classify_strength(self):
        self.assertEqual(_classify_strength(0.95), "very strong")
        self.assertEqual(_classify_strength(-0.95), "very strong")
        self.assertEqual(_classify_strength(0.75), "strong")
        self.assertEqual(_classify_strength(0.45), "moderate")
        self.assertEqual(_classify_strength(0.25), "weak")
        self.assertEqual(_classify_strength(0.15), "very weak")
        self.assertEqual(_classify_strength(0.0), "very weak")

    def test_classify_direction(self):
        self.assertEqual(_classify_direction(0.5), "positive")
        self.assertEqual(_classify_direction(-0.5), "negative")
        self.assertEqual(_classify_direction(0.0), "none")

    def test_classify_confidence(self):
        # Beta safety: pairwise N is not tracked, so confidence is limited.
        self.assertEqual(_classify_confidence(25, "plot_level", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")
        self.assertEqual(_classify_confidence(20, "plot_level", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")
        self.assertEqual(_classify_confidence(15, "plot_level", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")
        self.assertEqual(_classify_confidence(10, "plot_level", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")
        self.assertEqual(_classify_confidence(5, "plot_level", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")

        # Adjustments for genotype_mean (currently no effect due to pairwise N limitation)
        self.assertEqual(_classify_confidence(25, "genotype_mean", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")
        self.assertEqual(_classify_confidence(25, "plot_level", False, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")

        # Adjustments for GxE (currently no effect due to pairwise N limitation)
        self.assertEqual(_classify_confidence(25, "plot_level", True, ["pairwise_n_not_tracked"]), "limited_by_pairwise_n")

    def test_classify_selection_relevance(self):
        # Exploratory only
        self.assertEqual(_classify_selection_relevance(0.3, 0.1, 0.05, 15), "exploratory only")

        # Potentially useful with validation
        self.assertEqual(_classify_selection_relevance(0.8, 0.01, 0.05, 5), "potentially useful with validation")
        self.assertEqual(_classify_selection_relevance(0.8, 0.01, 0.05, 15), "useful with validation")

    def test_compute_risk_flags(self):
        # No flags (except pairwise_n_not_tracked)
        self.assertEqual(_compute_risk_flags(15, "plot_level", False), ["pairwise_n_not_tracked"])

        # Small sample size
        self.assertEqual(_compute_risk_flags(5, "plot_level", False), ["small_sample_size", "pairwise_n_not_tracked"])

        # Genotype mean
        self.assertEqual(_compute_risk_flags(15, "genotype_mean", False), ["genotype_mean_based", "pairwise_n_not_tracked"])

        # GxE significant
        self.assertEqual(_compute_risk_flags(15, "plot_level", True), ["gxe_significant", "pairwise_n_not_tracked"])

        # Multiple flags
        self.assertEqual(_compute_risk_flags(5, "genotype_mean", True), ["small_sample_size", "genotype_mean_based", "gxe_significant", "pairwise_n_not_tracked"])

    def test_build_matrix_dict(self):
        trait_names = ["A", "B", "C"]
        matrix = [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ]

        expected = {
            "A": {"A": 1.0, "B": 0.5, "C": 0.2},
            "B": {"A": 0.5, "B": 1.0, "C": 0.3},
            "C": {"A": 0.2, "B": 0.3, "C": 1.0}
        }

        self.assertEqual(_build_matrix_dict(trait_names, matrix), expected)


class TestTraitAssociationEndpoint(unittest.IsolatedAsyncioTestCase):
    """Test the trait association endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock dataset
        self.test_df = pd.DataFrame({
            'genotype': ['G1', 'G1', 'G2', 'G2', 'G3', 'G3'],
            'rep': ['R1', 'R2', 'R1', 'R2', 'R1', 'R2'],
            'Yield': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            'Height': [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
        })

        # Mock dataset cache
        self.mock_ctx = {
            'base64_content': base64.b64encode(self.test_df.to_csv(index=False).encode()).decode(),
            'file_type': 'csv',
            'genotype_column': 'genotype',
            'rep_column': 'rep',
            'environment_column': None,
            'mode': 'single'
        }

        # Mock R engine response
        self.mock_r_response = {
            'trait_names': ['Yield', 'Height'],
            'n_observations': 3,
            'r_matrix': [[1.0, 0.99], [0.99, 1.0]],
            'p_matrix': [[0.0, 0.001], [0.001, 0.0]],
            'warnings': []
        }

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_successful_two_trait_analysis_fixed(self, mock_tr_engine, mock_dataset_cache):
        """Test successful analysis with two traits."""
        # Setup mocks
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        mock_tr_engine.run_correlation.return_value = self.mock_r_response

        # Create test request
        from module_schemas import TraitAssociationModuleRequest
        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        # Await the endpoint execution natively
        response = await analyze_trait_association(request)
        
        self.assertEqual(response.n_observations, 3)
        self.assertEqual(len(response.significant_pairs), 1)
        self.assertEqual(response.significant_pairs[0].trait_1, "Yield")
        self.assertEqual(response.significant_pairs[0].trait_2, "Height")
        self.assertEqual(response.strongest_positive_pair.r, 0.99)
        self.assertIsNone(response.strongest_negative_pair)
        # Check flags based on corrected logic
        self.assertIn("genotype_mean_based", response.risk_flags)
        self.assertIn("pairwise_n_not_tracked", response.risk_flags)
        self.assertEqual(response.significant_pairs[0].confidence_status, "limited_by_pairwise_n")
        self.assertEqual(response.significant_pairs[0].confidence, "limited_by_pairwise_n")
        self.assertEqual(response.significant_pairs[0].selection_signal, "potentially useful with validation")
        self.assertEqual(response.significant_pairs[0].selection_relevance, "potentially useful with validation")

    async def test_insufficient_traits_validation(self):
        """Test that endpoint validation requires at least 2 traits."""
        from module_schemas import TraitAssociationModuleRequest
        from fastapi import HTTPException

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )
        
        with self.assertRaises(HTTPException) as cm:
            await analyze_trait_association(request)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("at least 2 trait columns", cm.exception.detail)

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_all_negative_correlations(self, mock_tr_engine, mock_dataset_cache):
        """Test analysis with all negative correlations."""
        # Setup mocks with negative correlations
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        mock_r_response_negative = {
            'trait_names': ['Yield', 'Height'],
            'n_observations': 3,
            'r_matrix': [[1.0, -0.95], [-0.95, 1.0]],
            'p_matrix': [[0.0, 0.001], [0.001, 0.0]],
            'warnings': []
        }
        mock_tr_engine.run_correlation.return_value = mock_r_response_negative

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='plot_level',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        response = await analyze_trait_association(request)
        
        self.assertIsNone(response.strongest_positive_pair)
        self.assertEqual(response.strongest_negative_pair.r, -0.95)
        self.assertEqual(response.strongest_negative_pair.trait_1, "Yield")
        self.assertEqual(response.strongest_negative_pair.trait_2, "Height")

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_all_positive_correlations(self, mock_tr_engine, mock_dataset_cache):
        """Test analysis with all positive correlations."""
        # Setup mocks (using the existing positive correlation mock)
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        mock_tr_engine.run_correlation.return_value = self.mock_r_response

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='plot_level',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        response = await analyze_trait_association(request)
        
        self.assertEqual(response.strongest_positive_pair.r, 0.99)
        self.assertEqual(response.strongest_positive_pair.trait_1, "Yield")
        self.assertEqual(response.strongest_positive_pair.trait_2, "Height")
        self.assertIsNone(response.strongest_negative_pair)

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_constant_column_handling(self, mock_tr_engine, mock_dataset_cache):
        """Test handling of constant columns (undefined correlations)."""
        # Setup mocks with NaN correlations (constant columns)
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        mock_r_response_nan = {
            'trait_names': ['Yield', 'Height', 'Constant'],
            'n_observations': 3,
            'r_matrix': [[1.0, None, None], [None, 1.0, None], [None, None, 1.0]],
            'p_matrix': [[0.0, None, None], [None, 0.0, None], [None, None, 0.0]],
            'warnings': ['Constant column detected']
        }
        mock_tr_engine.run_correlation.return_value = mock_r_response_nan

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height', 'Constant'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        response = await analyze_trait_association(request)
        
        # Should have no significant pairs due to NaN correlations
        self.assertEqual(len(response.significant_pairs), 0)
        self.assertIsNone(response.strongest_positive_pair)
        self.assertIsNone(response.strongest_negative_pair)
        self.assertIn('Constant column detected', response.warnings)

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_no_significant_pairs(self, mock_tr_engine, mock_dataset_cache):
        """Test analysis with no significant pairs (high p-values)."""
        # Setup mocks with high p-values
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        mock_r_response_no_sig = {
            'trait_names': ['Yield', 'Height'],
            'n_observations': 3,
            'r_matrix': [[1.0, 0.3], [0.3, 1.0]],
            'p_matrix': [[0.0, 0.8], [0.8, 0.0]],  # p > 0.05
            'warnings': []
        }
        mock_tr_engine.run_correlation.return_value = mock_r_response_no_sig

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        response = await analyze_trait_association(request)
        
        self.assertEqual(len(response.significant_pairs), 0)
        # Still track strongest correlations even if not significant
        self.assertEqual(response.strongest_positive_pair.r, 0.3)
        self.assertIsNone(response.strongest_negative_pair)

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_genotype_mean_classification_path(self, mock_tr_engine, mock_dataset_cache):
        """Test genotype_mean analysis path with appropriate classifications."""
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        mock_tr_engine.run_correlation.return_value = self.mock_r_response

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        response = await analyze_trait_association(request)
        
        # Check that confidence status is limited for beta safety
        self.assertEqual(response.significant_pairs[0].confidence_status, "limited_by_pairwise_n")
        self.assertEqual(response.significant_pairs[0].confidence, "limited_by_pairwise_n")
        # Check risk flags include genotype_mean_based
        self.assertIn("genotype_mean_based", response.risk_flags)
        self.assertIn("pairwise_n_not_tracked", response.risk_flags)
        # Check selection signal is capped for beta safety
        self.assertEqual(response.significant_pairs[0].selection_signal, "potentially useful with validation")
        self.assertEqual(response.significant_pairs[0].selection_relevance, "potentially useful with validation")

    @patch('analysis_trait_association_routes.dataset_cache')
    async def test_missing_dataset_404(self, mock_dataset_cache):
        """Test that endpoint returns 404 for missing dataset token."""
        from module_schemas import TraitAssociationModuleRequest
        from fastapi import HTTPException

        # Setup mock to return None (dataset not found)
        mock_dataset_cache.get_dataset.return_value = None

        request = TraitAssociationModuleRequest(
            dataset_token='missing_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        with self.assertRaises(HTTPException) as cm:
            await analyze_trait_association(request)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertIn("not found", cm.exception.detail)

    @patch('analysis_trait_association_routes.dataset_cache')
    async def test_invalid_dataset_400(self, mock_dataset_cache):
        """Test that endpoint returns 400 for invalid/corrupted dataset."""
        from module_schemas import TraitAssociationModuleRequest
        from fastapi import HTTPException

        # Setup mock with invalid base64 content
        invalid_ctx = {
            'base64_content': 'invalid_base64!',
            'file_type': 'csv',
            'genotype_column': 'genotype',
            'rep_column': 'rep',
            'environment_column': None,
            'mode': 'single'
        }
        mock_dataset_cache.get_dataset.return_value = invalid_ctx

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        with self.assertRaises(HTTPException) as cm:
            await analyze_trait_association(request)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("Could not read dataset", cm.exception.detail)

    @patch('analysis_trait_association_routes.dataset_cache')
    @patch('analysis_trait_association_routes.tr_engine')
    async def test_engine_failure_500(self, mock_tr_engine, mock_dataset_cache):
        """Test that endpoint returns 500 when R engine fails."""
        from module_schemas import TraitAssociationModuleRequest
        from fastapi import HTTPException

        # Setup mocks
        mock_dataset_cache.get_dataset.return_value = self.mock_ctx
        # Make the engine raise a RuntimeError
        mock_tr_engine.run_correlation.side_effect = RuntimeError("R engine crashed")

        request = TraitAssociationModuleRequest(
            dataset_token='test_token',
            trait_columns=['Yield', 'Height'],
            analysis_unit='genotype_mean',
            alpha=0.05,
            gxe_significant=False,
            environment_context='single_environment'
        )

        with self.assertRaises(HTTPException) as cm:
            await analyze_trait_association(request)
        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("Trait association analysis failed", cm.exception.detail)


if __name__ == '__main__':
    unittest.main()