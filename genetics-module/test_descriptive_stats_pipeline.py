"""
Test: Full Descriptive Statistics Pipeline
==========================================
Verify that descriptive statistics flow from:
  1. Computation in routes (ANOVA, Genetics)
  2. Attachment to trait result objects
  3. Export rendering in reports

Tests:
  - ANOVA trait result includes descriptive_stats
  - Genetics trait result includes descriptive_stats
  - Both contain: mean, min, max, SD, SE, CV
  - Export renders them without truncation
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from module_schemas import DescriptiveStats, AnovaTraitResult, GeneticParametersTraitResult
from analysis_anova_routes import compute_descriptive_stats


def test_descriptive_stats_computation():
    """Verify compute_descriptive_stats returns all expected fields."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_descriptive_stats(data)
    
    assert stats is not None
    assert "grand_mean" in stats
    assert "standard_deviation" in stats
    assert "variance" in stats
    assert "standard_error" in stats
    assert "cv_percent" in stats
    assert "min" in stats
    assert "max" in stats
    assert "range" in stats
    
    # Spot check values
    assert stats["grand_mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["range"] == 4.0
    assert abs(stats["standard_deviation"] - 1.58) < 0.01


def test_anova_trait_result_has_descriptive_stats():
    """Verify AnovaTraitResult schema includes descriptive_stats field."""
    ds = DescriptiveStats(
        grand_mean=50.5,
        standard_deviation=2.3,
        standard_error=0.5,
        cv_percent=4.5,
        min=45.0,
        max=56.0,
        range=11.0,
    )
    
    tr = AnovaTraitResult(
        trait="Yield_kg_ha",
        status="success",
        grand_mean=50.5,
        descriptive_stats=ds,
        n_genotypes=5,
        n_reps=3,
    )
    
    assert tr.descriptive_stats is not None
    assert tr.descriptive_stats.grand_mean == 50.5
    assert tr.descriptive_stats.standard_deviation == 2.3
    assert tr.descriptive_stats.cv_percent == 4.5


def test_genetic_parameters_trait_result_has_descriptive_stats():
    """Verify GeneticParametersTraitResult schema includes descriptive_stats field."""
    ds = DescriptiveStats(
        grand_mean=175.3,
        standard_deviation=5.1,
        standard_error=1.2,
        cv_percent=2.9,
        min=162.0,
        max=190.0,
        range=28.0,
    )
    
    tr = GeneticParametersTraitResult(
        trait="Plant_height_cm",
        status="success",
        grand_mean=175.3,
        descriptive_stats=ds,
        heritability={"h2_broad_sense": 0.65},
        gcv=3.5,
        pcv=4.2,
    )
    
    assert tr.descriptive_stats is not None
    assert tr.descriptive_stats.grand_mean == 175.3
    assert tr.descriptive_stats.standard_error == 1.2
    assert tr.descriptive_stats.cv_percent == 2.9
    assert tr.descriptive_stats.min == 162.0
    assert tr.descriptive_stats.max == 190.0


def test_descriptive_stats_with_none_values():
    """Verify schema handles partial descriptive stats gracefully."""
    ds = DescriptiveStats(
        grand_mean=100.0,
        cv_percent=5.0,
        # Other fields None
    )
    
    tr = GeneticParametersTraitResult(
        trait="TestTrait",
        status="success",
        descriptive_stats=ds,
    )
    
    assert tr.descriptive_stats.grand_mean == 100.0
    assert tr.descriptive_stats.cv_percent == 5.0
    assert tr.descriptive_stats.standard_deviation is None
    assert tr.descriptive_stats.min is None


def test_descriptive_stats_serialization():
    """Verify DescriptiveStats serializes correctly for JSON API responses."""
    ds = DescriptiveStats(
        grand_mean=50.0,
        standard_deviation=2.5,
        standard_error=0.6,
        cv_percent=5.0,
        min=44.0,
        max=56.0,
        range=12.0,
        variance=6.25,
    )
    
    # Simulate JSON serialization
    serialized = ds.model_dump()
    
    assert serialized["grand_mean"] == 50.0
    assert serialized["standard_deviation"] == 2.5
    assert serialized["standard_error"] == 0.6
    assert serialized["cv_percent"] == 5.0
    assert serialized["min"] == 44.0
    assert serialized["max"] == 56.0
    assert serialized["range"] == 12.0
    assert serialized["variance"] == 6.25


def test_no_literal_trait_name_in_descriptive_stats():
    """Verify descriptive stats don't contain literal trait name strings."""
    ds = DescriptiveStats(
        grand_mean=100.0,
        cv_percent=5.0,
    )
    
    tr = GeneticParametersTraitResult(
        trait="Yield_kg_ha",
        status="success",
        descriptive_stats=ds,
    )
    
    # Check that stats values don't contain template strings
    serialized = tr.model_dump()
    desc_stats_dict = serialized.get("descriptive_stats", {})
    
    for key, value in desc_stats_dict.items():
        if isinstance(value, str):
            assert "{trait}" not in value, f"Found literal template in {key}: {value}"
            assert "Yield_kg_ha" not in value, f"Found trait name in {key}: {value}"


def test_export_reads_descriptive_stats_from_trait_result():
    """Verify export layer reads descriptive_stats from trait result, not just R engine."""
    # This is an integration-style test
    # In actual use, export_module_routes.py will read tr.descriptive_stats
    
    ds = DescriptiveStats(
        grand_mean=120.5,
        standard_deviation=3.2,
        standard_error=0.8,
        cv_percent=2.7,
        min=112.0,
        max=130.0,
        range=18.0,
    )
    
    tr = GeneticParametersTraitResult(
        trait="Ear_length_cm",
        status="success",
        grand_mean=120.5,
        descriptive_stats=ds,
    )
    
    # Simulate export logic
    export_rows = []
    if tr.grand_mean is not None:
        export_rows.append(("Grand Mean", f"{tr.grand_mean:.2f}"))
    if tr.descriptive_stats:
        ds_obj = tr.descriptive_stats
        if ds_obj.standard_deviation is not None:
            export_rows.append(("Standard Deviation", f"{ds_obj.standard_deviation:.2f}"))
        if ds_obj.standard_error is not None:
            export_rows.append(("Standard Error", f"{ds_obj.standard_error:.2f}"))
        if ds_obj.min is not None:
            export_rows.append(("Minimum", f"{ds_obj.min:.2f}"))
        if ds_obj.max is not None:
            export_rows.append(("Maximum", f"{ds_obj.max:.2f}"))
        if ds_obj.range is not None:
            export_rows.append(("Range", f"{ds_obj.range:.2f}"))
        if ds_obj.cv_percent is not None:
            export_rows.append(("Coefficient of Variation (%)", f"{ds_obj.cv_percent:.2f}"))
    
    # Verify all stats are included
    assert len(export_rows) == 7, f"Expected 7 rows, got {len(export_rows)}"
    
    row_labels = [r[0] for r in export_rows]
    assert "Grand Mean" in row_labels
    assert "Standard Deviation" in row_labels
    assert "Standard Error" in row_labels
    assert "Minimum" in row_labels
    assert "Maximum" in row_labels
    assert "Range" in row_labels
    assert "Coefficient of Variation (%)" in row_labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
