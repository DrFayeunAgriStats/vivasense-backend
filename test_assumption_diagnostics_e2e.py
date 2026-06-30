#!/usr/bin/env python3
"""
End-to-End Test for Assumption Diagnostics Implementation

Tests:
1. Backend API response structure (assumption_tests, diagnostic_observations, etc.)
2. Word export rendering with new diagnostic fields
3. Frontend component compatibility with real response data

Run from vivasense-backend/:
    python test_assumption_diagnostics_e2e.py
"""

import base64
import io
import json
import sys
import time
import os
from typing import Any, Dict, Optional

import pandas as pd
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKEND_URL = "https://vivasense-backend-r-production.up.railway.app"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload/dataset"
ANOVA_ENDPOINT = f"{BACKEND_URL}/analysis/anova"
EXPORT_ENDPOINT = f"{BACKEND_URL}/export/anova-word"

# ============================================================================
# TEST DATA BUILDER
# ============================================================================

def create_test_csv_rcbd(
    n_genotypes: int = 4,
    n_reps: int = 3,
    seed: int = 42,
) -> str:
    """
    Create a simple RCBD (Randomized Complete Block Design) test CSV.
    Returns base64-encoded CSV string.
    """
    import random
    random.seed(seed)

    rows = []
    for rep in range(1, n_reps + 1):
        for geno in range(1, n_genotypes + 1):
            # Add realistic variation with some noise
            base_value = 50.0 + (geno - 1) * 5.0 + random.gauss(0, 3)
            rows.append({
                "genotype": f"G{geno:02d}",
                "rep": f"R{rep}",
                "yield": round(base_value, 2),
            })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()
    return csv_b64


# ============================================================================
# TEST 1: UPLOAD DATASET
# ============================================================================

def test_upload_dataset() -> Optional[str]:
    """Upload test dataset and return dataset_token."""
    print("\n" + "=" * 70)
    print("TEST 1: Upload Dataset")
    print("=" * 70)

    csv_b64 = create_test_csv_rcbd()
    print("[PASS] Created test CSV (RCBD: 4 genotypes x 3 reps)")

    payload = {
        "base64_content": csv_b64,
        "file_type": "csv",
        "genotype_column": "genotype",
        "rep_column": "rep",
        "design_type": "rcbd",
        "mode": "single",
    }

    try:
        resp = requests.post(UPLOAD_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()

        result = resp.json()
        dataset_token = result.get("dataset_token")

        print(f"[PASS] Upload successful (HTTP {resp.status_code})")
        print(f"  - dataset_token: {dataset_token[:20]}...")
        print(f"  - n_genotypes: {result.get('n_genotypes')}")
        print(f"  - n_reps: {result.get('n_reps')}")
        print(f"  - n_rows: {result.get('n_rows')}")

        return dataset_token
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Upload failed: {e}")
        return None


# ============================================================================
# TEST 2: ANOVA ANALYSIS
# ============================================================================

def test_anova_analysis(dataset_token: str) -> Optional[Dict[str, Any]]:
    """Run ANOVA analysis and verify assumption_tests + diagnostic fields."""
    print("\n" + "=" * 70)
    print("TEST 2: ANOVA Analysis with Assumption Diagnostics")
    print("=" * 70)

    payload = {
        "dataset_token": dataset_token,
        "trait_columns": ["yield"],
    }

    try:
        resp = requests.post(ANOVA_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()

        result = resp.json()
        print(f"[PASS] ANOVA analysis successful (HTTP {resp.status_code})")

        # Check response structure
        trait_results = result.get("trait_results", {})
        print(f"  - trait_results keys: {list(trait_results.keys())}")

        if "yield" in trait_results:
            yield_result = trait_results["yield"]
            print(f"\n  === YIELD TRAIT RESULT ===")

            # Check core fields
            print(f"  - status: {yield_result.get('status')}")
            print(f"  - grand_mean: {yield_result.get('grand_mean')}")
            print(f"  - n_genotypes: {yield_result.get('n_genotypes')}")
            print(f"  - n_reps: {yield_result.get('n_reps')}")

            # Check ANOVA table
            anova_table = yield_result.get("anova_table")
            if anova_table:
                print(f"\n  === ANOVA TABLE ===")
                print(f"  - sources: {anova_table.get('source', [])}")
                print(f"  - df: {anova_table.get('df', [])}")

            # **CRITICAL: Check new diagnostic fields**
            print(f"\n  === ASSUMPTION TESTS & DIAGNOSTICS ===")

            assumption_tests = yield_result.get("assumption_tests")
            if assumption_tests:
                print(f"  [PASS] assumption_tests present")
                # Check structure
                if isinstance(assumption_tests, dict):
                    print(f"    - Keys: {list(assumption_tests.keys())}")

                    overall = assumption_tests.get("overall", {})
                    print(f"    - overall.passed: {overall.get('passed')}")
                    interp = overall.get('interpretation', '')[:60]
                    print(f"    - overall.interpretation: {interp}...")

                    normality = assumption_tests.get("normality")
                    if normality:
                        print(f"    - normality.test: {normality.get('test')}")
                        print(f"    - normality.statistic: {normality.get('statistic')}")
                        print(f"    - normality.p_value: {normality.get('p_value')}")
                        print(f"    - normality.passed: {normality.get('passed')}")

                    homogeneity = assumption_tests.get("homogeneity")
                    if homogeneity:
                        print(f"    - homogeneity.test: {homogeneity.get('test')}")
                        print(f"    - homogeneity.statistic: {homogeneity.get('statistic')}")
                        print(f"    - homogeneity.p_value: {homogeneity.get('p_value')}")
                        print(f"    - homogeneity.passed: {homogeneity.get('passed')}")

                    outlier_detection = assumption_tests.get("outlier_detection")
                    if outlier_detection:
                        print(f"    - outlier_detection.n_extreme_outliers: {outlier_detection.get('n_extreme_outliers')}")
                        print(f"    - outlier_detection.n_influential_observations: {outlier_detection.get('n_influential_observations')}")
            else:
                print(f"  [FAIL] assumption_tests MISSING")

            diagnostic_observations = yield_result.get("diagnostic_observations")
            if diagnostic_observations:
                print(f"  [PASS] diagnostic_observations present ({len(diagnostic_observations)} records)")
                if len(diagnostic_observations) > 0:
                    first_obs = diagnostic_observations[0]
                    print(f"    - First record keys: {list(first_obs.keys())}")
                    obs_num = first_obs.get('observation')
                    treatment = first_obs.get('treatment')
                    residual = first_obs.get('residual')
                    std_resid = first_obs.get('standardized_residual')
                    cooks_d = first_obs.get('cooks_distance')
                    print(f"    - Example: obs {obs_num}, treatment={treatment}, residual={residual:.3f}, std_resid={std_resid:.3f}, cooks_d={cooks_d:.5f}")
            else:
                print(f"  [FAIL] diagnostic_observations MISSING")

            diagnostic_plots = yield_result.get("diagnostic_plots")
            if diagnostic_plots:
                print(f"  [PASS] diagnostic_plots present")
                print(f"    - Keys: {list(diagnostic_plots.keys())}")
            else:
                print(f"  [FAIL] diagnostic_plots MISSING")

            standardized_residuals = yield_result.get("standardized_residuals")
            if standardized_residuals:
                print(f"  [PASS] standardized_residuals present ({len(standardized_residuals)} values)")
            else:
                print(f"  [FAIL] standardized_residuals MISSING")

            cooks_distance = yield_result.get("cooks_distance")
            if cooks_distance:
                print(f"  [PASS] cooks_distance present ({len(cooks_distance)} values)")
            else:
                print(f"  [FAIL] cooks_distance MISSING")

            outlier_summary = yield_result.get("outlier_summary")
            if outlier_summary:
                print(f"  [PASS] outlier_summary present")
                print(f"    - standardized_residual_threshold: {outlier_summary.get('standardized_residual_threshold')}")
                print(f"    - cooks_distance_threshold: {outlier_summary.get('cooks_distance_threshold')}")
                print(f"    - n_extreme_outliers: {outlier_summary.get('n_extreme_outliers')}")
                print(f"    - n_influential_observations: {outlier_summary.get('n_influential_observations')}")
            else:
                print(f"  [FAIL] outlier_summary MISSING")

        return result
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] ANOVA analysis failed: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text[:500]}")
        return None


# ============================================================================
# TEST 3: WORD EXPORT
# ============================================================================

def test_word_export(anova_response: Dict[str, Any]) -> bool:
    """Test Word export with diagnostic fields present."""
    print("\n" + "=" * 70)
    print("TEST 3: Word Export with Diagnostic Rendering")
    print("=" * 70)

    # Convert ANOVA response to export request format
    export_payload = {
        "dataset_token": anova_response.get("dataset_token"),
        "mode": anova_response.get("mode", "single"),
        "trait_results": anova_response.get("trait_results", {}),
        "failed_traits": anova_response.get("failed_traits", []),
    }

    try:
        resp = requests.post(EXPORT_ENDPOINT, json=export_payload, timeout=60)
        resp.raise_for_status()

        # Check response is binary (docx file)
        content_type = resp.headers.get("content-type", "")
        if "application/vnd.openxmlformats" in content_type or resp.status_code == 200:
            print(f"[PASS] Word export successful (HTTP {resp.status_code})")
            print(f"  - Content-Type: {content_type}")
            print(f"  - Content size: {len(resp.content)} bytes")

            # Save for manual inspection
            output_file = "test_assumption_diagnostics_output.docx"
            with open(output_file, "wb") as f:
                f.write(resp.content)
            print(f"  - Saved to: {output_file}")

            return True
        else:
            print(f"[FAIL] Word export returned unexpected content type: {content_type}")
            print(f"  Response: {resp.text[:500]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Word export failed: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text[:500]}")
        return False


# ============================================================================
# TEST 4: FRONTEND DATA COMPATIBILITY
# ============================================================================

def test_frontend_compatibility(anova_response: Dict[str, Any]) -> bool:
    """Verify the response structure is compatible with AssumptionDiagnosticsDashboard."""
    print("\n" + "=" * 70)
    print("TEST 4: Frontend Component Data Compatibility")
    print("=" * 70)

    trait_results = anova_response.get("trait_results", {})

    if not trait_results:
        print("[FAIL] No trait_results in response")
        return False

    all_pass = True
    for trait_name, trait_result in trait_results.items():
        print(f"\n  Trait: {trait_name}")

        # Check required fields for AssumptionDiagnosticsDashboard
        required_fields = [
            "assumption_tests",
            "diagnostic_observations",
            "diagnostic_plots",
            "standardized_residuals",
            "cooks_distance",
            "outlier_summary",
        ]

        for field in required_fields:
            value = trait_result.get(field)
            if value is not None:
                print(f"    [PASS] {field}")
            else:
                print(f"    [FAIL] {field} (NULL)")
                all_pass = False

        # Validate assumption_tests structure
        assumption_tests = trait_result.get("assumption_tests")
        if assumption_tests and isinstance(assumption_tests, dict):
            expected_keys = ["overall", "normality", "homogeneity", "outlier_detection", "reviewer_mode"]
            found_keys = [k for k in expected_keys if k in assumption_tests]
            print(f"    - assumption_tests sub-keys: {found_keys} (expected {expected_keys})")

        # Validate diagnostic_observations structure
        diagnostic_obs = trait_result.get("diagnostic_observations")
        if diagnostic_obs and isinstance(diagnostic_obs, list) and len(diagnostic_obs) > 0:
            first_record = diagnostic_obs[0]
            expected_obs_keys = [
                "observation", "treatment", "block", "observed", "fitted",
                "residual", "standardized_residual", "cooks_distance",
                "extreme_outlier", "influential"
            ]
            found_obs_keys = [k for k in expected_obs_keys if k in first_record]
            print(f"    - diagnostic_observations[0] has {len(found_obs_keys)}/{len(expected_obs_keys)} keys")
            if len(found_obs_keys) < len(expected_obs_keys):
                missing = [k for k in expected_obs_keys if k not in first_record]
                print(f"      Missing keys: {missing}")
                all_pass = False

    if all_pass:
        print(f"\n  [PASS] All frontend compatibility checks passed")
    else:
        print(f"\n  [FAIL] Some frontend compatibility checks failed")

    return all_pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("ASSUMPTION DIAGNOSTICS END-TO-END TEST")
    print("=" * 70)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "test_1_upload": False,
        "test_2_anova": False,
        "test_3_export": False,
        "test_4_frontend": False,
    }

    # Test 1: Upload
    dataset_token = test_upload_dataset()
    results["test_1_upload"] = dataset_token is not None

    if not dataset_token:
        print("\n[FAIL] Cannot proceed without dataset_token")
        print("\nFINAL RESULTS:")
        for test, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status}: {test}")
        return 1

    # Test 2: ANOVA
    anova_response = test_anova_analysis(dataset_token)
    results["test_2_anova"] = anova_response is not None

    if not anova_response:
        print("\n[FAIL] Cannot proceed without ANOVA response")
        print("\nFINAL RESULTS:")
        for test, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status}: {test}")
        return 1

    # Test 3: Word Export
    results["test_3_export"] = test_word_export(anova_response)

    # Test 4: Frontend Compatibility
    results["test_4_frontend"] = test_frontend_compatibility(anova_response)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {test}")

    all_passed = all(results.values())
    print("\n" + ("=" * 70))
    if all_passed:
        print("[PASS] ALL TESTS PASSED - Ready for commit!")
        print("=" * 70)
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED - Review above for details")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
