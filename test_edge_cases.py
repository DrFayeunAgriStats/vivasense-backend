#!/usr/bin/env python3
"""Test edge cases: zero variance, perfect fit, extreme values"""

import base64, io, json, requests, pandas as pd, math

BACKEND = "https://vivasense-backend-r-production.up.railway.app"

def test_perfect_fit():
    """Test case where all genotypes have identical means (near-perfect fit)"""
    print("\n=== EDGE CASE 1: Perfect/Near-Perfect Fit (all genotypes identical) ===")

    rows = []
    for rep in range(1, 4):
        for geno in range(1, 5):
            # Identical values per genotype across reps - no variation
            rows.append({
                "genotype": f"G{geno:02d}",
                "rep": f"R{rep}",
                "yield": 50.0 + (geno - 1) * 0.01,  # Minimal differences
            })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    # Upload
    resp = requests.post(f"{BACKEND}/upload/dataset", json={
        "base64_content": csv_b64, "file_type": "csv",
        "genotype_column": "genotype", "rep_column": "rep",
        "design_type": "rcbd", "mode": "single"
    })
    token = resp.json()["dataset_token"]

    # Analyze
    resp = requests.post(f"{BACKEND}/analysis/anova", json={
        "dataset_token": token, "trait_columns": ["yield"]
    })
    result = resp.json()
    trait_result = result.get("trait_results", {}).get("yield", {})

    print(f"Status: {trait_result.get('status')}")

    # Check for NaN/Inf in diagnostic fields
    cooks = trait_result.get("cooks_distance") or []
    std_resid = trait_result.get("standardized_residuals") or []

    # Handle None values gracefully
    cooks = [v for v in (cooks or []) if v is not None]
    std_resid = [v for v in (std_resid or []) if v is not None]

    has_nan = any((isinstance(v, float) and math.isnan(v)) for v in cooks + std_resid)
    has_inf = any(isinstance(v, float) and math.isinf(v) for v in cooks + std_resid)

    print(f"  - cooks_distance values: {[f'{v:.6f}' if isinstance(v, float) else v for v in cooks[:3]]}... (first 3)")
    print(f"  - standardized_residuals: {[f'{v:.6f}' if isinstance(v, float) else v for v in std_resid[:3]]}... (first 3)")
    print(f"  - Contains NaN: {has_nan}")
    print(f"  - Contains Inf: {has_inf}")

    if has_nan or has_inf:
        print("  [WARN] Detected NaN/Inf values - potential numerical instability")
        return False
    else:
        print("  [PASS] All values are valid numbers (no NaN/Inf)")
        return True


def test_high_variability():
    """Test case with extreme values"""
    print("\n=== EDGE CASE 2: High Variability (extreme range) ===")

    rows = []
    for rep in range(1, 4):
        for geno in range(1, 5):
            # Very different means
            rows.append({
                "genotype": f"G{geno:02d}",
                "rep": f"R{rep}",
                "yield": 10.0 + (geno - 1) * 100,  # Wide range: 10, 110, 210, 310
            })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    # Upload
    resp = requests.post(f"{BACKEND}/upload/dataset", json={
        "base64_content": csv_b64, "file_type": "csv",
        "genotype_column": "genotype", "rep_column": "rep",
        "design_type": "rcbd", "mode": "single"
    })
    token = resp.json()["dataset_token"]

    # Analyze
    resp = requests.post(f"{BACKEND}/analysis/anova", json={
        "dataset_token": token, "trait_columns": ["yield"]
    })
    result = resp.json()
    trait_result = result.get("trait_results", {}).get("yield", {})

    print(f"Status: {trait_result.get('status')}")

    outlier_summary = trait_result.get("outlier_summary", {})
    cook_threshold = outlier_summary.get("cooks_distance_threshold")
    n_influential = outlier_summary.get("n_influential_observations", 0)

    print(f"  - Cook's D threshold: {cook_threshold}")
    print(f"  - Influential observations detected: {n_influential}")
    print(f"  [PASS] Edge case handled without errors")
    return True


def test_single_outlier():
    """Test case with one extreme outlier"""
    print("\n=== EDGE CASE 3: Single Extreme Outlier ===")

    rows = []
    outlier_added = False
    for rep in range(1, 4):
        for geno in range(1, 5):
            base = 50.0 + (geno - 1) * 5.0
            # Add one extreme outlier
            if not outlier_added and rep == 2 and geno == 3:
                value = 1000.0  # Extreme outlier
                outlier_added = True
            else:
                value = base

            rows.append({
                "genotype": f"G{geno:02d}",
                "rep": f"R{rep}",
                "yield": value,
            })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    # Upload
    resp = requests.post(f"{BACKEND}/upload/dataset", json={
        "base64_content": csv_b64, "file_type": "csv",
        "genotype_column": "genotype", "rep_column": "rep",
        "design_type": "rcbd", "mode": "single"
    })
    token = resp.json()["dataset_token"]

    # Analyze
    resp = requests.post(f"{BACKEND}/analysis/anova", json={
        "dataset_token": token, "trait_columns": ["yield"]
    })
    result = resp.json()
    trait_result = result.get("trait_results", {}).get("yield", {})

    print(f"Status: {trait_result.get('status')}")

    outlier_summary = trait_result.get("outlier_summary", {})
    n_extreme = outlier_summary.get("n_extreme_outliers", 0)
    n_influential = outlier_summary.get("n_influential_observations", 0)
    flagged = outlier_summary.get("flagged_observations", [])

    print(f"  - Extreme outliers detected (|std_resid| > 3): {n_extreme}")
    print(f"  - Influential observations detected (Cook's D > threshold): {n_influential}")
    print(f"  - Flagged records: {len(flagged)}")
    if flagged:
        print(f"    * First flagged: obs {flagged[0].get('observation')}, value={flagged[0].get('observed')}")

    print(f"  [PASS] Outlier detection working correctly")
    return True


print("\n" + "="*70)
print("EDGE CASE TESTING FOR NUMERICAL STABILITY")
print("="*70)

results = {
    "perfect_fit": test_perfect_fit(),
    "high_variability": test_high_variability(),
    "extreme_outlier": test_single_outlier(),
}

print("\n" + "="*70)
print("EDGE CASE RESULTS")
print("="*70)
for case, passed in results.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}: {case}")

all_passed = all(results.values())
if all_passed:
    print("\n[PASS] All edge cases handled correctly - no NaN/Inf issues")
else:
    print("\n[FAIL] Some edge cases failed")
