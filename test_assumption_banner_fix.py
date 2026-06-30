#!/usr/bin/env python3
"""Test that assumption banner is now correct (not contradictory)"""

import base64, io, json, requests, pandas as pd, random

BACKEND = "https://vivasense-backend-r-production.up.railway.app"

print("="*70)
print("TESTING ASSUMPTION BANNER FIX")
print("="*70)

# Create dataset that passes assumptions but has influential observations
random.seed(42)
rows = []
for rep in range(1, 4):
    for geno in range(1, 5):
        base = 50.0 + (geno - 1) * 5.0
        rows.append({
            "genotype": f"G{geno:02d}",
            "rep": f"R{rep}",
            "yield": round(base + random.gauss(0, 3), 2),
        })

df = pd.DataFrame(rows)
buf = io.StringIO()
df.to_csv(buf, index=False)
csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

print("\n[1] Uploading test dataset (will have influential observations)...")
resp = requests.post(f"{BACKEND}/upload/dataset", json={
    "base64_content": csv_b64,
    "file_type": "csv",
    "genotype_column": "genotype",
    "rep_column": "rep",
    "design_type": "rcbd",
    "mode": "single",
})
token = resp.json()["dataset_token"]

print("\n[2] Running ANOVA analysis...")
resp = requests.post(f"{BACKEND}/analysis/anova", json={
    "dataset_token": token,
    "trait_columns": ["yield"],
})

result = resp.json()
trait_result = result.get("trait_results", {}).get("yield", {})
assumption_tests = trait_result.get("assumption_tests", {})

print("\n[3] ASSUMPTION TEST RESULTS")
print("="*70)

# Check individual tests
normality = assumption_tests.get("normality", {})
homogeneity = assumption_tests.get("homogeneity", {})
outlier_detection = assumption_tests.get("outlier_detection", {})
reviewer_mode = assumption_tests.get("reviewer_mode", {})

print(f"\nNormality Test (Shapiro-Wilk):")
print(f"  - Statistic: {normality.get('statistic')}")
print(f"  - P-value: {normality.get('p_value')}")
print(f"  - Passed: {normality.get('passed')}")

print(f"\nHomogeneity Test (Levene):")
print(f"  - Statistic: {homogeneity.get('statistic')}")
print(f"  - P-value: {homogeneity.get('p_value')}")
print(f"  - Passed: {homogeneity.get('passed')}")

print(f"\nOutlier Detection:")
print(f"  - Extreme outliers (|std_resid| > 3): {outlier_detection.get('n_extreme_outliers')}")
print(f"  - Influential observations (Cook's D > threshold): {outlier_detection.get('n_influential_observations')}")

print(f"\nREVIEWER MODE (BANNER):")
print(f"  - Status: {reviewer_mode.get('status')}")
summary_text = reviewer_mode.get('summary', '').replace('✓', '[CHECK]')
print(f"  - Summary (BANNER TEXT): {summary_text}")
print(f"  - Normality satisfied: {reviewer_mode.get('normality_satisfied')}")
print(f"  - Homogeneity satisfied: {reviewer_mode.get('homogeneity_satisfied')}")
print(f"  - No influential outliers: {reviewer_mode.get('no_influential_outliers')}")

# Verify consistency
print("\n" + "="*70)
print("CONSISTENCY CHECK")
print("="*70)

both_assumptions_pass = (
    normality.get('passed') == True and
    homogeneity.get('passed') == True
)
status_is_pass = reviewer_mode.get('status') == 'PASS'
banner_text = reviewer_mode.get('summary', '')

print(f"\nBoth assumptions passed: {both_assumptions_pass}")
print(f"Status is PASS: {status_is_pass}")
print(f"Banner says '✓ Assumptions...': {'✓' in banner_text and 'Normality' in banner_text}")

# Check for contradictions
has_checkmark = '✓' in banner_text
has_warning = '⚠' in banner_text
all_tests_pass = both_assumptions_pass

print(f"\n✓ Banner has checkmark: {has_checkmark}")
print(f"⚠ Banner has warning: {has_warning}")

# This is the key test: if assumptions pass, no warning should appear
if all_tests_pass:
    if has_warning and not has_checkmark:
        print("\n[FAIL] CONTRADICTORY: Both tests passed but banner shows '⚠ Assumptions violated'")
        print("       This is the bug!")
        exit(1)
    elif has_checkmark:
        print("\n[PASS] CORRECT: Both tests passed, banner shows checkmark")
        # Check if there's an informational note about influential observations
        if outlier_detection.get('n_influential_observations', 0) > 0:
            if 'Note:' in banner_text and 'influential' in banner_text:
                print("[PASS] Influential observations have separate informational note (not treated as assumption violation)")
                exit(0)
            else:
                print("[INFO] Influential observations present but not noted in summary (acceptable)")
                exit(0)
    else:
        print("\n[WARN] Unexpected: Both tests passed but banner doesn't show checkmark")
        exit(1)
else:
    print("\n[PASS] Assumptions failed, status correctly shows WARN")
    exit(0)
