#!/usr/bin/env python3
"""Test R diagnostic computation with error capture"""

import base64, io, json, requests, pandas as pd, random

BACKEND = "https://vivasense-backend-r-production.up.railway.app"

# Create simple RCBD data
random.seed(42)
rows = []
for rep in range(1, 4):
    for geno in range(1, 5):
        rows.append({
            "genotype": f"G{geno:02d}",
            "rep": f"R{rep}",
            "yield": round(50.0 + (geno - 1) * 5.0 + random.gauss(0, 3), 2),
        })
df = pd.DataFrame(rows)
buf = io.StringIO()
df.to_csv(buf, index=False)
csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

print("[1] Uploading test data...")
resp = requests.post(f"{BACKEND}/upload/dataset", json={
    "base64_content": csv_b64,
    "file_type": "csv",
    "genotype_column": "genotype",
    "rep_column": "rep",
    "design_type": "rcbd",
    "mode": "single",
})
token = resp.json()["dataset_token"]
print(f"[OK] Token: {token[:20]}...")

print("\n[2] Running ANOVA (will capture any R debug output)...")
resp = requests.post(f"{BACKEND}/analysis/anova", json={
    "dataset_token": token,
    "trait_columns": ["yield"],
})

result = resp.json()
trait_result = result.get("trait_results", {}).get("yield", {})

print("\n[3] Response diagnostic fields:")
print(f"  - diagnostic_observations: {trait_result.get('diagnostic_observations')}")
print(f"  - diagnostic_plots: {trait_result.get('diagnostic_plots')}")
print(f"  - standardized_residuals: {trait_result.get('standardized_residuals')}")
print(f"  - cooks_distance: {trait_result.get('cooks_distance')}")
print(f"  - outlier_summary: {trait_result.get('outlier_summary')}")

# Check related fields that ARE working
print("\n[4] Related fields that ARE present:")
print(f"  - residuals length: {len(trait_result.get('residuals', []))}")
print(f"  - fitted_values length: {len(trait_result.get('fitted_values', []))}")
print(f"  - assumption_tests keys: {list(trait_result.get('assumption_tests', {}).keys())}")

# The mystery: why do residuals/fitted_values work but not the others?
print("\n[5] HYPOTHESIS: diag_n must be 0 in R (diagnostic arrays failed)")
print("    This would explain why assumption_tests works (separate code path)")
print("    But residuals/fitted_values ARE in response, which contradicts this...")
print("\n    => There may be a code path issue or the deployed code is different")
