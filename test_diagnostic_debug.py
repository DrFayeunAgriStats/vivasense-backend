#!/usr/bin/env python3
"""
Debug script to check what's happening in R diagnostic computation
"""

import base64
import io
import json
import requests
import pandas as pd
import random

BACKEND_URL = "https://vivasense-backend-r-production.up.railway.app"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload/dataset"
ANOVA_ENDPOINT = f"{BACKEND_URL}/analysis/anova"

# Create test CSV
def create_test_csv(seed=42):
    random.seed(seed)
    rows = []
    for rep in range(1, 4):
        for geno in range(1, 5):
            base_value = 50.0 + (geno - 1) * 5.0 + random.gauss(0, 3)
            rows.append({
                "genotype": f"G{geno:02d}",
                "rep": f"R{rep}",
                "yield": round(base_value, 2),
            })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()

# Upload
csv_b64 = create_test_csv()
resp = requests.post(UPLOAD_ENDPOINT, json={
    "base64_content": csv_b64,
    "file_type": "csv",
    "genotype_column": "genotype",
    "rep_column": "rep",
    "design_type": "rcbd",
    "mode": "single",
})
dataset_token = resp.json()["dataset_token"]
print(f"[OK] Uploaded dataset: {dataset_token[:20]}...")

# Analyze
resp = requests.post(ANOVA_ENDPOINT, json={
    "dataset_token": dataset_token,
    "trait_columns": ["yield"],
})
result = resp.json()

# Check the full structure
print("\n=== FULL ANOVA RESPONSE STRUCTURE ===")
print(json.dumps(result, indent=2, default=str)[:5000])

# Focus on yield result
yield_result = result.get("trait_results", {}).get("yield", {})
print("\n=== DIAGNOSTIC FIELDS IN RESPONSE ===")
for field in ["assumption_tests", "diagnostic_observations", "diagnostic_plots",
              "standardized_residuals", "cooks_distance", "outlier_summary"]:
    val = yield_result.get(field)
    if val is None:
        print(f"[MISSING] {field}: None")
    elif isinstance(val, dict) and len(val) == 0:
        print(f"[EMPTY] {field}: {{}}")
    elif isinstance(val, list) and len(val) == 0:
        print(f"[EMPTY] {field}: []")
    elif isinstance(val, dict):
        print(f"[PRESENT] {field}: dict with keys {list(val.keys())}")
    elif isinstance(val, list):
        print(f"[PRESENT] {field}: list with {len(val)} items")
    else:
        print(f"[PRESENT] {field}: {type(val).__name__}")

# Check assumption_tests structure
assumption_tests = yield_result.get("assumption_tests", {})
if assumption_tests and isinstance(assumption_tests, dict):
    print(f"\n=== ASSUMPTION_TESTS KEYS ===")
    print(f"Keys: {list(assumption_tests.keys())}")
    print(f"Expected: ['overall', 'normality', 'homogeneity', 'outlier_detection', 'reviewer_mode']")
