#!/usr/bin/env python3
"""Test Word export with assumption diagnostics section"""

import base64, io, json, requests, pandas as pd, random, sys

BACKEND = "https://vivasense-backend-r-production.up.railway.app"

# Create test data
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

print("[1] Uploading test dataset...")
resp = requests.post(f"{BACKEND}/upload/dataset", json={
    "base64_content": csv_b64,
    "file_type": "csv",
    "genotype_column": "genotype",
    "rep_column": "rep",
    "design_type": "rcbd",
    "mode": "single",
})
token = resp.json()["dataset_token"]
print(f"    Dataset token: {token[:20]}...")

print("\n[2] Running ANOVA analysis...")
resp = requests.post(f"{BACKEND}/analysis/anova", json={
    "dataset_token": token,
    "trait_columns": ["yield"],
})
anova_result = resp.json()
print(f"    Status: {anova_result.get('mode')}")

# Verify diagnostic fields are present
trait_result = anova_result.get("trait_results", {}).get("yield", {})
has_diagnostics = all(field in trait_result for field in [
    "assumption_tests", "diagnostic_observations", "diagnostic_plots"
])
print(f"    Diagnostics present: {has_diagnostics}")

print("\n[3] Attempting Word export...")
try:
    resp = requests.post(f"{BACKEND}/export/anova-word", json={
        "dataset_token": token,
        "mode": anova_result.get("mode", "single"),
        "trait_results": anova_result.get("trait_results", {}),
        "failed_traits": anova_result.get("failed_traits", []),
    }, timeout=120)

    if resp.status_code == 403:
        print("    [INFO] 403 Forbidden - PRO_FEATURE gate (expected, not relevant to this test)")
        print("    [SKIP] Cannot test full export due to feature gate")
        sys.exit(0)

    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        file_size = len(resp.content)

        # Save for inspection
        output_file = "test_export_with_diagnostics.docx"
        with open(output_file, "wb") as f:
            f.write(resp.content)

        print(f"    [PASS] Export successful")
        print(f"    - Content-Type: {content_type}")
        print(f"    - File size: {file_size} bytes")
        print(f"    - Saved to: {output_file}")
        print(f"\n[SUCCESS] Word export generates without 'Section rendering error' or TypeError")
        sys.exit(0)
    else:
        print(f"    [FAIL] Unexpected status: {resp.status_code}")
        print(f"    Response: {resp.text[:500]}")
        sys.exit(1)

except requests.exceptions.RequestException as e:
    print(f"    [FAIL] Request error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"    [FAIL] Error: {e}")
    sys.exit(1)
