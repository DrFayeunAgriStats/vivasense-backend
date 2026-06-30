#!/usr/bin/env python3
"""Test Correlation Word export with CorrelationStats Pydantic fix"""

import base64, io, json, requests, pandas as pd, random, sys

BACKEND = "https://vivasense-backend-r-production.up.railway.app"

# Create test data with 3 traits for correlation
random.seed(42)
rows = []
for rep in range(1, 4):
    for geno in range(1, 5):
        rows.append({
            "genotype": f"G{geno:02d}",
            "rep": f"R{rep}",
            "trait_a": round(50.0 + (geno - 1) * 5.0 + random.gauss(0, 3), 2),
            "trait_b": round(75.0 + (geno - 1) * 3.0 + random.gauss(0, 2.5), 2),
            "trait_c": round(100.0 + (geno - 1) * 8.0 + random.gauss(0, 4), 2),
        })

df = pd.DataFrame(rows)
buf = io.StringIO()
df.to_csv(buf, index=False)
csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

print("[1] Uploading test dataset with 3 traits...")
resp = requests.post(f"{BACKEND}/upload/dataset", json={
    "base64_content": csv_b64,
    "file_type": "csv",
    "genotype_column": "genotype",
    "rep_column": "rep",
    "design_type": "rcbd",
    "mode": "single",
})
if resp.status_code != 200:
    print(f"    [FAIL] Upload failed: {resp.status_code}")
    print(f"    Response: {resp.text[:500]}")
    sys.exit(1)

token = resp.json()["dataset_token"]
print(f"    Dataset token: {token[:20]}...")

print("\n[2] Running Correlation analysis...")
resp = requests.post(f"{BACKEND}/analysis/correlation", json={
    "dataset_token": token,
    "trait_columns": ["trait_a", "trait_b", "trait_c"],
    "method": "pearson",
    "user_objective": "Field understanding",
})

if resp.status_code != 200:
    print(f"    [FAIL] Correlation analysis failed: {resp.status_code}")
    print(f"    Response: {resp.text[:500]}")
    sys.exit(1)

corr_result = resp.json()
print(f"    Correlation analysis complete")
print(f"    - Phenotypic n_observations: {corr_result.get('phenotypic', {}).get('n_observations')}")

print("\n[3] Attempting Correlation Word export...")
print("    (This should trigger the CorrelationStats Pydantic validation)")

try:
    # Export with the full correlation result
    export_payload = {
        "dataset_token": corr_result.get("dataset_token"),
        "trait_names": corr_result.get("trait_names"),
        "method": corr_result.get("method"),
        "phenotypic": corr_result.get("phenotypic"),
        "between_genotype": corr_result.get("between_genotype"),
        "genotypic": corr_result.get("genotypic"),
        "interpretation": corr_result.get("interpretation", ""),
        "warnings": corr_result.get("warnings", []),
        "statistical_note": corr_result.get("statistical_note", ""),
    }

    resp = requests.post(
        f"{BACKEND}/export/correlation-word",
        json=export_payload,
        timeout=120
    )

    if resp.status_code == 403:
        print("    [INFO] 403 Forbidden - PRO_FEATURE gate (expected, not relevant to this test)")
        print("    [SKIP] Cannot test full export due to feature gate")
        sys.exit(0)

    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        file_size = len(resp.content)

        # Save for inspection
        output_file = "test_correlation_export.docx"
        with open(output_file, "wb") as f:
            f.write(resp.content)

        print(f"    [PASS] Correlation export successful")
        print(f"    - Content-Type: {content_type}")
        print(f"    - File size: {file_size} bytes")
        print(f"    - Saved to: {output_file}")
        print(f"\n[SUCCESS] CorrelationStats Pydantic ValidationError is FIXED")
        sys.exit(0)
    elif resp.status_code == 500:
        error_msg = resp.text
        print(f"    [FAIL] 500 Internal Server Error")
        print(f"    Error detail: {error_msg[:500]}")
        if "Input should be a valid dictionary" in error_msg and "CorrelationStats" in error_msg:
            print(f"    >> This is the CorrelationStats Pydantic ValidationError — FIX NOT APPLIED")
        sys.exit(1)
    else:
        print(f"    [FAIL] Unexpected status: {resp.status_code}")
        print(f"    Response: {resp.text[:500]}")
        sys.exit(1)

except requests.exceptions.RequestException as e:
    print(f"    [FAIL] Request error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"    [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
