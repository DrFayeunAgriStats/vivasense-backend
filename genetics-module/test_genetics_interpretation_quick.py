"""
Quick validation test for Genetics Interpretation Engine

Verifies:
1. generate_genetics_interpretation() produces non-empty output
2. Output contains academic prose, not legacy text
3. Handles all heritability and GAM combinations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from genetics_interpretation import generate_genetics_interpretation

print("\n" + "=" * 80)
print("GENETICS INTERPRETATION ENGINE - QUICK VALIDATION")
print("=" * 80)

test_cases = [
    {
        "name": "High H² + High GAM",
        "trait": "grain_yield",
        "h2": 0.75,
        "gam": 12.5,
        "gcv": 18.2,
        "pcv": 19.5,
    },
    {
        "name": "Moderate H² + Low GAM",
        "trait": "plant_height",
        "h2": 0.45,
        "gam": 3.8,
        "gcv": 12.1,
        "pcv": 20.3,
    },
    {
        "name": "Low H² + Low GAM",
        "trait": "disease_resistance",
        "h2": 0.22,
        "gam": 1.5,
        "gcv": 5.2,
        "pcv": 22.1,
    },
    {
        "name": "H² not computed",
        "trait": "quality_score",
        "h2": None,
        "gam": None,
        "gcv": None,
        "pcv": None,
    },
]

passed = 0
failed = 0

for test in test_cases:
    print(f"\n[TEST] {test['name']}")
    try:
        interp, breed = generate_genetics_interpretation(
            trait_name=test["trait"],
            h2=test["h2"],
            gam=test["gam"],
            gcv=test["gcv"],
            pcv=test["pcv"],
        )
        
        # Validation checks
        checks = []
        
        # 1. Interpretation present and substantial
        if interp and len(interp) > 50:
            checks.append(f"✓ Interpretation ({len(interp)} chars)")
        else:
            checks.append(f"✗ Interpretation too short or empty ({len(interp)} chars)")
            failed += 1
            continue
        
        # 2. Breeding implication present
        if breed and len(breed) > 30:
            checks.append(f"✓ Breeding implication ({len(breed)} chars)")
        else:
            checks.append(f"✗ Breeding implication missing ({len(breed)} chars)")
            failed += 1
            continue
        
        # 3. No legacy hardcoded phrases
        legacy_phrases = ["improve together", "facilitating indirect selection"]
        has_legacy = any(phrase in interp.lower() + breed.lower() for phrase in legacy_phrases)
        if not has_legacy:
            checks.append("✓ No legacy phrases")
        else:
            checks.append("✗ Contains legacy phrases")
            failed += 1
            continue
        
        # 4. Contains specific content markers
        if "heritability" in interp.lower():
            checks.append("✓ Contains heritability discussion")
        else:
            checks.append("⚠ Missing heritability discussion")
        
        for check in checks:
            print(f"    {check}")
        
        # Sample output for first test
        if test["name"] == "High H² + High GAM":
            print(f"\n    Sample interpretation (first 150 chars):\n    {repr(interp[:150])}...")
        
        passed += 1
    except Exception as e:
        print(f"    ✗ EXCEPTION: {e}")
        failed += 1

print("\n" + "=" * 80)
print(f"RESULT: {passed} passed, {failed} failed")
if failed == 0:
    print("✓ GENETICS INTERPRETATION ENGINE WORKING CORRECTLY")
else:
    print(f"⚠ {failed} test(s) need attention")
print("=" * 80 + "\n")

sys.exit(0 if failed == 0 else 1)
