# Integrating Multi-Trait Upload into VivaSenseGeneticsPage

All Python-side logic has been integration-tested across four scenarios.
This document contains the exact steps to wire everything into the live
Lovable project at fieldtoinsightacademy.com.ng/vivasense/genetics.

---

## Files to copy into your Lovable project

```
src/services/geneticsUploadApi.ts                ← API client + all TypeScript types
src/components/vivasense-genetics/
  FileUpload.tsx                                  ← drag-drop file input
  ColumnMappingConfirm.tsx                        ← column dropdowns + trait checkboxes
  ResultsDisplay.tsx                              ← summary table + expandable details
  MultiTraitUpload.tsx                            ← state-machine orchestrator
  DataSourceTabs.tsx                              ← "Manual Input | Upload File" tabs
```

Do not copy `INTEGRATION.md` or `test_upload_integration.py`.

---

## Environment variable

In Lovable → Settings → Environment Variables:

```
VITE_GENETICS_ENGINE_BASE = https://vivasense-genetics.onrender.com
```

`geneticsUploadApi.ts` falls back to this URL when the variable is unset.

---

## The one change to VivaSenseGeneticsPage.tsx

Find where `<DynamicInputForm ... />` (or equivalent manual-input form) is
rendered. Wrap it with `DataSourceTabs`:

```tsx
// Before
import { DynamicInputForm } from "./DynamicInputForm";

export function VivaSenseGeneticsPage() {
  return (
    <div>
      {/* ... existing layout ... */}
      <DynamicInputForm
        onSubmit={handleSubmit}
        loading={loading}
        {/* ... other props ... */}
      />
    </div>
  );
}

// After
import { DynamicInputForm }  from "./DynamicInputForm";
import { DataSourceTabs }    from "./DataSourceTabs";
import { MultiTraitUpload }  from "./MultiTraitUpload";

export function VivaSenseGeneticsPage() {
  return (
    <div>
      {/* ... existing layout ... */}
      <DataSourceTabs
        manualContent={
          <DynamicInputForm
            onSubmit={handleSubmit}
            loading={loading}
            {/* ... same props as before ... */}
          />
        }
        uploadContent={<MultiTraitUpload />}
      />
    </div>
  );
}
```

`DataSourceTabs` uses `display: block / hidden` — the manual form is mounted
at all times, its state is not lost when switching tabs.

---

## What was integration-tested

Four scenarios were run end-to-end against the Python route logic with the
R engine mocked to a deterministic stub. All 4 passed (0 assertion failures).

| # | Scenario | Key assertion |
|---|----------|--------------|
| 1 | Single-env CSV, 1 trait, balanced | PlotID excluded; TraitResult.analysis_result present |
| 2 | Multi-env XLSX, 3 traits, balanced | n_environments=3; all 3 traits in trait_results |
| 3 | 1 trait fails (< 6 valid obs) | analysis_result=null, error message, Yield unaffected |
| 4 | ~20% missing values | data_warnings populated for Yield; PlantHeight clean |

**Not tested here (requires Render deployment with R + jsonlite + agricolae):**
- Actual variance component values from R
- Interpretation text from vivasense_interpretation_engine.R
- Timeout behaviour under large files

---

## Backend changes made (already in this branch)

| File | Change |
|------|--------|
| `app_genetics.py` | Added `CORSMiddleware` for `fieldtoinsightacademy.com.ng` (was missing — would have blocked all browser requests) |
| `app_genetics.py` | Registered `multitrait_upload_routes.router` |
| `multitrait_upload_routes.py` | New endpoints: `POST /genetics/upload-preview`, `POST /genetics/analyze-upload` |
| `multitrait_upload_schemas.py` | New Pydantic models for upload request/response |
| `genetics-module/requirements.txt` | Added `openpyxl==3.1.2` |

---

## Response shape reference

For a successful single-env analysis of one trait the wire JSON is:

```json
{
  "summary_table": [
    {
      "trait": "Yield_kg_ha",
      "grand_mean": 48.85,
      "h2": 0.72,
      "gcv": 14.3,
      "pcv": 18.9,
      "gam_percent": 24.0,
      "heritability_class": "high",
      "status": "success",
      "error": null
    }
  ],
  "trait_results": {
    "Yield_kg_ha": {
      "status": "success",
      "analysis_result": {
        "status": "SUCCESS",
        "mode": "single",
        "data_validation": { "is_valid": true, "warnings": [] },
        "variance_warnings": { "is_valid": true, "warnings": [] },
        "result": {
          "environment_mode": "single",
          "n_genotypes": 10,
          "n_reps": 3,
          "n_environments": null,
          "grand_mean": 48.85,
          "variance_components": {
            "sigma2_genotype": 14.655,
            "sigma2_error": 7.816,
            "sigma2_ge": null,
            "sigma2_phenotypic": 26.379,
            "heritability_basis": "entry-mean"
          },
          "heritability": {
            "h2_broad_sense": 0.72,
            "interpretation_basis": "entry-mean",
            "formula": "σ²G / (σ²G + σ²E/r)"
          },
          "genetic_parameters": {
            "GCV": 14.3, "PCV": 18.9,
            "GAM": 11.72, "GAM_percent": 24.0,
            "selection_intensity": 1.4
          }
        },
        "interpretation": "Broad-sense heritability was high (H² = 0.72)..."
      },
      "error": null,
      "data_warnings": []
    }
  },
  "dataset_summary": {
    "n_genotypes": 10, "n_reps": 3,
    "n_environments": null, "n_traits": 1, "mode": "single"
  },
  "failed_traits": []
}
```

---

## How frontend components use this shape

| Component | What it reads |
|-----------|--------------|
| `ResultsDisplay` header | `dataset_summary.n_genotypes`, `.n_environments`, `.n_reps` |
| `ResultsDisplay` banner | `failed_traits[]` |
| `SummaryRow` | `summary_table[i]` fields: `h2`, `gcv`, `pcv`, `gam_percent`, `heritability_class`, `status`, `error` |
| `TraitDetails` warnings | `trait_results[name].data_warnings[]` |
| `TraitDetails` variance grid | `trait_results[name].analysis_result.result.variance_components` |
| `TraitDetails` interpretation | `trait_results[name].analysis_result.interpretation` |
