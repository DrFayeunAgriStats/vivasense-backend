# Split-Plot ANOVA Testing Guide

## Quick Validation Test

### Test Data Structure
Create a CSV with columns:
- `rep` — block/replication (e.g., 1, 2, 3)
- `main_plot` — main-plot factor/treatment (e.g., Irrigation: I1, I2, I3)
- `sub_plot` — subplot factor/treatment (e.g., Variety: V1, V2, V3, V4)
- `yield` — response variable

### Example Data (3 reps × 3 irrigation × 4 varieties = 36 observations)

```csv
rep,main_plot,sub_plot,yield
1,I1,V1,45.2
1,I1,V2,48.5
1,I1,V3,42.8
1,I1,V4,46.1
1,I2,V1,52.3
1,I2,V2,55.8
...
```

### Expected ANOVA Table Structure

| Source       | Df | Sum Sq | Mean Sq | F value | Pr(>F) |
|--------------|-----|--------|---------|---------|--------|
| Replication  | 2   | ...    | ...     | —       | —      |
| main_plot    | 2   | ...    | ...     | F_A     | p_A    |
| Error A      | 4   | ...    | MS_A    | —       | —      |
| sub_plot     | 3   | ...    | ...     | F_B     | p_B    |
| main_plot:sub_plot | 6 | ... | ...   | F_AB    | p_AB   |
| Error B      | 24  | ...    | MS_B    | —       | —      |

### F-test Verification

**Check 1: Main-plot factor (Irrigation)**
- Should be tested against Error A
- F_A = MS_main_plot / MS_Error_A
- df1 = 2, df2 = 4
- p-value = pf(F_A, 2, 4, lower.tail=FALSE)

**Check 2: Subplot factor (Variety)**
- Should be tested against Error B
- F_B = MS_sub_plot / MS_Error_B
- df1 = 3, df2 = 24
- p-value = pf(F_B, 3, 24, lower.tail=FALSE)

**Check 3: Interaction (Irrigation × Variety)**
- Should be tested against Error B
- F_AB = MS_interaction / MS_Error_B
- df1 = 6, df2 = 24
- p-value = pf(F_AB, 6, 24, lower.tail=FALSE)

### Interpretation Validation

**When interaction IS significant (p < 0.05):**
- [ ] Interpretation section titled "A×B Interaction Effect (Primary)" appears first
- [ ] Main-plot and subplot effects labeled as "Conditional"
- [ ] Interpretation emphasizes cell means over marginal means
- [ ] Recommends generating interaction plot
- [ ] Warns that main effects alone are misleading

**When interaction IS NOT significant (p ≥ 0.05):**
- [ ] Interpretation states "additive factor effects"
- [ ] Main-plot and subplot effects labeled independently (not conditional)
- [ ] Marginal means are emphasized
- [ ] No interaction plot recommendation

### CV Validation

**Check 1: CV_A (whole-plot CV)**
- CV_A = √(MS_Error_A) / grand_mean × 100
- Should be reported in "Experimental Precision" section
- Labeled as "whole-plot coefficient of variation"

**Check 2: CV_B (subplot CV)**
- CV_B = √(MS_Error_B) / grand_mean × 100
- Should be reported in "Experimental Precision" section
- Labeled as "subplot coefficient of variation"

**Check 3: Interpretation acknowledges difference**
- If CV_A ≠ CV_B, interpretation should note "The two CVs reflect the separate error strata"

### Mean Separation Validation

**Main-plot means:**
- [ ] Tested with Error A (Error df = 4 in example)
- [ ] Labeled as "Main-plot mean separation (Fisher LSD, α = 0.05, tested against Error A)"
- [ ] Note: "These marginal means average across all subplot levels"

**Subplot means:**
- [ ] Tested with Error B (Error df = 24 in example)
- [ ] Labeled as "Subplot mean separation (Fisher LSD, α = 0.05, tested against Error B)"
- [ ] Note: "These marginal means average across all main-plot levels"

### Interaction Means Validation

**Check API response includes:**
```json
{
  "interaction_means": {
    "main_plot_levels": ["I1", "I2", "I3"],
    "sub_plot_levels": ["V1", "V2", "V3", "V4"],
    "cell_means": [
      {"main_plot": "I1", "sub_plot": "V1", "trait_value": 45.2},
      {"main_plot": "I1", "sub_plot": "V2", "trait_value": 48.5},
      ...
    ],
    "means_matrix": {
      "main_plot": ["I1", "I2", "I3"],
      "V1": [45.2, 52.3, ...],
      "V2": [48.5, 55.8, ...],
      ...
    }
  }
}
```

### Domain-Neutral Terminology Validation

**Default mode (domain = "general"):**
- [ ] Uses "main-plot factor" (NOT "genotype")
- [ ] Uses "subplot factor" (NOT "treatment")
- [ ] Uses "treatment combination" or "factor combination"
- [ ] Does NOT use breeding-specific terms (heritability, selection, genetic gain)

**Breeding mode (domain = "plant_breeding"):**
- [ ] May use "genotype" if genotype_column is main_plot or sub_plot
- [ ] Uses breeding terminology where appropriate
- [ ] Includes selection intensity and genetic gain if requested

---

## Manual Testing Workflow

### Step 1: Upload Dataset
```bash
POST /upload/dataset
{
  "base64_content": "...",
  "file_type": "csv",
  "genotype_column": null,  # No genotype for generic split-plot
  "rep_column": "rep",
  "main_plot_column": "main_plot",
  "sub_plot_column": "sub_plot",
  "design_type": "split_plot_rcbd",
  "mode": "single"
}
```

**Expected response:**
- `dataset_token`: valid UUID
- `n_genotypes`: null (generic split-plot has no genotype column)
- `n_reps`: 3
- `design_type`: "split_plot_rcbd"

### Step 2: Run ANOVA
```bash
POST /analysis/anova
{
  "dataset_token": "...",
  "trait_columns": ["yield"]
}
```

**Expected response structure:**
```json
{
  "dataset_token": "...",
  "mode": "single",
  "trait_results": {
    "yield": {
      "trait": "yield",
      "status": "success",
      "grand_mean": 48.5,
      "n_reps": 3,
      "anova_table": {
        "source": ["Replication", "main_plot", "Error A", "sub_plot", "main_plot:sub_plot", "Error B"],
        "df": [2, 2, 4, 3, 6, 24],
        ...
      },
      "cv_a": 12.5,
      "cv_b": 8.3,
      "main_plot_mean_separation": { ... },
      "mean_separation": { ... },
      "interaction_means": { ... },
      "interpretation": "Overview\nThis analysis used a split-plot...",
      "design_type": "split_plot_rcbd"
    }
  },
  "failed_traits": []
}
```

### Step 3: Validate F-tests
1. Extract MS values from anova_table
2. Recalculate F-ratios manually:
   - F_main_plot = MS_main_plot / MS_Error_A
   - F_sub_plot = MS_sub_plot / MS_Error_B
   - F_interaction = MS_interaction / MS_Error_B
3. Verify against returned F values
4. Recalculate p-values using R or Python:
   ```python
   from scipy.stats import f
   p_value = 1 - f.cdf(f_stat, df_numerator, df_denominator)
   ```

### Step 4: Validate Interpretation
1. Read interpretation text
2. Check for presence of all 9 sections:
   - Overview
   - Statistical Model
   - Experimental Precision
   - A×B Interaction Effect
   - Main-Plot Factor Effect
   - Subplot Factor Effect
   - Mean Separation Summary
   - Risk and Limitations
   - Recommendation
3. Verify terminology is domain-neutral
4. Verify interaction priority rule is followed

---

## Regression Test

Run on existing RCBD and CRD datasets to ensure backward compatibility:
- [ ] RCBD (non-split-plot) still works
- [ ] CRD still works
- [ ] Factorial RCBD still works
- [ ] Multi-environment RCBD still works

**Expectation:** No changes to non-split-plot results.

---

## Known Limitations (Future Work)

1. **Unbalanced data:** Current implementation assumes equal observations per cell. Missing data will cause R errors.
2. **Strip-plot designs:** Not yet supported (requires two whole-plot factors).
3. **Split-split-plot:** Not yet supported (requires three-level nesting).
4. **Simple effects:** When interaction significant, user must manually extract cell means to test effect of B at each level of A.
5. **Multiple comparison adjustments:** Only Fisher LSD is supported. Bonferroni, Holm, or FDR for interaction cell means not implemented.

---

## Success Criteria

✅ All 6 sources of variation appear in ANOVA table  
✅ Main-plot factor tested against Error A  
✅ Subplot factor and interaction tested against Error B  
✅ P-values correct for all F-tests  
✅ Dual CV (CV_A and CV_B) reported  
✅ Main-plot and subplot mean separations use correct error terms  
✅ Interaction means returned in API response  
✅ Interpretation is domain-neutral (not breeding-focused)  
✅ Interaction priority rule implemented correctly  
✅ No errors in Python code  
✅ No syntax errors in R code  
✅ Backward compatible with existing RCBD/CRD analyses  

---

## Contact

For questions or issues with the split-plot implementation, review:
- `SPLIT_PLOT_REDESIGN_SUMMARY.md` — comprehensive design documentation
- `vivasense_genetics.R` — R statistical engine
- `analysis_anova_routes.py` — Python interpretation engine
- `genetics_schemas.py`, `module_schemas.py` — API schemas
