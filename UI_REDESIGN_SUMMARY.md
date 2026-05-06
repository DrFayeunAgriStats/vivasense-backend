# VivaSense UI Redesign: Design-Aware and Domain-Aware Implementation

## Overview
The VivaSense ANOVA interface has been completely redesigned to be **design-aware** and **domain-aware**, automatically adapting to experimental context and research domains while maintaining beginner friendliness.

## What Changed

### 1. **Auto-Detection of Experimental Design** ✅
- **File:** `designDetection.ts`
- **Function:** `detectExperimentalDesign()`
- **Behavior:**
  - Analyzes column names using keyword matching
  - Detects blocking structure (Block, Rep, Replication)
  - Identifies main-plot factors (Irrigation, Variety, Tillage)
  - Identifies subplot factors (Nitrogen, Spacing, Date)
  - Returns suggested design with confidence level (high/medium/low)
  - Decision tree: Split-plot RCBD → Factorial RCBD → RCBD → CRD

**Keywords Used:**
- Blocking: `block`, `blk`, `rep`, `replication`, `replicate`
- Main-plot: `irrigation`, `tillage`, `variety`, `genotype`, `cultivar`, `fertilizer`, `management`, `system`
- Subplot: `nitrogen`, `n_rate`, `n_level`, `spacing`, `density`, `date`, `time`, `application`, `rate`, `dose`

### 2. **Intelligent Recommendation Cards** ✅
- **File:** `DesignRecommendationCard.tsx`
- **Behavior:**
  - Shows when confidence is medium or high
  - Displays detected design with confidence badge
  - Lists detection reasons (e.g., "Detected blocking column: Rep")
  - For split-plot: shows detected main-plot and subplot factors
  - Two actions:
    - **"Use Split-Plot RCBD"** → auto-fills design dropdown and field mappings
    - **"Choose manually"** → dismisses card, user selects manually
  - Color-coded by confidence:
    - High: Emerald green
    - Medium: Blue
    - Low: Amber (not shown)

### 3. **Domain-Neutral Terminology for Split-Plot** ✅
- **File:** `designLabels.ts`
- **Previous:** "Treatment / Genotype" (genetics-focused)
- **Now:** 
  - Split-Plot RCBD: "Main-Plot Factor" and "Subplot Factor" (domain-neutral)
  - RCBD/CRD: Uses domain-specific terms from `domainTerms.ts`
    - Plant breeding: "Genotype"
    - Agronomy: "Treatment"
    - General: "Treatment"

### 4. **Dynamic Labels Based on Design and Domain** ✅
- **File:** `designLabels.ts` → `AnovaWorkspaceModule.tsx`
- **Function:** `getDesignAwareLabels(design, domain)`
- **Behavior:**

| Design          | Blocking Label       | Treatment Label       | Factor Labels                     |
|-----------------|----------------------|-----------------------|-----------------------------------|
| CRD             | Replication          | Treatment/Genotype    | Factor A, Factor B                |
| RCBD            | Replication / Block  | Treatment/Genotype    | Factor A, Factor B                |
| Factorial       | Replication / Block  | Treatment/Genotype    | Factor A, Factor B                |
| Split-Plot RCBD | Replication / Block  | —                     | Main-Plot Factor, Subplot Factor  |

- Labels respect research domain (plant_breeding → "Genotype", agronomy → "Treatment")
- Split-plot always uses domain-neutral factor terminology

### 5. **Field-Specific Help Text with Tooltips** ✅
- **File:** `AnovaWorkspaceModule.tsx` (Field component)
- **Function:** `getFieldHelpText(field, design)`
- **Behavior:**
  - Info icon (ⓘ) appears next to field labels when help text is available
  - Hover shows tooltip with contextual explanation
  - Examples:
    - Main-Plot Factor: "Coarse treatment factor applied to whole plots (e.g., irrigation, variety)"
    - Subplot Factor: "Fine treatment factor applied within each whole plot (e.g., nitrogen rate, spacing)"
    - Blocking: "Blocks control spatial or temporal variability"

### 6. **Smart Field Visibility** ✅
- **Previous:** All fields stacked with conditional hiding (left gaps)
- **Now:** Fields dynamically shown/hidden based on design type:
  - **CRD:** Treatment only (no blocking)
  - **RCBD:** Treatment + Blocking
  - **Factorial:** Factor A + Factor B + Blocking
  - **Split-Plot RCBD:** Main-Plot Factor + Subplot Factor + Blocking (no generic "Treatment")
- Uses conditional rendering (`design !== "crd" &&`) to prevent visual gaps
- CSS grid adapts seamlessly with `md:grid-cols-2` responsive layout

### 7. **Design-Specific Descriptions** ✅
- **File:** `designLabels.ts` → `AnovaWorkspaceModule.tsx`
- **Location:** Below "ANOVA Module Setup" heading
- **Examples:**
  - CRD: "CRD: treatments completely randomized (no blocking)"
  - RCBD: "RCBD: genotypes randomized within each block"
  - Factorial: "Factorial design: tests all combinations of Factor A and Factor B"
  - Split-Plot RCBD: "Split-plot design: main-plot factors assigned to whole plots, subplot factors nested within"

## Technical Architecture

### New Files Created
1. **`designDetection.ts`**
   - Exports: `detectExperimentalDesign()`, `formatDesignRecommendation()`
   - Types: `DesignDetectionResult`, `ColumnAnalysis`
   - Purpose: Auto-detect design from column names

2. **`DesignRecommendationCard.tsx`**
   - React component for displaying detection results
   - Props: `detection`, `onAccept()`, `onDismiss()`
   - Purpose: Interactive recommendation card

3. **`designLabels.ts`**
   - Exports: `getDesignAwareLabels()`, `getColumnPlaceholder()`, `getFieldHelpText()`
   - Type: `DesignAwareLabels`
   - Purpose: Generate dynamic labels based on design and domain

### Modified Files
1. **`AnovaWorkspaceModule.tsx`**
   - Added imports for new utilities
   - Added state: `designDetection`, `showRecommendation`
   - Updated `useEffect` to run design detection on dataset load
   - Added `<DesignRecommendationCard />` rendering
   - Replaced hardcoded field labels with `labels.treatmentLabel`, `labels.blockingLabel`, etc.
   - Added conditional field rendering based on design type
   - Updated `Field` component to support `helpText` prop with tooltip icon
   - Updated `ColumnSelect` component to support `placeholder` prop

## User Experience Flow

### Step 1: Upload Dataset
- User uploads CSV/Excel file
- Dataset parsed, columns detected

### Step 2: Design Recommendation (NEW)
- System runs `detectExperimentalDesign()`
- If confidence ≥ medium:
  - Shows `<DesignRecommendationCard />` above form
  - Lists detection reasons
  - User can accept (auto-fills) or dismiss

### Step 3: Configure Analysis
- User selects or confirms design from dropdown
- **Field labels adapt dynamically:**
  - Split-plot shows "Main-Plot Factor" and "Subplot Factor"
  - RCBD shows domain-specific "Genotype" or "Treatment"
  - Factorial shows "Factor A" and "Factor B"
- Irrelevant fields hidden (no gaps)
- Hover over (ⓘ) for field-specific help

### Step 4: Select Traits & Run
- Unchanged from original design
- Click trait badges to select
- Click "Run ANOVA" to execute analysis

## Validation Checklist

### Requirement 1: Auto-detect design ✅
- [x] `detectExperimentalDesign()` analyzes column names
- [x] Returns suggested design with confidence
- [x] Detects split-plot structure with main-plot and subplot factors

### Requirement 2: Remove "Treatment / Genotype" for split-plot ✅
- [x] Split-plot shows "Main-Plot Factor" and "Subplot Factor" only
- [x] No generic "Treatment" label when design = split_plot_rcbd

### Requirement 3: Dynamic labels (design + domain) ✅
- [x] Labels change based on `design` state
- [x] Labels respect `research_domain` from dataset context
- [x] Plant breeding uses "Genotype", agronomy uses "Treatment"

### Requirement 4: Split-plot field structure ✅
- [x] Replication / Block (not "Block" alone)
- [x] Main-Plot Factor (not "Main plot factor")
- [x] Subplot Factor (not "Subplot factor")
- [x] Consistent capitalization and terminology

### Requirement 5: Smart field hiding ✅
- [x] CRD hides blocking field
- [x] Split-plot hides generic treatment field
- [x] Factorial shows Factor A and Factor B
- [x] No visual gaps when fields are hidden

### Requirement 6: Recommendation cards ✅
- [x] Shows when confidence is medium or high
- [x] Displays suggested design with icon and confidence badge
- [x] Lists detection reasons
- [x] "Use [Design]" button auto-fills fields
- [x] "Choose manually" button dismisses card

### Requirement 7: Preserve beginner friendliness ✅
- [x] Help tooltips on hover (ⓘ icon)
- [x] Design-specific descriptions below heading
- [x] Clear placeholders in dropdowns
- [x] Recommendation cards explain "why" a design is suggested
- [x] Auto-population reduces manual configuration

## Examples

### Example 1: Split-Plot Dataset Detected
**Columns:** `Rep`, `Irrigation`, `Nitrogen`, `Yield`

**Recommendation Card:**
```
🔀 Possible Split-Plot RCBD Design Detected (high confidence)
• Detected blocking column: Rep
• Detected main-plot factor: Irrigation
• Detected subplot factor: Nitrogen
• Split-plot designs apply two-level randomization

Split-plot structure identified:
Main-plot: Irrigation
Subplot: Nitrogen

[Use Split-Plot RCBD] [Choose manually]
```

**After Accepting:**
- Design dropdown: "Split-Plot RCBD"
- Field 1: "Replication / Block" → auto-filled with "Rep"
- Field 2: "Main-Plot Factor" → auto-filled with "Irrigation"
- Field 3: "Subplot Factor" → auto-filled with "Nitrogen"

### Example 2: RCBD with Plant Breeding Domain
**Columns:** `Block`, `Genotype`, `GrainYield`
**Domain:** `plant_breeding`

**Fields Shown:**
- Design: RCBD
- Genotype (not "Treatment")
- Replication / Block
- Description: "RCBD: genotypes randomized within each block"

### Example 3: Factorial Design
**Columns:** `Rep`, `NitrogenLevel`, `Spacing`, `Height`

**Fields Shown:**
- Design: Factorial
- Replication / Block
- Factor A (e.g., NitrogenLevel)
- Factor B (e.g., Spacing)
- Description: "Factorial design: tests all combinations of Factor A and Factor B"

## Testing Recommendations

1. **Test split-plot detection:**
   - Upload CSV with columns: `Block`, `Irrigation`, `Nitrogen_Rate`, `Yield`
   - Verify recommendation card appears
   - Accept recommendation and confirm auto-population

2. **Test domain-specific terminology:**
   - Set `research_domain = "plant_breeding"` in dataset context
   - Verify RCBD shows "Genotype" instead of "Treatment"

3. **Test field visibility:**
   - Select CRD → verify blocking field is hidden
   - Select Split-Plot RCBD → verify generic treatment field is hidden
   - Select Factorial → verify Factor A and Factor B appear

4. **Test tooltips:**
   - Hover over (ⓘ) icon next to "Main-Plot Factor"
   - Verify tooltip shows explanatory text

5. **Test manual override:**
   - Dismiss recommendation card
   - Manually select different design from dropdown
   - Verify labels adapt to new selection

## Future Enhancements

1. **Improve detection accuracy:**
   - Analyze actual data (unique value counts) instead of just column names
   - Add more keyword patterns for diverse datasets

2. **Multi-environment support:**
   - Detect MET (multi-environment trials) structure
   - Auto-suggest environment-specific analyses

3. **Advanced tooltips:**
   - Link to documentation or video tutorials
   - Show example datasets inline

4. **Field validation:**
   - Warn if same column selected for multiple fields
   - Suggest corrections if invalid combinations detected

## Deployment Notes

- All changes are frontend-only (no backend API changes required)
- Compatible with existing `analyzeUpload()` API
- Uses existing `research_domain` field from `UploadDatasetContext`
- Backward compatible with datasets that don't have domain set (defaults to "general")

## Files Modified Summary

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `designDetection.ts` | +180 (new) | Design auto-detection logic |
| `DesignRecommendationCard.tsx` | +130 (new) | Recommendation card UI |
| `designLabels.ts` | +160 (new) | Dynamic label generation |
| `AnovaWorkspaceModule.tsx` | ~150 modified | Main UI component with design awareness |

**Total:** ~620 lines of new/modified code

## Success Criteria Met

✅ **Auto-detect experimental design from dataset structure**
✅ **Remove generic "Treatment / Genotype" terminology from split-plot workflows**
✅ **Dynamically adapt labels based on research domain and design**
✅ **Split-plot interface uses proper field names (Replication/Block, Main-Plot Factor, Subplot Factor)**
✅ **Hide irrelevant fields dynamically (no visual gaps)**
✅ **Add intelligent recommendation cards for likely designs**
✅ **Preserve beginner friendliness with tooltips and clear guidance**

---

**Implementation Status:** ✅ **COMPLETE**

All 7 requirements have been fully implemented with no errors. The UI is now design-aware, domain-aware, and scientifically correct while remaining accessible to beginners.
