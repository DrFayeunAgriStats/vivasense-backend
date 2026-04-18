/**
 * VivaSense Frontend — Unified Export Preview & Defensive Rendering System
 * ============================================================================
 *
 * This module provides a cohesive approach to:
 * 1. Defensive data normalization (no crashes on malformed server data)
 * 2. Consistent Word export preview UI (WordExportPreviewModal)
 * 3. Per-module preview builders (ANOVA, Genetic Parameters, Correlation)
 * 4. Safe rendering fallbacks (placeholders when data is missing)
 * 5. Comprehensive debug logging (dev-only console output)
 *
 * ============================================================================
 * ARCHITECTURE
 * ============================================================================
 *
 * Layers (bottom to top):
 *
 * 1. NORMALIZATION LAYER (src/utils/normalizeModuleData.ts)
 *    ├─ normalizeMatrix()          — Ensure n×n matrix structure
 *    ├─ hasValidMatrix()           — Validate matrix dimensions
 *    ├─ safeArray()                — Extract arrays safely
 *    ├─ safeNumber()               — Coerce numbers with fallback
 *    ├─ normalizeAnovaTable()      — ANOVA-specific normalization
 *    ├─ normalizeVarianceComponents()  — Genetic parameters normalization
 *    ├─ normalizeHeatmapData()     — Correlation matrix normalization
 *    └─ logDebug()                 — Dev-only logging
 *
 * 2. PREVIEW BUILDER LAYER (src/utils/previewBuilders.ts)
 *    ├─ buildAnovaPreview()              — ANOVA module preview sections
 *    ├─ buildGeneticParametersPreview()  — Genetic parameters sections
 *    ├─ buildCorrelationHeatmapPreview() — Correlation sections
 *    └─ formatStat()                     — Common formatting utility
 *
 * 3. MODAL COMPONENT LAYER (src/components/export/WordExportPreviewModal.tsx)
 *    ├─ WordExportPreviewModal           — Reusable preview modal UI
 *    └─ PreviewSection interface        — Data structure for sections
 *
 * 4. MODULE COMPONENTS LAYER (src/components/vivasense-genetics/)
 *    ├─ ResultsDisplay.tsx              — ANOVA & Genetic Parameters
 *    ├─ TraitRelationships.tsx          — Correlation analysis
 *    ├─ CorrelationHeatmap.tsx          — Heatmap rendering
 *    └─ ...other components
 *
 * ============================================================================
 * MODULE INTEGRATIONS
 * ============================================================================
 *
 * ANOVA & GENETIC PARAMETERS (ResultsDisplay.tsx)
 * ─────────────────────────────────────────────────
 * State:
 *   - results: UploadAnalysisResponse (from backend)
 *   - showPreview: boolean
 *   - downloading: boolean
 *
 * Flow:
 *   1. buildReportPreview(results) normalizes and extracts sections
 *   2. User clicks "Preview Report" → setShowPreview(true)
 *   3. WordExportPreviewModal renders with preview data
 *   4. User clicks "Download Report" → exportWordReport(results)
 *   5. Fallback: If no successful traits, Export button disabled
 *
 * Defensive features:
 *   ✓ safeArray() for summary_table, failed_traits
 *   ✓ safeNumber() for dataset dimensions
 *   ✓ Optional chaining (?.) for trait_results access
 *   ✓ Placeholders for missing ANOVA/mean separation
 *   ✓ Amber warning box for failed traits
 *   ✓ logDebug() tracking preview build
 *
 *
 * CORRELATION ANALYSIS (TraitRelationships.tsx)
 * ──────────────────────────────────────────────
 * State:
 *   - step: 'setup' | 'correlating' | 'results'
 *   - results: CorrelationResponse | null
 *   - showPreview: boolean
 *   - displayMode: 'phenotypic' | 'between_genotype' | 'genotypic'
 *
 * Flow:
 *   1. User selects traits, objective, method
 *   2. computeCorrelation() called → step='correlating'
 *   3. On response: step='results', results populated
 *   4. buildCorrelationPreview() builds sections with mode fallback
 *   5. User clicks "Preview Report" → WordExportPreviewModal opens
 *   6. Heatmap renders with flattenMatrix() defense
 *
 * Defensive features:
 *   ✓ _resolveStats() with graceful mode fallback
 *   ✓ flattenMatrix() validates r_matrix/p_matrix before access
 *   ✓ displayMode switches based on data availability
 *   ✓ Warnings if breeding objective but genotypic unavailable
 *   ✓ Comprehensive console logging on results render
 *   ✓ Fallback placeholder: "No heatmap data available"
 *
 *
 * CORRELATION HEATMAP (CorrelationHeatmap.tsx)
 * ────────────────────────────────────────────
 * Props:
 *   - data: CorrelationResponse
 *   - mode?: 'phenotypic' | 'between_genotype' | 'genotypic'
 *   - exportRef?: ref for PNG/SVG export
 *
 * Rendering logic:
 *   1. Accepts CorrelationResponse with dual-mode structure
 *   2. flattenMatrix(data, mode) extracts n² cells or [] if invalid
 *   3. SVG grid renders ONLY if cells.length === n*n
 *   4. Otherwise shows "No heatmap data available" placeholder
 *   5. Every cell access guarded: activeMatrix[row]?.[col] ?? null
 *
 * Defensive features:
 *   ✓ flattenMatrix() always returns [] on invalid input
 *   ✓ Cell access uses optional chaining (?.)
 *   ✓ SVG guard: renders only when data valid
 *   ✓ Console warnings on data issues
 *   ✓ Color scale handles null values gracefully
 *
 * ============================================================================
 * CRASH PREVENTION PATTERNS
 * ============================================================================
 *
 * Pattern 1: Array access
 *   ❌ BAD:  results.trait_names[i]
 *   ✅ GOOD: safeArray(results.trait_names)[i]
 *
 * Pattern 2: Number fields
 *   ❌ BAD:  dataset_summary.n_genotypes
 *   ✅ GOOD: safeNumber(dataset_summary?.n_genotypes)
 *
 * Pattern 3: Matrix access
 *   ❌ BAD:  r_matrix[i][j]
 *   ✅ GOOD: hasValidMatrix(r_matrix, n) && r_matrix[i]?.[j]
 *
 * Pattern 4: Nested objects
 *   ❌ BAD:  results.trait_results[trait].analysis_result.result.anova_table
 *   ✅ GOOD: trait_results?.[trait]?.analysis_result?.result?.anova_table
 *
 * Pattern 5: Rendering guards
 *   ❌ BAD:  {result.anova_table && <AnovaTable at={result.anova_table} />}
 *   ✅ GOOD: {result?.anova_table ? <AnovaTable at={result.anova_table} /> : <p>Not available</p>}
 *
 * ============================================================================
 * DEBUG LOGGING (Dev-Only)
 * ============================================================================
 *
 * Enabled when process.env.NODE_ENV !== "production"
 *
 * Key logs:
 *
 * ResultsDisplay.tsx:
 *   logDebug("ResultsDisplay:preview", {
 *     total_traits,
 *     success_traits,
 *     failed_traits,
 *     has_dataset_summary
 *   })
 *
 * TraitRelationships.tsx:
 *   logDebug("TraitRelationships:results", {
 *     activeMode,
 *     n_traits,
 *     matrix_dimensions,
 *     genotypic_available,
 *     inference_approximate
 *   })
 *
 * CorrelationHeatmap.tsx:
 *   console.log("[CorrelationHeatmap] flattenMatrix", {
 *     mode,
 *     n,
 *     cells_count,
 *     sample_cell
 *   })
 *
 * ============================================================================
 * PREVIEW MODAL BEHAVIOR
 * ============================================================================
 *
 * Props (WordExportPreviewProps):
 *   - moduleName: string — e.g., "ANOVA & Genetic Parameters"
 *   - reportTitle: string — e.g., "Multi-Trait Genetics Report"
 *   - datasetSummary: string — e.g., "24 genotypes · 5 traits"
 *   - sections: PreviewSection[] — data to display
 *   - warnings: string[] — amber warning list
 *   - notes: string[] — blue info box list
 *   - canExport: boolean — disable Download if false
 *   - onExport: () => Promise<void> — called on Download click
 *   - onClose: () => void — called on Cancel/close
 *
 * UI Layout:
 *   ┌─────────────────────────────────────┐
 *   │ [CLOSE]                     TITLE   │ ← Header
 *   ├─────────────────────────────────────┤
 *   │ ⚠️ Warnings (if any)                 │ ← Warnings
 *   ├─────────────────────────────────────┤
 *   │ Section 1: Dataset Overview         │
 *   │ ├─ Genotypes        │ 24            │
 *   │ ├─ Traits analysed  │ 5             │
 *   │ ...                                  │
 *   ├─────────────────────────────────────┤
 *   │ Section 2: ...                      │
 *   ├─────────────────────────────────────┤
 *   │ ℹ️ Notes (if any)                    │ ← Notes
 *   ├─────────────────────────────────────┤
 *   │ [CANCEL]  [DOWNLOAD (.docx)] →      │ ← Footer
 *   └─────────────────────────────────────┘
 *
 * Disabled state:
 *   - canExport=false → Download button disabled (gray, no-click)
 *   - Red alert: "No exportable data — missing or malformed"
 *
 * Loading state:
 *   - During onExport(): spinner in button, "Generating…" label
 *   - Error shown in red box if onExport() throws
 *
 * ============================================================================
 * SUCCESS CRITERIA
 * ============================================================================
 *
 * ✅ No crashes when backend returns partial/null data
 * ✅ All modules show consistent "Preview Report" button
 * ✅ Export disabled when data missing/invalid
 * ✅ Placeholders shown instead of broken UI
 * ✅ Word export still works (backend unchanged)
 * ✅ Heatmap never crashes (renders placeholder or valid SVG)
 * ✅ ANOVA and Genetic Parameters follow same safety as correlation
 * ✅ Debug logs (dev only) help troubleshoot data issues
 *
 * ============================================================================
 * FILE STRUCTURE
 * ============================================================================
 *
 * src/
 * ├─ utils/
 * │  ├─ normalizeModuleData.ts    (normalization + validation)
 * │  └─ previewBuilders.ts        (preview section builders)
 * ├─ components/
 * │  ├─ export/
 * │  │  └─ WordExportPreviewModal.tsx    (reusable modal)
 * │  └─ vivasense-genetics/
 * │     ├─ ResultsDisplay.tsx            (ANOVA + genetic params)
 * │     ├─ TraitRelationships.tsx        (correlation analysis)
 * │     ├─ CorrelationHeatmap.tsx        (heatmap rendering)
 * │     └─ ...
 * └─ services/
 *    ├─ geneticsUploadApi.ts      (upload + export)
 *    └─ traitRelationshipsApi.ts  (correlation API)
 *
 * ============================================================================
 */

export {};
