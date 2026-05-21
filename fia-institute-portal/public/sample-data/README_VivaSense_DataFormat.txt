================================================================
  VivaSense Sample Data Files — Format Guide
  Field-to-Insight Academy | fieldtoinsightacademy.com.ng
================================================================

These sample files show the exact CSV format required for each
analysis design in VivaSense. Download any file, study the
column arrangement, then prepare your own data the same way.
This guide currently includes 7 sample files.

----------------------------------------------------------------
FILE 1: 01_oneway_CRD.csv
  Design  : One-way ANOVA (Completely Randomised Design)
  Crop    : Cowpea genotype trial
  Columns : Genotype, Yield_kg_ha, Days_to_flowering, 100_seed_weight_g
  Use when: Treatments assigned randomly, no blocking
  Required: 1 factor column + 1 or more trait columns
----------------------------------------------------------------

----------------------------------------------------------------
FILE 2: 05_oneway_RCBD.csv
  Design  : One-way ANOVA (Randomised Complete Block Design)
  Crop    : Maize variety trial
  Columns : Block, Variety, Yield_kg_ha, Plant_height_cm, Ear_length_cm
  Use when: Field trial with blocks controlling environmental variation
  Required: Block column + Treatment column + 1 or more trait columns
  Note    : Block column MUST be named to match your block column in VivaSense
----------------------------------------------------------------

----------------------------------------------------------------
FILE 3: 02_twoway_CRD_factorial.csv
  Design  : Two-way Factorial ANOVA (CRD)
  Crop    : Maize nitrogen x variety trial
  Columns : Nitrogen, Variety, Yield_kg_ha, Plant_height_cm, Ear_length_cm
  Use when: Two treatment factors, no blocking
  Required: Factor A column + Factor B column + trait column(s)
----------------------------------------------------------------

----------------------------------------------------------------
FILE 4: 03_factorial_RCBD.csv
  Design  : Factorial ANOVA in RCBD
  Crop    : Sorghum variety x fertilizer trial
  Columns : Block, Variety, Fertilizer, Grain_yield_kg_ha, Stover_yield_kg_ha, Plant_height_cm
  Use when: Two treatment factors with blocking
  Required: Block + Factor A + Factor B + trait column(s)
----------------------------------------------------------------

----------------------------------------------------------------
FILE 5: 04_splitplot.csv
  Design  : Split-plot ANOVA
  Crop    : Cassava irrigation x variety trial
  Columns : Block, Irrigation, Variety, Fresh_root_yield_t_ha, Dry_matter_percent, HCN_mg_kg
  Use when: Main plot factor is harder to change (e.g. irrigation)
            Sub-plot factor is easier to randomise (e.g. variety)
  Required: Block + Main plot factor + Sub-plot factor + trait column(s)
----------------------------------------------------------------

----------------------------------------------------------------
FILE 6: 06_multitrait_RCBD.csv
  Design  : Multi-trait Analysis (any design)
  Crop    : Cassava genotype evaluation
  Columns : Genotype, Block, Fresh_root_yield_t_ha, Dry_matter_percent,
            HCN_mg_kg, Plant_height_cm, Branching_height_cm,
            Stem_diameter_cm, Leaf_retention_score
  Use when: You want to analyse multiple traits simultaneously
  Required: Design columns + 2 to 15 numeric trait columns
  Note    : Select only the trait columns in the Traits to Analyse field
            Do NOT select Block, Genotype, or design columns as traits
----------------------------------------------------------------

----------------------------------------------------------------
FILE 7: vivasense_sample_MET.csv
  Design  : Multi-Environment Trial (MET)
  Crop    : Maize MET workshop dataset
  Columns : Environment, Genotype, Replication, Yield_kg_ha,
            Plant_height_cm, Days_to_maturity, 100seed_weight_g
  Use when: Comparing genotype performance and GxE across environments
  Required: Environment + Genotype + Replication + numeric trait column(s)
  Note    : Balanced example = 5 genotypes × 4 environments × 3 reps
            Suitable for GGE biplot and AMMI analysis
----------------------------------------------------------------

================================================================
  GENERAL RULES FOR YOUR CSV FILE
================================================================

1. COLUMN NAMES
   - Use simple names without spaces (use underscore: Yield_kg_ha)
   - Avoid Python reserved words as column names:
     NEVER use: yield, lambda, class, return, import, pass, break
   - Good examples: Yield_t_ha, Plant_ht_cm, Days_flower
   - Columns starting with numbers are supported but use with care:
     100seed_weight_g is valid; prefer seed_weight_100g for maximum compatibility

2. DATA FORMAT
   - Trait columns must contain numbers only
   - Factor/treatment columns can be text (GENOTYPE, Block, etc.)
   - No merged cells (this is not Excel — plain CSV only)
   - First row must be column headers
   - No empty rows between data

3. FILE FORMAT
   - Save as CSV (Comma Separated Values)
   - UTF-8 encoding
   - Maximum recommended: 500 rows, 20 columns

4. REPLICATION
   - Minimum 2 replicates per treatment for ANOVA
   - Recommended 3-4 replicates for reliable results

5. MISSING DATA
   - VivaSense handles missing values automatically
   - Rows with missing trait values are excluded from analysis
   - Flag missing values as empty cells (not zeros)

6. MET ANALYSIS REQUIREMENTS
  - Environment column must contain environment labels
  - Genotype column must contain genotype names
  - Replication column must be numeric
  - Trait columns must be numeric with no missing values for MET analysis

================================================================
  QUICK REFERENCE: WHICH DESIGN TO USE?
================================================================

My trial has...                         Use this design
---------------------------------------------------------------
One treatment factor, no blocks      →  One-way ANOVA (CRD)
One treatment factor, with blocks    →  One-way ANOVA (RCBD)  ← most common
Two factors, no blocks               →  Two-way Factorial (CRD)
Two factors, with blocks             →  Factorial RCBD
Two factors, one harder to change    →  Split-plot
Multiple traits, any design          →  Multi-trait Analysis
Multiple environments, genotype comparison  →  Multi-Environment Trial (MET)

================================================================
  NEED HELP?
================================================================

Contact: info@fieldtoinsightacademy.com.ng
Website: https://fieldtoinsightacademy.com.ng/vivasense
WhatsApp: +234 902 215 8026 (Field-to-Insight Academy)

Generated by VivaSense — Field-to-Insight Academy
================================================================
