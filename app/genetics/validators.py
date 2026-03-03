"""
Input validation for genetics/breeding datasets.
Returns {"errors": [...], "warnings": [...]} — never raises exceptions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from .config import TrialDesign, MarkerConfig


def validate_trial(df: pd.DataFrame, design: TrialDesign) -> Dict[str, List[str]]:
    """
    Validate a multilocational trial DataFrame before analysis.
    Checks structure, balance, minimum counts, and column types.
    """
    errors: List[str] = []
    warnings: List[str] = []
    g_col = design.genotype_col
    l_col = design.location_col
    r_col = design.rep_col

    # --- Required columns ---
    for col in [g_col, r_col]:
        if col not in df.columns:
            errors.append(f"Required column '{col}' not found in data.")

    if l_col and l_col not in df.columns:
        errors.append(f"Location column '{l_col}' not found in data.")

    if errors:
        return {"errors": errors, "warnings": warnings}

    # --- Minimum rows ---
    if len(df) < 6:
        errors.append("Dataset must have at least 6 rows for meaningful analysis.")

    # --- Genotype count ---
    n_geno = df[g_col].nunique()
    if n_geno < 3:
        errors.append(f"At least 3 genotypes required; found {n_geno}.")

    # --- Location count ---
    if l_col:
        n_loc = df[l_col].nunique()
        if n_loc < 2:
            warnings.append(
                f"Only 1 location found. G×L, AMMI, and GGE analyses will be skipped. "
                "Add data from ≥2 locations for multi-environment analysis."
            )

    # --- Rep count ---
    if r_col in df.columns:
        if l_col:
            rep_counts = df.groupby([g_col, l_col])[r_col].nunique()
        else:
            rep_counts = df.groupby(g_col)[r_col].nunique()
        if rep_counts.min() < 2:
            warnings.append(
                "Some genotype-location cells have fewer than 2 replicates. "
                "Variance component estimates may be unreliable."
            )

    # --- Trait columns ---
    trait_cols = design.trait_cols
    if trait_cols is None:
        structural = {g_col, r_col}
        if l_col:
            structural.add(l_col)
        if design.season_col:
            structural.add(design.season_col)
        if design.block_col:
            structural.add(design.block_col)
        trait_cols = [c for c in df.columns if c not in structural and pd.api.types.is_numeric_dtype(df[c])]

    if len(trait_cols) == 0:
        errors.append("No numeric trait columns found. Check that trait data is numeric.")
    elif len(trait_cols) < 2:
        warnings.append("Only 1 trait found. Correlation, path, and selection index analyses require ≥2 traits.")

    # --- Missing values ---
    for col in trait_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                pct = 100 * n_miss / len(df)
                msg = f"Trait '{col}': {n_miss} missing values ({pct:.1f}%)."
                if pct > 20:
                    warnings.append(msg + " High missingness may bias results.")
                else:
                    warnings.append(msg)

    # --- Balance check ---
    if l_col and l_col in df.columns and g_col in df.columns:
        balance = df.groupby([g_col, l_col]).size()
        expected = balance.mode()[0]
        unbalanced_cells = (balance != expected).sum()
        if unbalanced_cells > 0:
            warnings.append(
                f"Unbalanced design: {unbalanced_cells} genotype-location cells have "
                f"non-modal rep count ({expected}). Variance partitioning uses approximations."
            )

    return {"errors": errors, "warnings": warnings}


def validate_markers(df: pd.DataFrame, config: MarkerConfig) -> Dict[str, List[str]]:
    """
    Validate a binary marker DataFrame (accessions × markers, values 0/1).
    """
    errors: List[str] = []
    warnings: List[str] = []
    acc_col = config.accession_col

    if acc_col not in df.columns:
        errors.append(f"Accession column '{acc_col}' not found.")
        return {"errors": errors, "warnings": warnings}

    # Identify marker columns
    prefix = config.marker_prefix
    marker_cols = [
        c for c in df.columns
        if c != acc_col and (prefix is None or str(c).startswith(prefix))
    ]

    if len(marker_cols) < 5:
        errors.append(f"At least 5 marker columns required; found {len(marker_cols)}.")

    if df[acc_col].nunique() < 4:
        errors.append(f"At least 4 accessions required; found {df[acc_col].nunique()}.")

    # Check binary values
    for col in marker_cols:
        unique_vals = set(df[col].dropna().unique())
        invalid = unique_vals - {0, 1, 0.0, 1.0}
        if invalid:
            errors.append(
                f"Marker column '{col}' contains non-binary values: {invalid}. "
                "All marker values must be 0 (absent) or 1 (present)."
            )

    # Missing data
    total_cells = len(df) * len(marker_cols)
    missing = sum(df[c].isna().sum() for c in marker_cols)
    if missing > 0:
        pct = 100 * missing / total_cells
        warnings.append(f"Marker matrix has {missing} missing values ({pct:.1f}%). Missing treated as unknown.")

    return {"errors": errors, "warnings": warnings}


def detect_trait_cols(df: pd.DataFrame, design: TrialDesign) -> List[str]:
    """Auto-detect trait columns: all numeric columns not in structural role."""
    structural = {design.genotype_col, design.rep_col}
    if design.location_col:
        structural.add(design.location_col)
    if design.season_col:
        structural.add(design.season_col)
    if design.block_col:
        structural.add(design.block_col)
    return [c for c in df.columns if c not in structural and pd.api.types.is_numeric_dtype(df[c])]
