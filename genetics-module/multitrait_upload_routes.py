"""
VivaSense Genetics - Multi-Trait File Upload Endpoints

POST /genetics/upload-preview  — Read file, detect columns, return preview
POST /genetics/analyze-upload  — Analyze each trait using the R genetics engine

Design principle: No genetics logic in this file.
All variance component estimation, heritability calculation, and interpretation
are handled by the R engine (vivasense_genetics.R) through RGeneticsEngine.run_analysis().
There is no Python-level ANOVA derivation here. The R engine is the sole source
of truth for all genetic computations.

Engine integration:
    The shared genetics core is app_genetics.r_engine (RGeneticsEngine).
    It is the same engine used by POST /genetics/analyze.
    Accessed lazily inside handlers via `import app_genetics` to avoid
    capturing the initial None value set before the startup event fires.
"""

import asyncio
import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import dataset_cache
from multitrait_upload_schemas import (
    DatasetSummary,
    DetectedColumn,
    DetectedColumns,
    SummaryTableRow,
    TraitResult,
    UploadAnalysisRequest,
    UploadAnalysisResponse,
    UploadPreviewResponse,
)
from genetics_schemas import GeneticsResponse
import result_cache
from interpretation import InterpretationEngine
from genetics_interpretation import build_breeding_synthesis

logger = logging.getLogger(__name__)



class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


router = APIRouter(tags=["Multi-Trait Upload"], default_response_class=UTF8JSONResponse)


# ============================================================================
# COLUMN DETECTION PATTERNS
# ============================================================================

_GENOTYPE_PATTERNS = frozenset(
    ["genotype", "variety", "cultivar", "entry", "line", "accession", "clone", "treatment"]
)
_REP_PATTERNS = frozenset(
    ["rep", "replication", "block", "replicate", "repetition"]
)
_ENV_PATTERNS = frozenset(
    ["environment", "location", "site", "season", "env", "loc", "year", "place"]
)

# Numeric columns whose names indicate identifiers/indices rather than traits.
# These are excluded from the candidate-trait list even if they are numeric.
_NUMERIC_ID_PATTERNS = frozenset([
    "plot", "plotid", "plot_id", "plotno", "plot_no", "plot_number", "plot_num",
    "row", "col", "column",
    "id", "obs", "observation", "serial", "index", "no", "num", "number",
    "farmerid", "farmer_id", "entry_id", "plant_no", "plant_number", "plant_num",
])


def _match_pattern(col_name: str, patterns: frozenset) -> Optional[str]:
    """
    Return a confidence string if col_name matches any pattern.
    'high' = exact (case-insensitive) match.
    'medium' = col contains pattern or pattern contains col.
    None = no match.
    """
    lower = col_name.lower().strip()
    if lower in patterns:
        return "high"
    for p in patterns:
        if p in lower or lower in p:
            return "medium"
    return None


def _is_numeric_id(col_name: str) -> bool:
    """
    Return True if the column name looks like a numeric identifier
    (plot number, row, ID, etc.) that should not be treated as a trait.
    """
    lower = col_name.lower().strip()
    if lower in _NUMERIC_ID_PATTERNS:
        return True
    # Ends with common ID suffixes
    if lower.endswith(("_id", "_no", "_num", "_number", "_index", "_serial", "_code")):
        return True
    return False


MET_ENVIRONMENT_KEYWORDS = {
    "environment",
    "env",
    "location",
    "loc",
    "site",
    "station",
    "place",
    "season_env",
}

BLOCK_KEYWORDS = {
    "block",
    "blk",
    "rep",
    "replication",
    "replicate",
}


def _contains_keyword(column: str, keywords: set[str]) -> bool:
    name = str(column).strip().lower()
    return any(keyword in name for keyword in keywords)


def suggest_experimental_design(column_names: list[str]) -> str:
    if any(_contains_keyword(column, MET_ENVIRONMENT_KEYWORDS) for column in column_names):
        return "MET"

    # If no block/replication-like column is detected, default to CRD.
    if not any(_contains_keyword(column, BLOCK_KEYWORDS) for column in column_names):
        return "CRD"

    return "RCBD"


def detect_columns(df: pd.DataFrame) -> DetectedColumns:
    """
    Detect structural columns (genotype, rep, environment) by name matching,
    then classify remaining numeric columns as candidate traits — excluding
    obvious numeric identifiers.

    Patterns are matched in priority order: genotype → rep → env.
    A column matched as structural is removed from the trait candidate pool.
    """
    assigned: set = set()
    genotype_col = rep_col = env_col = None

    for col in df.columns:
        if genotype_col is None:
            conf = _match_pattern(col, _GENOTYPE_PATTERNS)
            if conf:
                genotype_col = DetectedColumn(column=col, confidence=conf)
                assigned.add(col)
                continue

        if rep_col is None:
            conf = _match_pattern(col, _REP_PATTERNS)
            if conf:
                rep_col = DetectedColumn(column=col, confidence=conf)
                assigned.add(col)
                continue

        if env_col is None:
            conf = _match_pattern(col, _ENV_PATTERNS)
            if conf:
                env_col = DetectedColumn(column=col, confidence=conf)
                assigned.add(col)
                continue

    # Candidate traits: numeric, not assigned as structural, not an ID column
    trait_cols: List[str] = []
    for col in df.columns:
        if col in assigned:
            continue
        if _is_numeric_id(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            trait_cols.append(col)
        elif df[col].dtype == object:
            # Coerce and accept if ≥70 % of rows parse as numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() >= len(df) * 0.70:
                trait_cols.append(col)

    return DetectedColumns(
        genotype=genotype_col,
        rep=rep_col,
        environment=env_col,
        traits=trait_cols,
    )


# ============================================================================
# FILE READING
# ============================================================================

def read_file(content: bytes, file_type: str) -> pd.DataFrame:
    """Read CSV or Excel bytes into a DataFrame."""
    try:
        if file_type == "csv":
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}") from exc

    if df.empty:
        raise ValueError("File is empty or has no data rows")
    if len(df) < 6:
        raise ValueError(f"File has only {len(df)} rows; minimum 6 required")

    df.columns = [str(c).strip() for c in df.columns]
    return df


# ============================================================================
# OBSERVATION BUILDER
# ============================================================================

def build_observations(
    df: pd.DataFrame,
    genotype_col: str,
    rep_col: Optional[str],
    trait_col: str,
    env_col: Optional[str],
    factor_col: Optional[str] = None,
    design_type: Optional[str] = None,
    main_plot_col: Optional[str] = None,
    sub_plot_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Reshape a wide DataFrame into the flat observation-record format expected
    by RGeneticsEngine.run_analysis().

        Simple RCBD (rep_col provided, no factor_col):
            Single-env: [{"genotype": ..., "rep": ..., "trait_value": ...}, ...]
            Multi-env:  [{"genotype": ..., "environment": ..., "rep": ...,
                          "trait_value": ...}, ...]

        Factorial RCBD (rep_col + factor_col provided):
            [{"genotype": ..., "rep": ..., "factor": ..., "trait_value": ...}, ...]

        Split-Plot RCBD (design_type == split_plot_rcbd):
            [{"genotype": ..., "rep": ..., "main_plot": ..., "sub_plot": ..., "trait_value": ...}, ...]

        Simple CRD (rep_col is None, no factor_col):
            [{"genotype": ..., "rep": "<synthetic>",
              "crd": True, "trait_value": ...}, ...]

        Factorial CRD (rep_col is None, factor_col or legacy env_col provided):
            [{"genotype": ..., "rep": "<synthetic>", "crd": True,
              "factor": ..., "trait_value": ...}, ...]

    factor_col takes explicit priority.  For CRD datasets, if factor_col is
    None but env_col is provided, env_col is used as the treatment factor
    (legacy factorial CRD path).  For RCBD datasets env_col always means
    multi-environment, never a treatment factor.

    For split-plot RCBD the build process includes main_plot and sub_plot
    fields for the R engine.

    For CRD datasets the rep field is a synthetic row-counter within each
    genotype group (e.g. "R1", "R2", …).  The R engine uses the "crd" flag
    to select the correct model formula instead of blocking on rep.

    Rows where trait_col is non-numeric or null are dropped.
    Raises ValueError if fewer than 6 valid observations remain.

    NOTE: The R engine (vivasense_genetics.R) performs all ANOVA computations.
    build_observations() is a pure data-reshaping step.
    """
    crd_mode = rep_col is None

    if design_type == "split_plot_rcbd":
        if rep_col is None or main_plot_col is None or sub_plot_col is None:
            raise ValueError("Split-plot RCBD requires rep_column, main_plot_column, and sub_plot_column")
        # Genotype is NOT part of the generic split-plot model.
        # Only structural role columns + trait are kept.
        keep_cols = [rep_col, main_plot_col, sub_plot_col, trait_col]
    else:
        # Determine the effective factor column.
        # - factor_col is the explicit factorial RCBD / CRD treatment factor.
        # - For CRD datasets, env_col is the legacy path for factorial CRD.
        # - For RCBD datasets, env_col means multi-environment (never a factor here).
        eff_factor = factor_col or (env_col if crd_mode else None)

        if crd_mode:
            keep_cols = [genotype_col, trait_col]
            if eff_factor:
                keep_cols = [genotype_col, eff_factor, trait_col]
        else:
            keep_cols = [genotype_col, rep_col, trait_col]
            if env_col and not factor_col:
                # Multi-environment RCBD
                keep_cols = [genotype_col, env_col, rep_col, trait_col]
            elif factor_col:
                # Factorial RCBD
                keep_cols = [genotype_col, rep_col, factor_col, trait_col]

    # Remove empty/None column names before subsetting (e.g., blank rep_column in CRD payloads)
    keep_cols = [c for c in keep_cols if c and c.strip()]
    subset = df[keep_cols].copy()
    subset[trait_col] = pd.to_numeric(subset[trait_col], errors="coerce")
    subset = subset.dropna(subset=[trait_col])

    if len(subset) < 6:
        raise ValueError(
            f"Trait '{trait_col}': only {len(subset)} valid observation(s) after "
            "dropping missing values (minimum 6 required)"
        )

    if crd_mode:
        # Assign synthetic replication numbers within each genotype group.
        # R uses these for n_reps estimation only — they are NOT included in
        # the CRD model formula.
        subset = subset.copy()
        subset["_synth_rep"] = (
            subset.groupby(genotype_col).cumcount().add(1)
            .astype(str).radd("R")
        )

    records: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        if design_type == "split_plot_rcbd":
            # Generic split-plot: role-based record with no genotype term.
            # R formula: trait_value ~ main_plot * sub_plot + Error(rep/main_plot)
            rec: Dict[str, Any] = {
                "rep":         str(row[rep_col]),
                "main_plot":   str(row[main_plot_col]),
                "sub_plot":    str(row[sub_plot_col]),
                "trait_value": float(row[trait_col]),
            }
            records.append(rec)
            continue

        # All other designs carry the observation-unit identifier.
        rec = {
            "genotype":    str(row[genotype_col]),
            "trait_value": float(row[trait_col]),
        }
        if crd_mode:
            rec["rep"] = str(row["_synth_rep"])
            rec["crd"] = True
            if eff_factor:
                rec["factor"] = str(row[eff_factor])
        else:
            rec["rep"] = str(row[rep_col])
            if factor_col:
                # Factorial RCBD: factor key signals R to use factorial model
                rec["factor"] = str(row[factor_col])
            elif env_col:
                # Multi-environment RCBD
                rec["environment"] = str(row[env_col])
        records.append(rec)

    return records


# ============================================================================
# BALANCE CHECKING
# ============================================================================

def check_balance(
    df: pd.DataFrame,
    genotype_col: Optional[str],
    rep_col: Optional[str],
    trait_col: str,
    env_col: Optional[str],
    factor_col: Optional[str] = None,
    design_type: Optional[str] = None,
    main_plot_col: Optional[str] = None,
    sub_plot_col: Optional[str] = None,
) -> List[str]:
    """
    Return human-readable warnings for unbalanced or incomplete experimental
    structure relevant to the given trait.

    Checks performed:
    - CRD (rep_col is None): whether all genotypes have ≥ 2 observations and
      equal replication (informational only — CRD can tolerate some imbalance).
    - RCBD single-env: whether all genotypes have the same number of observations.
    - Factorial RCBD (rep_col + factor_col): whether each genotype×factor cell
      has the same number of replications.
    - Multi-env: whether each genotype appears in every environment (completeness)
      and whether cell sizes are equal (balance).
    - Split-plot RCBD: whether rep × main_plot × sub_plot cells are complete
      and balanced.

    The R engine can still analyse unbalanced data, but results may be less
    reliable. Warnings are surfaced in TraitResult.data_warnings.
    """
    mask = pd.to_numeric(df[trait_col], errors="coerce").notna()
    sub = df.loc[mask].copy()
    warnings: List[str] = []

    n_genotypes = sub[genotype_col].nunique() if genotype_col else 0
    crd_mode = rep_col is None

    if factor_col and not crd_mode:
        # Factorial RCBD — check balance within genotype×factor cells
        n_factors = sub[factor_col].nunique()

        factors_per_geno = sub.groupby(genotype_col)[factor_col].nunique()
        incomplete = int((factors_per_geno < n_factors).sum())
        if incomplete:
            warnings.append(
                f"{incomplete} of {n_genotypes} genotype(s) missing from at least one "
                f"factor level — incomplete factorial structure "
                f"(expected {n_factors} factor levels per genotype)"
            )

        cell_sizes = sub.groupby([genotype_col, factor_col]).size()
        min_cell = int(cell_sizes.min())
        max_cell = int(cell_sizes.max())
        if min_cell != max_cell:
            warnings.append(
                f"Unbalanced factorial RCBD: genotype×factor cells range from "
                f"{min_cell} to {max_cell} replications"
            )
    elif design_type == "split_plot_rcbd" and main_plot_col and sub_plot_col:
        # Split-plot RCBD — check balance of rep × main_plot × sub_plot cells
        reps = sorted(sub[rep_col].dropna().unique())
        main_levels = sorted(sub[main_plot_col].dropna().unique())
        sub_levels = sorted(sub[sub_plot_col].dropna().unique())

        observed_pairs = {
            (str(r), str(m))
            for r, m in sub[[rep_col, main_plot_col]].drop_duplicates().itertuples(index=False, name=None)
        }
        expected_pairs = {(str(r), str(m)) for r in reps for m in main_levels}
        missing_pairs = expected_pairs - observed_pairs
        if missing_pairs:
            warnings.append(
                f"Incomplete split-plot layout: {len(missing_pairs)} missing rep × main_plot combinations."
            )

        subplot_cells = sub.groupby([rep_col, main_plot_col, sub_plot_col]).size()
        if len(subplot_cells) < len(reps) * len(main_levels) * len(sub_levels):
            warnings.append(
                "Incomplete split-plot structure: not all rep × main_plot × sub_plot cells are present."
            )

        if len(subplot_cells) > 0:
            min_cell = int(subplot_cells.min())
            max_cell = int(subplot_cells.max())
            if min_cell != max_cell:
                warnings.append(
                    f"Unbalanced split-plot RCBD: rep × main_plot × sub_plot cells range from "
                    f"{min_cell} to {max_cell} observations"
                )
    elif env_col and not crd_mode:
        # Multi-environment RCBD
        n_envs = sub[env_col].nunique()

        envs_per_geno = sub.groupby(genotype_col)[env_col].nunique()
        incomplete = int((envs_per_geno < n_envs).sum())
        if incomplete:
            warnings.append(
                f"{incomplete} of {n_genotypes} genotype(s) missing from at least one "
                f"environment — incomplete G×E structure "
                f"(expected {n_envs} environments per genotype)"
            )

        cell_sizes = sub.groupby([genotype_col, env_col]).size()
        min_cell = int(cell_sizes.min())
        max_cell = int(cell_sizes.max())
        if min_cell != max_cell:
            warnings.append(
                f"Unbalanced replication: genotype×environment cells range from "
                f"{min_cell} to {max_cell} observations"
            )
    else:
        # Single environment (RCBD or CRD) — check replication counts per genotype
        obs_per_geno = sub.groupby(genotype_col).size()
        min_obs = int(obs_per_geno.min())
        max_obs = int(obs_per_geno.max())

        if min_obs < 2:
            warnings.append(
                f"At least one genotype has only {min_obs} observation(s); "
                "a minimum of 2 replications per genotype is required for variance estimation."
            )
        elif min_obs != max_obs:
            if crd_mode:
                warnings.append(
                    f"Unbalanced CRD: genotypes have between {min_obs} and "
                    f"{max_obs} observations — replication is inferred from data"
                )
            else:
                warnings.append(
                    f"Unbalanced design: genotypes have between {min_obs} and "
                    f"{max_obs} observations (expected equal replication)"
                )

    return warnings


# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def _build_summary_row(trait: str, result_dict: Dict[str, Any], actual_module: str = "genetic_parameters") -> SummaryTableRow:
    """Extract scalar metrics from a GeneticsResponse dict for the summary table."""
    res = result_dict.get("result") or {}
    if actual_module == "anova":
        hp = {}
        gp = {}
    else:
        hp = res.get("heritability") or {}
        gp = res.get("genetic_parameters") or {}
    h2 = hp.get("h2_broad_sense")
    gam_percent = gp.get("GAM_percent")
    h2_class = InterpretationEngine.classify_heritability(h2) if h2 is not None else None
    gam_class = InterpretationEngine.classify_gam(gam_percent) if gam_percent is not None else None
    return SummaryTableRow(
        trait=trait,
        grand_mean=res.get("grand_mean"),
        h2=h2,
        gcv=gp.get("GCV"),
        pcv=gp.get("PCV"),
        gam_percent=gam_percent,
        heritability_class=h2_class,
        gam_class=gam_class,
        status="success",
    )


def _extract_gxe_stats(analysis_result: GeneticsResponse) -> tuple[Optional[float], Optional[float]]:
    at = analysis_result.result.anova_table if analysis_result.result else None
    if at is None:
        return None, None

    aliases = {
        "genotype:environment",
        "environment:genotype",
        "genotype x environment",
        "environment x genotype",
        "genotype*environment",
        "environment*genotype",
        "gxe",
        "gxe interaction",
    }

    def _norm(label: Any) -> str:
        s = str(label).strip().lower()
        # Keep alnum plus separators used in ANOVA labels, collapse spaces.
        s = " ".join(s.replace("×", "x").split())
        return s

    for idx, src in enumerate(at.source or []):
        src_norm = _norm(src)
        compact = src_norm.replace(" ", "")
        if (
            src_norm in aliases
            or compact in {"genotype:environment", "environment:genotype", "genotypexenvironment", "environmentxgenotype", "genotype*environment", "environment*genotype", "gxe", "gxeinteraction"}
            or ("genotype" in compact and "environment" in compact and any(sep in compact for sep in [":", "x", "*"]))
        ):
            f_val = at.f_value[idx] if idx < len(at.f_value) else None
            p_val = at.p_value[idx] if idx < len(at.p_value) else None
            return f_val, p_val
    return None, None


def _extract_genotype_stats(analysis_result: GeneticsResponse) -> tuple[Optional[float], Optional[float], Optional[bool]]:
    at = analysis_result.result.anova_table if analysis_result.result else None
    if at is None:
        return None, None, None

    aliases = {"genotype", "genotypes"}

    def _norm(label: Any) -> str:
        s = str(label).strip().lower()
        s = " ".join(s.replace("×", "x").split())
        return s

    for idx, src in enumerate(at.source or []):
        src_norm = _norm(src)
        if src_norm in aliases:
            f_val = at.f_value[idx] if idx < len(at.f_value) else None
            p_val = at.p_value[idx] if idx < len(at.p_value) else None
            f_num = float(f_val) if f_val is not None else None
            p_num = float(p_val) if p_val is not None else None
            return f_num, p_num, (p_num <= 0.05 if p_num is not None else None)

    return None, None, None


def _build_breeding_input(
    summary_table: List[SummaryTableRow],
    trait_results: Dict[str, TraitResult],
) -> List[Dict[str, Any]]:
    summary_map = {row.trait: row for row in summary_table if row.status == "success"}
    synthesis_input: List[Dict[str, Any]] = []

    for trait_name, tr in trait_results.items():
        if tr.status != "success" or tr.analysis_result is None or tr.analysis_result.result is None:
            continue

        result = tr.analysis_result.result
        summary_row = summary_map.get(trait_name)
        mean_sep = result.mean_separation

        genotype_means: List[Dict[str, Any]] = []
        top_genotype: Optional[str] = None
        if mean_sep is not None and mean_sep.genotype and mean_sep.mean:
            means = [float(m) if m is not None else float("-inf") for m in mean_sep.mean]
            order = sorted(range(len(mean_sep.genotype)), key=lambda i: means[i], reverse=True)
            rank_map = {idx: rank + 1 for rank, idx in enumerate(order)}

            if order:
                top_genotype = str(mean_sep.genotype[order[0]])

            for idx, geno in enumerate(mean_sep.genotype):
                mean_val = mean_sep.mean[idx] if idx < len(mean_sep.mean) else None
                group_val = mean_sep.group[idx] if idx < len(mean_sep.group) else None
                genotype_means.append({
                    "genotype": str(geno),
                    "mean": float(mean_val) if mean_val is not None else None,
                    "rank": int(rank_map.get(idx, len(mean_sep.genotype))),
                    "group": str(group_val) if group_val is not None else "",
                })

        f_gxe, p_gxe = _extract_gxe_stats(tr.analysis_result)
        f_genotype, p_genotype, genotype_significant = _extract_genotype_stats(tr.analysis_result)
        h2_value = None
        if summary_row is not None and summary_row.h2 is not None:
            h2_value = float(summary_row.h2)
        elif result.heritability and result.heritability.get("h2_broad_sense") is not None:
            h2_value = float(result.heritability.get("h2_broad_sense"))

        gam_class = summary_row.gam_class if summary_row is not None else None

        synthesis_input.append({
            "trait_name": trait_name,
            "h2": h2_value,
            "gam_class": gam_class,
            "top_genotype": top_genotype,
            "f_gxe": float(f_gxe) if f_gxe is not None else None,
            "p_gxe": float(p_gxe) if p_gxe is not None else None,
            "f_value": f_genotype,
            "p_value": p_genotype,
            "genotype_significant": genotype_significant,
            "genotype_means": genotype_means,
        })

    return synthesis_input


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/genetics/upload-preview",
    response_model=UploadPreviewResponse,
    summary="Preview uploaded file and detect columns",
)
async def upload_preview(file: UploadFile = File(...)):
    """
    Read an uploaded CSV or Excel file, detect structural columns (genotype,
    replication, environment) and candidate trait columns, and return a
    5-row data preview.

    No genetics computation is performed here. The user should confirm the
    column mapping in the UI before calling /genetics/analyze-upload.
    """
    filename = file.filename or ""
    if filename.endswith(".csv"):
        file_type = "csv"
    elif filename.endswith((".xlsx", ".xls")):
        file_type = "xlsx"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .csv or .xlsx/.xls file.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = read_file(content, file_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detected = detect_columns(df)

    warn: List[str] = []
    if detected.genotype is None:
        warn.append("Could not detect genotype column — please specify manually")
    if detected.rep is None:
        warn.append(
            "No replication column detected — CRD assumed "
            "(replication inferred from data)"
        )
    if not detected.traits:
        warn.append(
            "No numeric trait columns detected. "
            "Verify that trait data is numeric and not labelled as an ID column."
        )
    else:
        warn.append(
            "If your column contains numeric treatment levels (e.g. 0, 50, 100 kg N/ha or storage days 1, 3, 6, 9), toggle it to Treatment Factor."
        )

    suggested_design = suggest_experimental_design(list(df.columns))
    mode_suggestion = "multi" if detected.environment is not None else "single"

    # Serialise preview rows: replace NaN / NA with None (no numpy required)
    preview_head = df.head(5)
    data_preview: List[Dict[str, Any]] = [
        {
            col: (None if pd.isna(val) else val)
            for col, val in row.items()
        }
        for row in preview_head.to_dict(orient="records")
    ]

    # Register dataset in cache using auto-detected column defaults.
    # This gives the frontend a valid dataset_token immediately — so
    # /analysis/descriptive-stats (and other module endpoints) can be called
    # without requiring a separate POST /upload/dataset confirmation step.
    # The token is superseded when the user confirms mappings via /upload/dataset.
    preview_token: Optional[str] = None
    try:
        b64 = base64.b64encode(content).decode("ascii")
        preview_token = dataset_cache.create_token()
        dataset_cache.put_dataset(preview_token, {
            "base64_content":     b64,
            "file_type":          file_type,
            "genotype_column":    detected.genotype.column if detected.genotype else None,
            "rep_column":         detected.rep.column if detected.rep else None,
            "environment_column": detected.environment.column if detected.environment else None,
            "factor_column":      None,
            "main_plot_column":   None,
            "sub_plot_column":    None,
            "mode":               mode_suggestion,
            "design_type":        mode_suggestion,
            "random_environment": False,
            "selection_intensity": 2.06,
        })
        logger.info("upload-preview: auto-registered dataset token %s", preview_token)
    except Exception as exc:
        logger.warning("upload-preview: failed to register dataset token — %s", exc)
        preview_token = None

    return UploadPreviewResponse(
        detected_columns=detected,
        n_rows=len(df),
        n_columns=len(df.columns),
        data_preview=data_preview,
        suggested_design=suggested_design,
        mode_suggestion=mode_suggestion,
        column_names=list(df.columns),
        warnings=warn,
        dataset_token=preview_token,
    )


@router.post(
    "/genetics/analyze-upload",
    response_model=UploadAnalysisResponse,
    summary="Analyze all traits in an uploaded file",
)
async def analyze_upload(request: UploadAnalysisRequest, module: Optional[str] = None):
    """
    For each requested trait column, reshape the uploaded data into flat
    observation records and call the existing R genetics engine
    (RGeneticsEngine.run_analysis) — the same engine as POST /genetics/analyze.

    One trait failing does not abort the others. Failed traits are recorded
    with their error messages in trait_results and failed_traits.

    The R engine (vivasense_genetics.R) performs all ANOVA computations.
    This endpoint contains no genetics computation logic.
    """
    # module can arrive as a URL query param OR in the JSON body.
    # Body field takes priority; query param is the fallback; default is genetic_parameters.
    actual_module = getattr(request, "module", None) or module or "genetic_parameters"
    print(f"[MODULE ROUTING] Module selected: {actual_module}", flush=True)

    # Lazy import: r_engine is None at module load time and is assigned by
    # the FastAPI startup event in app_genetics.py. Accessing it through the
    # module object always gives the current (post-startup) value.
    import app_genetics  # noqa: PLC0415
    if app_genetics.r_engine is None:
        raise HTTPException(status_code=503, detail="R genetics engine not ready")
    r_engine = app_genetics.r_engine

    # Decode file
    try:
        file_bytes = base64.b64decode(request.base64_content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 content") from exc

    try:
        df = read_file(file_bytes, request.file_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("analyze-upload dataframe shape: %s", df.shape)
    logger.info("analyze-upload columns: %s", list(df.columns))
    logger.info("analyze-upload head:\n%s", df.head())

    # Normalise empty-string rep_column to None so CRD requests do not fall into RCBD paths.
    rep_column = request.rep_column
    if not rep_column or rep_column.strip() == "":
        rep_column = None

    # Validate that all named columns actually exist (rep_column is optional)
    required_cols = [request.genotype_column]
    if rep_column:
        required_cols.append(rep_column)
    required_cols += request.trait_columns
    if request.environment_column:
        required_cols.append(request.environment_column)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in file: {missing}",
        )

    # Dataset-level summary (whole file, not per-trait)
    n_genotypes = int(df[request.genotype_column].nunique())
    if rep_column:
        n_reps = int(df[rep_column].nunique())
    else:
        # CRD: infer max observations per genotype as effective n_reps
        n_reps = int(df.groupby(request.genotype_column).size().max())
    n_environments = (
        int(df[request.environment_column].nunique())
        if request.environment_column
        else None
    )
    dataset_summary = DatasetSummary(
        n_genotypes=n_genotypes,
        n_reps=n_reps,
        n_environments=n_environments,
        n_traits=len(request.trait_columns),
        mode=request.mode,
    )

    summary_table: List[SummaryTableRow] = []
    trait_results: Dict[str, TraitResult] = {}
    failed_traits: List[str] = []
    anova_type_warning: Optional[str] = None

    env_col_for_mode = request.environment_column if request.mode == "multi" else None
    # factor_column is only applicable in single-env mode
    factor_col = getattr(request, "factor_column", None) if request.mode == "single" else None
    numeric_factor_columns = [
        c for c in getattr(request, "numeric_factor_columns", [])
        if c and c.strip() and c in df.columns
    ]
    if request.mode == "single" and rep_column and not factor_col and len(numeric_factor_columns) == 1:
        # Numeric-coded treatment levels toggled as factors in UI.
        factor_col = numeric_factor_columns[0]

    # Bounded concurrency: Limit active R subprocesses to prevent Out-Of-Memory (OOM)
    # errors when processing massive datasets with 50+ traits.
    MAX_CONCURRENT_R_PROCESSES = 4
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_R_PROCESSES)

    crd_mode = rep_column is None and request.mode == "single"

    async def analyze_single_trait(trait: str):
        async with semaphore:
            print(f"[PIPELINE] Running {actual_module.upper()} pipeline for trait: {trait}", flush=True)
            logger.info("Analyzing trait: %s (module=%s, crd=%s)", trait, actual_module, crd_mode)
            try:
                balance_warnings = check_balance(
                    df=df,
                    genotype_col=request.genotype_column,
                    rep_col=rep_column,
                    trait_col=trait,
                    env_col=env_col_for_mode,
                    factor_col=factor_col,
                )
                if balance_warnings:
                    for w in balance_warnings:
                        logger.warning("Trait '%s': %s", trait, w)

                observations = build_observations(
                    df=df,
                    genotype_col=request.genotype_column,
                    rep_col=rep_column,
                    trait_col=trait,
                    env_col=env_col_for_mode,
                    factor_col=factor_col,
                )

                # Execute blocking R subprocess concurrently via ThreadPool
                result_dict = await asyncio.to_thread(
                    r_engine.run_analysis,
                    data=observations,
                    mode=request.mode,
                    trait_name=trait,
                    random_environment=request.random_environment,
                    crd_mode=crd_mode,
                    selection_intensity=request.selection_intensity,
                )

                # R returns status="ERROR" when computation fails
                if result_dict.get("status") == "ERROR":
                    r_errors = result_dict.get("errors") or {}
                    r_msg = (
                        result_dict.get("interpretation")
                        or next(iter(r_errors.values()), None)
                        or str(r_errors)
                        or "R analysis returned ERROR"
                    )
                    gxe_rep_match = re.search(r"Minimum reps per G(?:x|×)E:\s*(\d+)", str(r_msg), flags=re.IGNORECASE)
                    if gxe_rep_match:
                        min_reps = int(gxe_rep_match.group(1))
                        replication_word = "replication" if min_reps == 1 else "replications"
                        raise RuntimeError(
                            f"Insufficient replications: This dataset has only {min_reps} {replication_word} "
                            "per genotype-environment combination. VivaSense requires at least 2 replications "
                            "per GxE cell to estimate variance components reliably. Please check your data "
                            "structure or use a dataset with >= 2 replications."
                        )
                    raise RuntimeError(f"R ERROR: {r_msg}")

                r_result = result_dict.get("result") or {}
                
                # Enforce strict module isolation — ANOVA never exposes genetic parameters
                if actual_module == "anova":
                    if "result" in result_dict and isinstance(result_dict["result"], dict):
                        # Use {} not None: GeneticsResult.heritability / genetic_parameters
                        # are non-Optional Dict fields — None would fail Pydantic validation
                        result_dict["result"]["genetic_parameters"] = {}
                        result_dict["result"]["heritability"] = {}
                    # Clear interpretation completely — ANOVA interpretation comes
                    # from the Academic Mentor, not from InterpretationEngine
                    result_dict["interpretation"] = None
                    result_dict["breeding_implication"] = None
                
                logger.info(
                    "Trait '%s' R result keys: %s",
                    trait, list(r_result.keys())
                )

                # Validate the dict against the real GeneticsResponse schema.
                validated = GeneticsResponse(**result_dict)

                return trait, "success", validated, balance_warnings, result_dict, None

            except Exception as exc:
                import traceback as _tb
                print(f"[TRAIT_FAIL] trait={trait} exc_type={type(exc).__name__} exc={exc}", flush=True)
                print(_tb.format_exc(), flush=True)
                logger.warning("Trait '%s' failed: %s", trait, exc)
                return trait, "failed", None, [], None, str(exc)

    # Execute all trait analyses concurrently, respecting the semaphore limit
    tasks = [analyze_single_trait(trait) for trait in request.trait_columns]
    concurrent_results = await asyncio.gather(*tasks)

    # Aggregate results into response payload
    for trait, status, validated, balance_warnings, result_dict, error_msg in concurrent_results:
        if status == "success":
            trait_results[trait] = TraitResult(
                status="success",
                analysis_result=validated,
                data_warnings=balance_warnings,
            )
            if not anova_type_warning and getattr(validated, "anova_type_warning", None):
                anova_type_warning = validated.anova_type_warning
            summary_table.append(_build_summary_row(trait, result_dict, actual_module))
        else:
            failed_traits.append(trait)
            trait_results[trait] = TraitResult(
                status="failed",
                analysis_result=None,
                error=error_msg,
            )
            summary_table.append(
                SummaryTableRow(trait=trait, status="failed", error=error_msg)
            )

    # Ensure session state is preserved even if individual analyses fail
    categorical_columns: List[str] = []
    for col in [
        request.genotype_column,
        rep_column,
        request.environment_column,
        factor_col,
        getattr(request, "main_plot_column", None),
        getattr(request, "sub_plot_column", None),
        *numeric_factor_columns,
    ]:
        if col and col not in categorical_columns:
            categorical_columns.append(col)

    dataset_token = dataset_cache.create_token()
    dataset_cache.put_dataset(dataset_token, {
        "base64_content":     request.base64_content,
        "file_type":          request.file_type,
        "genotype_column":    request.genotype_column,
        "rep_column":         rep_column,
        "environment_column": request.environment_column,
        "factor_column":      factor_col,
        "numeric_factor_columns": numeric_factor_columns,
        "categorical_columns": categorical_columns,
        "main_plot_column":   getattr(request, "main_plot_column", None),
        "sub_plot_column":    getattr(request, "sub_plot_column", None),
        "mode":               request.mode,
        "design_type":        request.mode,
        "random_environment": request.random_environment,
        "selection_intensity": request.selection_intensity,
    })

    # Build the full response first (without token so the object is complete)
    response = UploadAnalysisResponse(
        summary_table=summary_table,
        trait_results=trait_results,
        dataset_summary=dataset_summary,
        failed_traits=failed_traits,
        anova_type_warning=anova_type_warning,
        dataset_token=dataset_token,
        breeding_summary=build_breeding_synthesis(_build_breeding_input(summary_table, trait_results)),
    )

    print(f"[EXPORT] Generating {actual_module.upper()} report — {len(summary_table)} trait(s) processed", flush=True)

    # Cache the complete response and attach the token.
    # The frontend echoes the token back when it calls /download-results,
    # allowing the export endpoint to recover analysis_result objects that
    # the frontend did not include in its POST body.
    token = result_cache.create_token()
    result_cache.put(token, response)
    return response.model_copy(update={"export_token": token})
