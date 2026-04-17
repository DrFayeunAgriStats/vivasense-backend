"""
VivaSense Genetics – Trait Relationships Endpoints
Phase 2 / Phase 1 scope: phenotypic correlation only.

POST /genetics/correlation
    Computes Pearson or Spearman phenotypic correlation on per-genotype means
    for all requested trait pairs in an uploaded CSV/XLSX file.

Design principle identical to multitrait_upload_routes:
    No genetics logic in Python. The R engine (vivasense_trait_relationships.R)
    performs all computation. Python validates, reshapes, dispatches, validates
    the response contract, and returns.

Engine access:
    TraitRelationshipsEngine is a sibling to RGeneticsEngine in app_genetics.py.
    It calls vivasense_trait_relationships.R via the same subprocess pattern.
    Initialised by init_trait_relationships_engine() called from
    app_genetics.startup_event (non-fatal — app continues if R script missing).
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from trait_relationships_schemas import CorrelationRequest, CorrelationResponse

# Import interpretation helpers (no circular dependencies — this module is a leaf)
from trait_association_interpretation import (
    generate_trait_association_interpretation,
    _compute_risk_flags,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Trait Relationships"])

_R_SCRIPT = "vivasense_trait_relationships.R"

# Included in every CorrelationResponse so consumers always know the
# statistical basis without having to read the documentation.
_STATISTICAL_NOTE = (
    "Correlations were computed using genotype-level means; "
    "significance levels are based on the number of genotypes."
)


# ============================================================================
# ENGINE
# ============================================================================

class TraitRelationshipsEngine:
    """
    Thin subprocess wrapper for vivasense_trait_relationships.R.
    Mirrors RGeneticsEngine in app_genetics.py.
    """

    def __init__(self, r_script_path: str = _R_SCRIPT):
        self.r_script_path = r_script_path
        if not Path(r_script_path).exists():
            raise FileNotFoundError(f"R script not found: {r_script_path}")
        logger.info("TraitRelationshipsEngine initialised: %s", r_script_path)

    def run_correlation(
        self,
        records: List[Dict[str, Any]],
        trait_cols: List[str],
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """
        Execute run_correlation_analysis() in R and return the parsed dict.

        records   — wide-format observation dicts (genotype, rep, trait1, …)
        trait_cols — names of trait columns within each record
        method    — "pearson" or "spearman"
        """
        data_json = json.dumps(records)
        # json.dumps gives ["t1","t2"] which is invalid R syntax.
        # Build R c("t1","t2",...) instead. json.dumps(t) handles quotes/escaping.
        trait_cols_r = "c(" + ", ".join(json.dumps(t) for t in trait_cols) + ")"

        r_code = f"""
source("{self.r_script_path}")

data_list <- jsonlite::fromJSON('{data_json}')
data <- as.data.frame(data_list)

trait_cols <- {trait_cols_r}
for (col in trait_cols) {{
  data[[col]] <- suppressWarnings(as.numeric(as.character(data[[col]])))
}}

result <- run_correlation_analysis(
  data       = data,
  trait_cols = trait_cols,
  method     = "{method}"
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, na = "null"))
"""

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".R", delete=False
            ) as tmp:
                tmp.write(r_code)
                tmp_path = tmp.name

            proc = subprocess.run(
                ["Rscript", tmp_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode != 0:
                logger.error("R correlation failed:\n%s", proc.stderr)
                raise RuntimeError(f"R error: {proc.stderr[:600]}")

            return json.loads(proc.stdout.strip())

        except subprocess.TimeoutExpired:
            raise RuntimeError("Correlation analysis timed out (120 s exceeded)")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from R: {exc}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


# Module-level engine instance.  None until init_trait_relationships_engine()
# is called from app_genetics.startup_event.
tr_engine: Optional[TraitRelationshipsEngine] = None


def init_trait_relationships_engine() -> None:
    """
    Non-fatal initialisation called from app_genetics.startup_event.
    If the R script is missing the engine stays None and the endpoint
    returns 503; the rest of the application is unaffected.
    """
    global tr_engine
    try:
        tr_engine = TraitRelationshipsEngine(_R_SCRIPT)
        logger.info("TraitRelationshipsEngine ready")
    except FileNotFoundError as exc:
        logger.error("TraitRelationshipsEngine unavailable: %s", exc)


def _compute_significant_pairs_and_strongest(
    trait_names: List[str],
    r_matrix: List[List[Optional[float]]],
    p_matrix: List[List[Optional[float]]],
    alpha: float = 0.05
) -> tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Compute significant pairs count and the strongest SIGNIFICANT positive/negative pair.

    Only pairs with p_value <= alpha are considered for strongest_positive and
    strongest_negative.  Non-significant pairs — however large their r — are
    excluded so the interpretation layer never describes them as biologically
    meaningful.

    Returns:
        n_significant_pairs:  count of pairs with p <= alpha
        strongest_positive:   dict(trait_1, trait_2, r) for the largest r among
                              significant positive pairs, or None
        strongest_negative:   dict(trait_1, trait_2, r) for the most-negative r
                              among significant negative pairs, or None
    """
    n_significant_pairs = 0
    strongest_positive = None
    strongest_negative = None
    max_positive_r = 0.0
    max_negative_r = 0.0

    n = len(trait_names)
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            r_val = r_matrix[i][j] if r_matrix and i < len(r_matrix) and j < len(r_matrix[i]) else None
            p_val = p_matrix[i][j] if p_matrix and i < len(p_matrix) and j < len(p_matrix[i]) else None

            if r_val is None or p_val is None or p_val > alpha:
                continue

            # Pair is significant — count it and track strongest by direction
            n_significant_pairs += 1

            if r_val > 0 and r_val > max_positive_r:
                max_positive_r = r_val
                strongest_positive = {
                    "trait_1": trait_names[i],
                    "trait_2": trait_names[j],
                    "r": r_val,
                }
            elif r_val < 0 and r_val < max_negative_r:
                max_negative_r = r_val
                strongest_negative = {
                    "trait_1": trait_names[i],
                    "trait_2": trait_names[j],
                    "r": r_val,
                }

    return n_significant_pairs, strongest_positive, strongest_negative


# ============================================================================
# DATA HELPERS
# ============================================================================

def _build_wide_records(
    df: pd.DataFrame,
    genotype_col: str,
    rep_col: str,
    trait_cols: List[str],
    env_col: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Subset df to structural + trait columns and return as a list of wide-format
    observation dicts suitable for run_correlation_analysis() in R.

    Rows where every trait value is missing are dropped. Individual NaN trait
    values are kept as None (R pairwise.complete.obs handles them).

    Raises ValueError when fewer than 6 usable rows remain.
    """
    keep = (
        [genotype_col, rep_col]
        + ([env_col] if env_col else [])
        + trait_cols
    )
    subset = df[keep].copy()
    for t in trait_cols:
        subset[t] = pd.to_numeric(subset[t], errors="coerce")

    # Drop rows where every trait is NaN
    all_missing = subset[trait_cols].isna().all(axis=1)
    subset = subset[~all_missing]

    if len(subset) < 6:
        raise ValueError(
            f"Only {len(subset)} usable observation(s) after removing "
            "all-missing rows (minimum 6 required)"
        )

    records: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        rec: Dict[str, Any] = {
            "genotype": str(row[genotype_col]),
            "rep": str(row[rep_col]),
        }
        if env_col:
            rec["environment"] = str(row[env_col])
        for t in trait_cols:
            v = row[t]
            rec[t] = None if pd.isna(v) else float(v)
        records.append(rec)
    return records


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/genetics/correlation",
    response_model=CorrelationResponse,
    summary="Compute phenotypic correlations between trait pairs",
    tags=["Trait Relationships"],
)
async def compute_correlation(request: CorrelationRequest):
    """
    Compute Pearson or Spearman phenotypic correlation coefficients and
    p-values for all pairs of the specified trait columns.

    Correlations are computed on per-genotype means (averaged across reps
    and environments).  This is the standard agronomic approach: it separates
    genetic signal from replication noise and avoids inflating r with
    repeated-measures structure.

    One trait pair with insufficient data does not abort the others; it
    produces a null entry in r_matrix/p_matrix and a warning in the response.

    Requires ≥ 2 trait columns and ≥ 6 valid observations (≥ 3 unique
    genotypes after mean aggregation).
    """
    if tr_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Trait relationships engine not ready — "
                "vivasense_trait_relationships.R may be missing"
            ),
        )

    # Decode file
    try:
        file_bytes = base64.b64decode(request.base64_content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 content") from exc

    # Re-use read_file from multitrait_upload_routes to keep parsing consistent
    from multitrait_upload_routes import read_file

    try:
        df = read_file(file_bytes, request.file_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate all named columns exist in the file
    required_cols = (
        [request.genotype_column, request.rep_column]
        + request.trait_columns
        + ([request.environment_column] if request.environment_column else [])
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in file: {missing}",
        )

    env_col = request.environment_column if request.mode == "multi" else None

    try:
        records = _build_wide_records(
            df=df,
            genotype_col=request.genotype_column,
            rep_col=request.rep_column,
            trait_cols=request.trait_columns,
            env_col=env_col,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = tr_engine.run_correlation(
            records=records,
            trait_cols=request.trait_columns,
            method=request.method,
        )
    except RuntimeError as exc:
        logger.error("Correlation R error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Extract data from R result
    trait_names = result.get("trait_names", request.trait_columns)
    n_observations = result.get("n_observations", 0)
    r_matrix = result.get("r_matrix", [])
    p_matrix = result.get("p_matrix", [])
    warnings = result.get("warnings", [])
    
    # Compute significant pairs and strongest correlations for interpretation
    n_significant_pairs, strongest_positive, strongest_negative = _compute_significant_pairs_and_strongest(
        trait_names, r_matrix, p_matrix
    )
    
    # Compute risk flags (assume single environment, no GxE for correlation endpoint)
    risk_flags = _compute_risk_flags(n_observations, "genotype_mean", False)
    
    # Generate new academic-grade interpretation instead of using R script's legacy text
    interpretation_text = generate_trait_association_interpretation(
        n_traits=len(trait_names),
        n_observations=n_observations,
        n_significant_pairs=n_significant_pairs,
        strongest_positive=strongest_positive,
        strongest_negative=strongest_negative,
        risk_flags=risk_flags,
        gxe_significant=False,  # Correlation endpoint doesn't handle GxE
        environment_context="single_environment"  # Default for correlation endpoint
    )
    
    # Override the R script's interpretation with the new validated interpretation
    result["interpretation"] = interpretation_text
    result["statistical_note"] = _STATISTICAL_NOTE
    
    import json as _json
    logger.info("[correlation] response keys: %s", list(result.keys()))
    logger.info("[correlation] trait_names: %s", result.get("trait_names"))
    logger.info("[correlation] n_observations: %s", result.get("n_observations"))
    logger.info("[correlation] r_matrix shape: %s×%s",
                len(result.get("r_matrix", [])),
                len(result["r_matrix"][0]) if result.get("r_matrix") else 0)
    logger.info("[correlation] warnings: %s", result.get("warnings"))
    logger.info("[correlation] new interpretation generated: %s", bool(interpretation_text))
    
    return CorrelationResponse(**result)
