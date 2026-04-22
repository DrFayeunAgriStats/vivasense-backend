"""
VivaSense – BLUP (Best Linear Unbiased Prediction) Analysis Module

POST /analysis/blup

Predicts genotype breeding values using a mixed linear model (Henderson, 1975).
Genotype is treated as a random effect; environments (if present) are treated
as fixed effects. BLUPs shrink raw genotype means towards the population mean
in proportion to the reliability of each estimate.

Model:
  Single-environment: y = μ + g_i + ε      (g_i ~ N(0, σ²g))
  Multi-environment:  y = μ + E_j + g_i + ε (g_i ~ N(0, σ²g))

Implementation uses statsmodels.MixedLM which provides REML-based estimates.

Reference:
  Henderson, C.R. (1975). Best linear unbiased estimation and prediction
  under a selection model. Biometrics, 31(2), 423-447.
"""

import logging
import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

import dataset_cache
from module_schemas import (
    BLUPRequest,
    BLUPResponse,
    GenotypeBLUP,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_blup(
    df: pd.DataFrame,
    trait_col: str,
    genotype_col: str,
    env_col: Optional[str],
    fixed_effects: List[str],
) -> Dict[str, Any]:
    """
    Fit a mixed model and return BLUPs, SEs, and variance components.

    genotype is always the random effect group.
    env_col (if provided) is added as a fixed effect.
    Additional fixed_effects columns may also be supplied.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for BLUP analysis.") from exc

    # Work on a copy with safe column names to avoid patsy backtick issues
    df = df.copy()
    df["__trait__"] = pd.to_numeric(df[trait_col], errors="coerce")
    df = df.dropna(subset=["__trait__", genotype_col])

    if df[genotype_col].nunique() < 2:
        raise ValueError("BLUP requires at least 2 genotypes.")

    # Build formula using safe "__trait__" name
    fixed_preds: List[str] = []
    if env_col and env_col in df.columns:
        df["__env__"] = df[env_col].astype(str)
        fixed_preds.append("C(__env__)")
    for fx in fixed_effects:
        if fx in df.columns and fx != genotype_col:
            safe_name = f"__fx_{fixed_effects.index(fx)}__"
            df[safe_name] = df[fx].astype(str)
            fixed_preds.append(f"C({safe_name})")

    rhs = " + ".join(fixed_preds) if fixed_preds else "1"
    formula = f"__trait__ ~ {rhs}"

    # Fit mixed model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, df, groups=df[genotype_col].astype(str))
        try:
            fitted = model.fit(reml=True, method="lbfgs", warn_convergence=False)
        except Exception:
            try:
                fitted = model.fit(reml=True, method="bfgs", warn_convergence=False)
            except Exception as exc:
                raise RuntimeError(f"Mixed model failed to converge: {exc}") from exc

    # Extract random effects (BLUPs)
    random_effects = fitted.random_effects   # dict: group -> Series with one value
    blup_dict: Dict[str, float] = {}
    for geno, re in random_effects.items():
        if hasattr(re, "iloc"):
            blup_dict[str(geno)] = float(re.iloc[0])
        else:
            blup_dict[str(geno)] = float(re)

    # Variance components
    sigma2_g = float(fitted.cov_re.values.flat[0]) if hasattr(fitted.cov_re, "values") else float(fitted.cov_re)
    sigma2_g = max(sigma2_g, 1e-10)
    sigma2_e = float(fitted.scale)

    # Per-genotype reliability and SE using the standard BLUP formula:
    #   reliability_i = σ²g / (σ²g + σ²e / n_i)
    #   PEV_i        = σ²g * (1 - reliability_i)
    #   SE_i         = sqrt(PEV_i)
    # This matches Henderson (1975) and is consistent with plant breeding practice.
    geno_counts = df.groupby(genotype_col).size().to_dict()

    genotypes = sorted(blup_dict.keys())
    blup_rows: List[Dict[str, Any]] = []
    for geno in genotypes:
        blup_val = blup_dict[geno]
        n_i = geno_counts.get(geno, 1)
        reliability = sigma2_g / (sigma2_g + sigma2_e / n_i)
        reliability = max(0.0, min(1.0, reliability))
        pev = sigma2_g * (1.0 - reliability)
        se_val = float(np.sqrt(max(pev, 0.0)))
        blup_rows.append({
            "genotype": geno,
            "blup": blup_val,
            "se": se_val,
            "reliability": reliability,
        })

    # Sort by BLUP descending, assign rank
    blup_rows.sort(key=lambda r: r["blup"], reverse=True)
    for rank, row in enumerate(blup_rows, start=1):
        row["rank"] = rank

    # Best genotypes: top 10% by BLUP
    n_top = max(1, math.ceil(len(blup_rows) * 0.10))
    best_genotypes = [r["genotype"] for r in blup_rows[:n_top]]

    model_type = "multi-environment" if (env_col and env_col in df.columns) else "single-environment"

    return {
        "blup_rows": blup_rows,
        "best_genotypes": best_genotypes,
        "variance_components": {
            "sigma2_genotype": round(sigma2_g, 6),
            "sigma2_residual": round(sigma2_e, 6),
            "sigma2_phenotypic": round(sigma2_g + sigma2_e, 6),
        },
        "model_type": model_type,
    }


def _generate_blup_interpretation(
    trait: str,
    model_type: str,
    blup_rows: List[Dict[str, Any]],
    best_genotypes: List[str],
    variance_components: Dict[str, float],
) -> str:
    """Generate plain-English thesis-quality interpretation of BLUP results."""
    sections: List[tuple] = []

    sigma2_g = variance_components.get("sigma2_genotype", 0.0)
    sigma2_e = variance_components.get("sigma2_residual", 0.0)
    sigma2_p = variance_components.get("sigma2_phenotypic", sigma2_g + sigma2_e)
    h2 = sigma2_g / sigma2_p if sigma2_p > 0 else 0.0

    # 1. Overview
    n_genos = len(blup_rows)
    overview = (
        f"Best Linear Unbiased Prediction (BLUP) was applied to {trait} "
        f"for {n_genos} genotypes using a {model_type} mixed model "
        f"(Henderson, 1975; Piepho et al., 2008). "
        f"Genotype was treated as a random effect, allowing BLUPs to shrink "
        f"raw means toward the population mean in proportion to the reliability "
        f"of each estimate. This approach accounts for data imbalance and "
        f"is preferred for selection decisions."
    )
    sections.append(("Overview", overview))

    # 2. Variance Components
    vc_text = (
        f"The estimated genetic variance (\u03c3\u00b2g) was {sigma2_g:.4f} and "
        f"the residual variance (\u03c3\u00b2e) was {sigma2_e:.4f}, "
        f"yielding a phenotypic variance of {sigma2_p:.4f}. "
        f"Broad-sense heritability (H\u00b2 = \u03c3\u00b2g / \u03c3\u00b2p) "
        f"was estimated at {h2:.3f} ({h2 * 100:.1f}%). "
    )
    if h2 >= 0.6:
        vc_text += "High heritability indicates that genotypic differences are reliably estimated and selection will be effective."
    elif h2 >= 0.3:
        vc_text += "Moderate heritability suggests that genotypic ranking is reasonably reliable, though environmental effects are substantial."
    else:
        vc_text += "Low heritability indicates large environmental influence; BLUP ranking should be interpreted cautiously."
    sections.append(("Variance Components", vc_text))

    # 3. Top BLUPs
    top3 = blup_rows[:3]
    top_text = "The highest-ranked genotypes by BLUP value were: " + ", ".join(
        f"{r['genotype']} (BLUP = {r['blup']:+.4f}, SE = {r['se']:.4f}, "
        f"reliability = {r['reliability']:.3f})"
        for r in top3
    ) + "."
    sections.append(("Top-Ranked Genotypes", top_text))

    # 4. Reliability
    avg_rel = float(np.mean([r["reliability"] for r in blup_rows]))
    low_rel = [r["genotype"] for r in blup_rows if r["reliability"] < 0.5]
    rel_text = (
        f"Average reliability across genotypes was {avg_rel:.3f}. "
        "Reliability (r\u00b2 = 1 \u2212 PEV/\u03c3\u00b2g) reflects the fraction "
        "of genetic variance captured by the BLUP estimate: values above 0.7 are "
        "considered trustworthy for selection. "
    )
    if low_rel:
        rel_text += (
            f"Genotypes with reliability below 0.5 ({', '.join(low_rel[:5])}"
            f"{'...' if len(low_rel) > 5 else ''}) should be evaluated with caution, "
            "ideally by increasing replication or testing across additional environments."
        )
    else:
        rel_text += "All genotypes showed acceptable reliability for selection decisions."
    sections.append(("Prediction Reliability", rel_text))

    # 5. Recommended Genotypes
    rec_text = (
        f"Genotypes recommended for selection (top 10% by BLUP): "
        + ", ".join(best_genotypes) + ". "
        "Positive BLUP values indicate above-average genetic potential; "
        "negative values indicate below-average performance. "
        "These rankings account for data imbalance and are more reliable than "
        "raw mean comparisons."
    )
    sections.append(("Selection Recommendations", rec_text))

    return "\n\n".join(f"{h}\n{c}" for h, c in sections)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/blup",
    response_model=BLUPResponse,
    summary="Best Linear Unbiased Prediction (BLUP) for genotype breeding values",
)
async def analysis_blup(request: BLUPRequest) -> BLUPResponse:
    """
    Fit a mixed linear model and return BLUP breeding values for each genotype.

    Genotype is treated as random; environments (if present) and any
    additional fixed_effects columns are treated as fixed.

    Requires a dataset_token from POST /upload/dataset.
    For multi-environment BLUPs, ensure the dataset was uploaded with
    environment_column set.
    """
    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset token not found. Please upload your file via "
                "POST /upload/dataset first."
            ),
        )

    try:
        raw_bytes = __import__("base64").b64decode(ctx["base64_content"])
        df = read_file(raw_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read dataset: {exc}") from exc

    genotype_col: Optional[str] = ctx.get("genotype_column")
    env_col: Optional[str] = ctx.get("environment_column")
    trait_col: str = request.trait_column

    if not genotype_col or genotype_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Genotype column not found in dataset.",
        )
    if trait_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Trait column '{trait_col}' not found in dataset.",
        )

    # Validate fixed_effects columns exist
    missing_fx = [
        fx for fx in request.fixed_effects
        if fx not in df.columns
    ]
    if missing_fx:
        raise HTTPException(
            status_code=400,
            detail=f"Fixed effect column(s) not found: {missing_fx}",
        )

    df[trait_col] = pd.to_numeric(df[trait_col], errors="coerce")
    df = df.dropna(subset=[trait_col, genotype_col])

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="No valid numeric observations found after filtering.",
        )

    if df[genotype_col].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="BLUP requires at least 2 genotypes.",
        )

    try:
        result = _compute_blup(
            df=df,
            trait_col=trait_col,
            genotype_col=genotype_col,
            env_col=env_col if env_col and env_col in df.columns else None,
            fixed_effects=request.fixed_effects,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("BLUP computation error")
        raise HTTPException(
            status_code=503,
            detail=f"BLUP analysis failed: {exc}",
        ) from exc

    interpretation = _generate_blup_interpretation(
        trait=trait_col,
        model_type=result["model_type"],
        blup_rows=result["blup_rows"],
        best_genotypes=result["best_genotypes"],
        variance_components=result["variance_components"],
    )

    genotype_blups = [
        GenotypeBLUP(
            genotype=r["genotype"],
            blup=round(r["blup"], 6),
            se=round(r["se"], 6),
            reliability=round(r["reliability"], 4),
            rank=r["rank"],
        )
        for r in result["blup_rows"]
    ]

    return BLUPResponse(
        status="success",
        trait=trait_col,
        model_type=result["model_type"],
        genotype_blups=genotype_blups,
        best_genotypes=result["best_genotypes"],
        variance_components=result["variance_components"],
        interpretation=interpretation,
    )
