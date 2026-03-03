"""
FastAPI router for the VivaSense genetics module — V2.2 aligned.

Prefix : /analyze/genetics  (mirrors V2.2's /analyze/anova/ pattern)
Format : FormData only (file + individual Form fields, no JSON body).
Envelope: {meta, tables, plots, interpretation, strict_template, intelligence}

All endpoints:
  1. Accept multipart/form-data with file + typed Form fields
  2. Delegate to GeneticsPipeline
  3. Transform the pipeline result to the V2.2 response envelope
  4. Return JSONResponse (numpy_to_python applied once, at response time)
"""
from __future__ import annotations

import io
import json
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .compat import (
    assumption_guidance,
    cv_percent,
    df_to_records,
    fmt_p_display,
    levene_test,
    require_cols,
    round_val,
    shapiro_test,
    sig_stars,
)
from .config import GeneticsConfig, MarkerConfig, TrialDesign
from .intelligence import (
    attach_genetics_template,
    build_genetics_decision_rules,
    build_genetics_executive_insight,
    build_genetics_reviewer_radar,
)
from .pipeline import GeneticsPipeline
from .serializers import numpy_to_python

logger = logging.getLogger(__name__)

genetics_router = APIRouter(
    prefix="/analyze/genetics",
    tags=["Genetics & Breeding"],
)


# ── File loading ──────────────────────────────────────────────────────────────

async def _load_file(file: UploadFile) -> pd.DataFrame:
    """
    Read uploaded CSV or Excel file into a DataFrame.
    Raises HTTP 400 for unsupported formats (V2.2 pattern).
    """
    name = (file.filename or "").lower()
    content = await file.read()
    buf = io.BytesIO(content)

    if name.endswith(".csv"):
        try:
            return pd.read_csv(buf)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(buf)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse Excel: {exc}")

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file type '{file.filename}'. "
            "Upload a .csv, .xlsx, or .xls file."
        ),
    )


# ── Config builders from FormData ────────────────────────────────────────────

def _build_design(
    genotype_col: str,
    location_col: str,
    rep_col: str,
    trait_cols_str: str,
    season_col: str,
    design_type: str,
) -> TrialDesign:
    trait_cols = (
        [t.strip() for t in trait_cols_str.split(",") if t.strip()]
        if trait_cols_str.strip()
        else None
    )
    return TrialDesign(
        genotype_col=genotype_col,
        location_col=location_col or None,
        rep_col=rep_col or None,
        trait_cols=trait_cols,
        season_col=season_col or None,
        design_type=design_type,
    )


def _build_config(alpha: float, n_ammi_axes: int) -> GeneticsConfig:
    return GeneticsConfig(alpha=alpha, n_ammi_axes=n_ammi_axes)


def _build_marker_config(
    accession_col: str,
    similarity_metric: str,
    n_clusters: int,
) -> MarkerConfig:
    return MarkerConfig(
        accession_col=accession_col,
        similarity_metric=similarity_metric,
        n_clusters=n_clusters,
    )


# ── V2.2 envelope assembly ────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """Cast NumPy scalar / nan / inf to JSON-safe Python native."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        fv = float(v)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    return v


def _anova_table_records(anova_raw: Any) -> List[Dict[str, Any]]:
    """
    Convert the raw anova_table from MultilocationEngine/single-location
    into V2.2 ANOVA records with PR(>F) and p_display fields.

    Engines return: {"columns": ["Source","df","SS","MS","F","p_value","significance"],
                     "rows": [[...], ...]}
    """
    from .compat import add_p_display_to_anova
    if not anova_raw:
        return []

    # ── columns+rows dict (engine format) ─────────────────────────────
    if isinstance(anova_raw, dict) and "columns" in anova_raw and "rows" in anova_raw:
        cols = anova_raw["columns"]
        recs: List[Dict[str, Any]] = []
        for row in anova_raw["rows"]:
            rec: Dict[str, Any] = {}
            for col, val in zip(cols, row):
                # normalise p_value column name to V2.2 convention
                key = "PR(>F)" if col.lower() in ("p_value", "p-value", "pr(>f)") else col
                # normalise "source" → "source"
                key = "source" if key == "Source" else key
                rec[key] = _safe(val)
            recs.append(rec)
        return add_p_display_to_anova(recs)

    # ── list of dicts (alternative format) ────────────────────────────
    if isinstance(anova_raw, list):
        recs = [{k: _safe(v) for k, v in r.items()} for r in anova_raw]
        return add_p_display_to_anova(recs)

    return []


def _vc_records(vc_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten one trait's variance-component result dict into a single flat row.
    Handles the nested engine format:
      vc_raw["components"]["sigma2_g"]["value"], vc_raw["heritability"]["H2_broad"]["value"], etc.
    """
    if not vc_raw:
        return []

    def _get_nested(d: dict, *path: str) -> Any:
        """Walk nested keys, return None if any step is missing."""
        cur = d
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    comps = vc_raw.get("components", {})
    hered = vc_raw.get("heritability", {})
    ga    = vc_raw.get("genetic_advance", {})
    cvs   = vc_raw.get("coefficients_of_variation", {})

    row: Dict[str, Any] = {
        "grand_mean":   round_val(_safe(vc_raw.get("grand_mean"))),
        "n_genotypes":  vc_raw.get("n_genotypes"),
        "n_locations":  vc_raw.get("n_locations"),
        # Variance components
        "sigma2_g":     round_val(_safe(_get_nested(comps, "sigma2_g",  "value"))),
        "sigma2_e":     round_val(_safe(_get_nested(comps, "sigma2_e",  "value"))),
        "sigma2_gl":    round_val(_safe(_get_nested(comps, "sigma2_gl", "value"))),
        "sigma2_p":     round_val(_safe(_get_nested(comps, "sigma2_p",  "value"))),
        # Heritability
        "H2_broad":     round_val(_safe(_get_nested(hered, "H2_broad",  "value"))),
        "H2_broad_pct": round_val(_safe(_get_nested(hered, "H2_broad",  "percentage"))),
        # Genetic advance
        "GA":           round_val(_safe(_get_nested(ga, "GA",        "value"))),
        "GA_percent":   round_val(_safe(_get_nested(ga, "GA_percent","value"))),
        # Coefficients of variation
        "GCV":          round_val(_safe(_get_nested(cvs, "GCV", "value"))),
        "PCV":          round_val(_safe(_get_nested(cvs, "PCV", "value"))),
        "ECV":          round_val(_safe(_get_nested(cvs, "ECV", "value"))),
    }
    # Strip None-valued entries for cleaner output
    row = {k: v for k, v in row.items() if v is not None}
    return [row] if row else []


def _stability_records(stab_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten Eberhart-Russell genotype list into records."""
    genotype_stability = stab_raw.get("genotype_stability", [])
    if not genotype_stability:
        return []
    records = []
    for g in genotype_stability:
        rec = {k: round_val(_safe(v)) if isinstance(v, (int, float, np.number)) else v
               for k, v in g.items()}
        records.append(rec)
    return records


def _means_records(
    stab_raw: Dict[str, Any],
    genotype_col: str,
) -> List[Dict[str, Any]]:
    """
    Pull mean-per-genotype from stability results (genotype_stability list),
    which reliably contains grand_mean for each entry.
    Falls back to an empty list for single-environment trials.
    """
    genotype_stability = stab_raw.get("genotype_stability", [])
    if not genotype_stability:
        return []
    records = []
    for g in genotype_stability:
        gid  = g.get("genotype") or g.get("Genotype") or "?"
        mean = _safe(g.get("grand_mean") or g.get("mean"))
        if mean is not None:
            records.append({genotype_col: gid, "mean": round_val(mean)})
    records.sort(key=lambda r: r.get("mean", 0), reverse=True)
    return records


def _extract_residuals(anova_raw: Dict[str, Any]) -> np.ndarray:
    """Best-effort extraction of residuals from a combined ANOVA result."""
    residuals = anova_raw.get("residuals")
    if residuals is not None:
        return np.asarray(residuals, dtype=float)
    return np.array([])


def _build_trial_envelope(
    raw: Dict[str, Any],
    trait: str,
    design: TrialDesign,
    alpha: float,
) -> Dict[str, Any]:
    """
    Transform GeneticsPipeline.run_trial_analysis() output into the V2.2
    response envelope for a single trait.
    """
    meta_raw   = raw.get("metadata", {})
    n_locs     = int(meta_raw.get("n_locations", 1))
    n_genos    = int(meta_raw.get("n_genotypes", 0))
    n_reps     = 1  # default; engines expose this if available
    mode       = meta_raw.get("mode", "single_environment")

    anova_raw  = (raw.get("anova_tables") or {}).get(trait, {})
    vc_raw     = (raw.get("variance_components") or {}).get(trait, {})
    stab_raw   = (raw.get("stability") or {}).get(trait, {})
    ammi_raw   = (raw.get("ammi") or {}).get(trait, {})
    gge_raw    = (raw.get("gge") or {}).get(trait, {})

    # ANOVA table records
    anova_recs  = _anova_table_records(anova_raw)
    vc_recs     = _vc_records(vc_raw)
    stab_recs   = _stability_records(stab_raw)
    means_recs  = _means_records(stab_raw, design.genotype_col)

    # Assumption tests (on residuals from the combined ANOVA)
    residuals   = _extract_residuals(anova_raw)
    sh_rec      = shapiro_test(residuals) if len(residuals) >= 3 else {
        "test": "Shapiro-Wilk", "statistic": None, "p_value": None,
        "p_display": "N/A", "passed": None,
        "interpretation": "Residuals not available for normality test.",
    }
    # Levene on genotype groups from cell_matrix — approximate from means records
    lv_rec = {
        "test": "Levene", "statistic": None, "p_value": None,
        "p_display": "N/A", "passed": None,
        "interpretation": "Levene test requires raw data per-group.",
    }

    guidance = assumption_guidance(sh_rec, lv_rec, alpha)

    # p-map for intelligence blocks
    p_map: Dict[str, Optional[float]] = {}
    for rec in anova_recs:
        src = rec.get("source", "")
        p   = rec.get("PR(>F)")
        if src and p is not None:
            p_map[src] = float(p)

    # Heritability + GA% for intelligence (nested inside vc_recs row)
    h2_f: Optional[float] = None
    ga_pct_f: Optional[float] = None
    if vc_recs:
        row0 = vc_recs[0]
        raw_h2  = row0.get("H2_broad")
        raw_ga  = row0.get("GA_percent")
        h2_f    = float(raw_h2)  if raw_h2  is not None else None
        ga_pct_f = float(raw_ga) if raw_ga  is not None else None

    # CV from means
    if means_recs:
        cv_vals = [r["mean"] for r in means_recs if r.get("mean") is not None]
        cv_series = pd.Series(cv_vals)
        cv_val = cv_percent(cv_series) if len(cv_series) > 1 else None
    else:
        cv_val = None

    # Best genotype
    best_geno = means_recs[0].get(design.genotype_col) if means_recs else None

    # Stable genotypes
    stable_genos = [
        str(r.get("Genotype") or r.get("genotype", ""))
        for r in stab_recs
        if str(r.get("classification", "")).startswith("stable")
    ]

    # Intelligence blocks
    exec_insight  = build_genetics_executive_insight(
        p_map=p_map, trait=trait, alpha=alpha,
        h2=h2_f, ga_percent=ga_pct_f,
        n_locations=n_locs, n_genotypes=n_genos,
        cv=cv_val, best_genotype=best_geno,
    )
    reviewer_radar = build_genetics_reviewer_radar(
        shapiro_rec=sh_rec, levene_rec=lv_rec,
        p_map=p_map, cv=cv_val,
        n_locations=n_locs, n_reps=n_reps,
        h2=h2_f, alpha=alpha,
    )
    decision_rules = build_genetics_decision_rules(
        means_table=means_recs,
        genotype_col=design.genotype_col,
        trait=trait, alpha=alpha,
        h2=h2_f, stable_genotypes=stable_genos or None,
    )

    # Plain-English interpretation
    from .compat import interpret_genetics
    interpretation = interpret_genetics(
        analysis_type=f"Multi-environment ANOVA ({mode})",
        p_map=p_map, trait=trait, alpha=alpha,
        extra={"heritability": h2_f, "ga_percent": ga_pct_f},
    )

    # Assemble envelope
    envelope: Dict[str, Any] = {
        "meta": {
            "design":       f"RCBD multi-environment ({mode})",
            "trait":        trait,
            "n_genotypes":  n_genos,
            "n_locations":  n_locs,
            "analysis_id":  raw.get("analysis_id"),
            "timestamp":    raw.get("timestamp"),
            "filename":     meta_raw.get("filename"),
            "genotype_col": design.genotype_col,
            "location_col": design.location_col,
            "rep_col":      design.rep_col,
            "mode":         mode,
            "warnings":     raw.get("warnings", []),
        },
        "tables": {
            "combined_anova":       anova_recs,
            "variance_components":  vc_recs,
            "genotype_means":       means_recs,
            "stability":            stab_recs,
            "assumptions":          [sh_rec, lv_rec],
            "assumption_guidance":  guidance,
        },
        "plots":          (raw.get("plots") or {}),
        "interpretation": interpretation,
        "strict_template": {},
        "intelligence": {
            "executive_insight":  exec_insight,
            "reviewer_radar":     reviewer_radar,
            "decision_rules":     decision_rules,
            "assumptions_verdict": guidance["overall"],
        },
    }

    # Add AMMI / GGE tables if present
    if ammi_raw:
        ipca = ammi_raw.get("ipca_scores", [])
        if isinstance(ipca, list):
            envelope["tables"]["ammi_ipca"] = ipca
        expl = ammi_raw.get("explained_variance", [])
        if isinstance(expl, list):
            envelope["tables"]["ammi_explained_variance"] = [
                {k: round_val(_safe(v)) if isinstance(v, (int, float, np.number)) else v
                 for k, v in row.items()}
                for row in expl
            ]
    if gge_raw:
        wwhere = gge_raw.get("which_won_where", [])
        if isinstance(wwhere, list):
            envelope["tables"]["gge_which_won_where"] = wwhere

    # Correlations / path / selection index (multi-trait)
    if raw.get("correlations"):
        envelope["tables"]["correlations"] = raw["correlations"]
    if raw.get("path_analysis"):
        envelope["tables"]["path_analysis"] = raw["path_analysis"]
    if raw.get("selection_index"):
        envelope["tables"]["selection_index"] = raw["selection_index"]
    if raw.get("multivariate"):
        envelope["tables"]["multivariate"] = raw["multivariate"]

    return attach_genetics_template(envelope, trait, alpha)


def _build_marker_envelope(
    raw: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """
    Transform run_marker_analysis() output into the V2.2 response envelope.
    The pipeline puts all marker engine output under raw["diversity"].
    """
    meta_raw  = raw.get("metadata", {})
    div       = raw.get("diversity", {})   # marker engine analyze() result
    n_acc     = div.get("n_accessions") or meta_raw.get("n_accessions") or "?"
    n_markers = div.get("n_markers")    or meta_raw.get("n_markers")    or "?"

    # Summary diversity indices → flat records
    summary = div.get("summary_diversity", {})
    div_records: List[Dict[str, Any]] = [
        {
            "metric": metric,
            "value":  round_val(_safe(val.get("value") if isinstance(val, dict) else val)),
            "description": (val.get("formula_description", "") if isinstance(val, dict) else ""),
        }
        for metric, val in summary.items()
    ]

    # Per-locus table
    per_locus = div.get("per_locus_diversity", {})
    locus_records: List[Dict[str, Any]] = []
    for locus, stats in per_locus.items():
        rec = {"locus": locus}
        for k, v in stats.items():
            rec[k] = round_val(_safe(v)) if isinstance(v, (int, float, np.number)) else v
        locus_records.append(rec)

    # Cluster membership table
    cluster_groups = div.get("cluster_groups", {})
    assignments    = cluster_groups.get("assignments", {})
    cluster_recs   = [
        {"accession": acc, "cluster": cl}
        for acc, cl in assignments.items()
    ]

    # PCA variance (from pca_molecular)
    pca_mol = div.get("pca_molecular", {})
    pca_var = pca_mol.get("explained_variance_ratio", [])
    pca_var_recs = [
        {"PC": f"PC{i+1}", "explained_variance_ratio": round_val(_safe(v))}
        for i, v in enumerate(pca_var)
    ]

    # Similarity matrix (Jaccard)
    sim_matrix = div.get("similarity_matrix", {})

    envelope: Dict[str, Any] = {
        "meta": {
            "design":       "Molecular marker diversity analysis",
            "analysis_id":  raw.get("analysis_id"),
            "timestamp":    raw.get("timestamp"),
            "filename":     meta_raw.get("filename"),
            "n_accessions": n_acc,
            "n_markers":    n_markers,
            "warnings":     raw.get("warnings", []),
        },
        "tables": {
            "diversity_indices":   div_records,
            "per_locus_diversity": locus_records,
            "cluster_membership":  cluster_recs,
            "pca_variance":        pca_var_recs,
        },
        "plots":          (raw.get("plots") or {}),
        "interpretation": (
            f"Molecular diversity analysis of {n_acc} accession(s) across "
            f"{n_markers} marker locus/loci completed using UPGMA clustering."
        ),
        "strict_template": {},
        "intelligence": {
            "executive_insight": (
                f"Molecular diversity analysis of {n_acc} accession(s) across "
                f"{n_markers} marker locus/loci completed.  "
                f"{cluster_groups.get('n_clusters', '?')} clusters identified via UPGMA."
            ),
            "reviewer_radar": [
                "Report polymorphic information content (PIC) per locus — required by most journals.",
                "Clarify similarity metric choice (Jaccard vs. Dice) based on marker system "
                "(SSR, RAPD, AFLP, DArT).",
                "State whether missing marker data were imputed or excluded from analysis.",
            ],
            "decision_rules": [
                "Select crossing parents from distinct UPGMA clusters to maximise genetic diversity.",
                "Accessions with highest PIC values are most informative for marker-assisted selection.",
                f"All {n_acc} accessions have been assigned to clusters — review dendrogram "
                "before finalising crossing scheme.",
            ],
            "assumptions_verdict": (
                "Binary marker data: parametric assumptions (normality, homogeneity) do not apply. "
                "Results are based on non-parametric similarity coefficients."
            ),
        },
    }

    if sim_matrix:
        envelope["tables"]["jaccard_similarity"] = sim_matrix
    dice_matrix = div.get("dice_matrix", {})
    if dice_matrix:
        envelope["tables"]["dice_similarity"] = dice_matrix

    return envelope


def _json_response(data: Any) -> JSONResponse:
    return JSONResponse(content=numpy_to_python(data))


# ── Health ────────────────────────────────────────────────────────────────────

@genetics_router.get("/health")
async def genetics_health():
    """Health check for the genetics module."""
    return {"status": "healthy", "module": "genetics", "version": "2.2.0"}


# ── Full trial analysis ───────────────────────────────────────────────────────

@genetics_router.post("/trial")
async def analyze_genetics_trial(
    file: UploadFile = File(
        ...,
        description="CSV or Excel breeding trial file",
    ),
    genotype_col: str = Form(
        "Genotype",
        description="Column name for genotype / variety identifier",
    ),
    location_col: str = Form(
        "Location",
        description="Column name for location / environment (leave blank for single-environment)",
    ),
    rep_col: str = Form(
        "Rep",
        description="Column name for replicate / block",
    ),
    trait_cols: str = Form(
        "",
        description="Comma-separated trait column names; leave blank for auto-detection",
    ),
    season_col: str = Form(
        "",
        description="Optional season column — combined with location to form environment ID",
    ),
    design_type: str = Form(
        "RCBD",
        description="Experimental design: RCBD | CRD | Alpha-lattice",
    ),
    alpha: float = Form(
        0.05,
        description="Significance level (default 0.05)",
    ),
    n_ammi_axes: int = Form(
        2,
        description="Number of AMMI axes to retain (default 2)",
    ),
):
    """
    Upload a CSV or Excel breeding trial file and run the full genetics pipeline.

    Returns variance components, AMMI, GGE, stability (Eberhart-Russell + ASV),
    phenotypic/genotypic correlations, path analysis, selection index, PCA and
    clustering — all in the V2.2 {meta, tables, plots, interpretation,
    strict_template, intelligence} envelope.

    **Minimum CSV columns**: Genotype, Rep, + at least one numeric trait column.
    Add a Location column for multi-environment analysis.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(
        genotype_col=genotype_col,
        location_col=location_col,
        rep_col=rep_col,
        trait_cols_str=trait_cols,
        season_col=season_col,
        design_type=design_type,
    )
    cfg = _build_config(alpha, n_ammi_axes)

    pipeline = GeneticsPipeline(config=cfg, design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Trial analysis failed.",
                "errors":  raw.get("errors", []),
                "warnings": raw.get("warnings", []),
            },
        )

    # Determine primary trait for single-trait envelope
    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary_trait = trait_list[0] if trait_list else "trait"

    envelope = _build_trial_envelope(raw, primary_trait, design, alpha)

    # If multiple traits: embed per-trait sub-envelopes
    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                per_trait[t] = _build_trial_envelope(raw, t, design, alpha)
            except Exception as exc:
                logger.warning("Could not build envelope for trait %s: %s", t, exc)
        envelope["per_trait"] = per_trait

    return _json_response(envelope)


# ── Variance components only ──────────────────────────────────────────────────

@genetics_router.post("/variance-components")
async def genetics_variance_components(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    alpha: float = Form(0.05),
):
    """
    Variance components only (σ²g, σ²e, σ²gl, H², GA, GCV, PCV).
    Requires ≥2 locations for σ²gl; falls back to single-environment estimates.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(genotype_col, location_col, rep_col, trait_cols, "", "RCBD")
    cfg    = _build_config(alpha, 2)

    pipeline = GeneticsPipeline(config=cfg, design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = trait_list[0] if trait_list else "trait"
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    # Slim down to VC-relevant keys
    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "variance_components": envelope["tables"]["variance_components"],
            "combined_anova":      envelope["tables"]["combined_anova"],
            "assumption_guidance": envelope["tables"]["assumption_guidance"],
        },
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }
    return _json_response(slim)


# ── Stability only ────────────────────────────────────────────────────────────

@genetics_router.post("/stability")
async def genetics_stability(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    alpha: float = Form(0.05),
):
    """
    Stability parameters only (Eberhart-Russell bi, S²di, ASV).
    Requires ≥2 locations.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col, location_col])

    design = _build_design(genotype_col, location_col, rep_col, trait_cols, "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = trait_list[0] if trait_list else "trait"
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "stability": envelope["tables"]["stability"],
            "genotype_means": envelope["tables"]["genotype_means"],
        },
        "plots":           {k: v for k, v in envelope["plots"].items()
                            if any(x in k for x in ["stability", "bi"])},
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }
    return _json_response(slim)


# ── AMMI only ─────────────────────────────────────────────────────────────────

@genetics_router.post("/ammi")
async def genetics_ammi(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    n_ammi_axes: int = Form(2),
    alpha: float = Form(0.05),
):
    """
    AMMI model (ANOVA partition + IPCA scores + biplot data).
    Requires ≥3 locations.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col, location_col])

    design = _build_design(genotype_col, location_col, rep_col, trait_cols, "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, n_ammi_axes), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = trait_list[0] if trait_list else "trait"
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "ammi_ipca":              envelope["tables"].get("ammi_ipca", []),
            "ammi_explained_variance": envelope["tables"].get("ammi_explained_variance", []),
            "combined_anova":         envelope["tables"]["combined_anova"],
        },
        "plots":           {k: v for k, v in envelope["plots"].items() if "ammi" in k},
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }
    return _json_response(slim)


# ── GGE only ──────────────────────────────────────────────────────────────────

@genetics_router.post("/gge")
async def genetics_gge(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    alpha: float = Form(0.05),
):
    """
    GGE biplot (which-won-where, ideal genotype, mega-environments).
    Requires ≥2 locations.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col, location_col])

    design = _build_design(genotype_col, location_col, rep_col, trait_cols, "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = trait_list[0] if trait_list else "trait"
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "gge_which_won_where": envelope["tables"].get("gge_which_won_where", []),
            "genotype_means":      envelope["tables"]["genotype_means"],
        },
        "plots":           {k: v for k, v in envelope["plots"].items() if "gge" in k},
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }
    return _json_response(slim)


# ── Correlations only ─────────────────────────────────────────────────────────

@genetics_router.post("/correlations")
async def genetics_correlations(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    alpha: float = Form(0.05),
):
    """
    Phenotypic + genotypic correlations, path analysis, Smith-Hazel selection index.
    Requires ≥2 trait columns.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(genotype_col, location_col, rep_col, trait_cols, "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = trait_list[0] if trait_list else "trait"
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "correlations":   envelope["tables"].get("correlations", {}),
            "path_analysis":  envelope["tables"].get("path_analysis", {}),
            "selection_index": envelope["tables"].get("selection_index", {}),
        },
        "plots":           {k: v for k, v in envelope["plots"].items()
                            if any(x in k for x in ["heatmap", "path", "corr"])},
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }
    return _json_response(slim)


# ── Multivariate only ─────────────────────────────────────────────────────────

@genetics_router.post("/multivariate")
async def genetics_multivariate(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    alpha: float = Form(0.05),
):
    """PCA + hierarchical clustering (Ward) + k-means on genotype means."""
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(genotype_col, location_col, rep_col, trait_cols, "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = trait_list[0] if trait_list else "trait"
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "multivariate": envelope["tables"].get("multivariate", {}),
        },
        "plots":           {k: v for k, v in envelope["plots"].items()
                            if any(x in k for x in ["pca", "scree", "dendrogram", "cluster"])},
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }
    return _json_response(slim)


# ── Molecular markers ─────────────────────────────────────────────────────────

@genetics_router.post("/markers")
async def genetics_markers(
    file: UploadFile = File(
        ...,
        description="CSV or Excel binary marker matrix (rows = accessions, cols = markers, values 0/1)",
    ),
    accession_col: str = Form(
        "Accession",
        description="Column name for accession / genotype IDs",
    ),
    similarity_metric: str = Form(
        "jaccard",
        description="Similarity metric: jaccard | dice | both",
    ),
    n_clusters: int = Form(
        3,
        description="Number of k-means clusters",
    ),
    alpha: float = Form(0.05),
):
    """
    Upload a binary marker matrix and run molecular diversity analysis.

    Returns: Jaccard/Dice similarity, Shannon H′, Simpson D, PIC, Nei gene
    diversity, UPGMA dendrogram, k-means clusters, PCA on binary matrix.
    """
    df = await _load_file(file)
    require_cols(df, [accession_col])

    mk_config = _build_marker_config(accession_col, similarity_metric, n_clusters)
    pipeline = GeneticsPipeline(marker_config=mk_config)
    raw = pipeline.run_marker_analysis(df, filename=file.filename or "markers")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Marker analysis failed.",
                "errors":  raw.get("errors", []),
                "warnings": raw.get("warnings", []),
            },
        )

    return _json_response(_build_marker_envelope(raw, alpha))
