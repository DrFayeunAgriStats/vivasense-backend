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
from .engines.table_generator import build_html_tables
from .engines.figure_generator import build_publication_figures

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


def _resolve_trait_cols(trait_cols: str, traits: str) -> str:
    """
    Normalise trait column input from any of the three formats Lovable may send:

      1. Comma-separated string:    "Days_to_Flowering,Plant_Height_cm"
      2. JSON array string:         '["Days_to_Flowering","Plant_Height_cm"]'
      3. Multiple form fields:      traits=Days_to_Flowering&traits=Plant_Height_cm
         (FastAPI gives this as the last value for a str field; use List[str] for real
          multi-value, but we handle that with the comma-join approach below)

    Prefers `traits` over `trait_cols` if `traits` is non-empty.
    Strips JSON array brackets and quotes. Returns a comma-separated string
    suitable for _build_design() / _build_config().
    """
    raw = (traits.strip() or trait_cols.strip())
    if not raw:
        return ""
    # JSON array: ["a","b"]  or  ['a','b']
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return ",".join(str(t).strip() for t in parsed if str(t).strip())
        except (json.JSONDecodeError, TypeError):
            pass
    return raw


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

    # Add publication-ready JSON tables (for programmatic use)
    try:
        envelope["publication_tables"] = build_genetics_publication_tables(
            envelope, trait, design, alpha
        )
    except Exception as _pub_exc:
        logger.warning("Could not build genetics publication tables: %s", _pub_exc)
        envelope["publication_tables"] = {}

    # Add publication-ready HTML tables (copy-paste ready for Word/LaTeX)
    try:
        envelope["html_tables"] = build_html_tables(
            envelope, trait,
            genotype_col=design.genotype_col,
        )
    except Exception as _ht_exc:
        logger.warning("Could not build HTML tables: %s", _ht_exc)
        envelope["html_tables"] = []

    # Add publication-quality figures (base64 PNG, 300 DPI)
    try:
        pub_figs = build_publication_figures(
            envelope, trait, genotype_col=design.genotype_col
        )
        # Merge into existing plots dict and also expose as named list
        if not isinstance(envelope["plots"], dict):
            envelope["plots"] = {}
        for fig in pub_figs:
            key = fig["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            envelope["plots"][key] = fig["image_base64"]
        envelope["publication_figures"] = pub_figs
    except Exception as _fig_exc:
        logger.warning("Could not build publication figures: %s", _fig_exc)
        envelope["publication_figures"] = []

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


# ── Publication tables for genetics ───────────────────────────────────────────

def _gsig(p) -> str:
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def _gfmt(v, decimals: int = 4) -> str:
    import math as _math
    if v is None:
        return "—"
    try:
        fv = float(v)
        return "—" if (_math.isnan(fv) or _math.isinf(fv)) else f"{fv:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _generate_genetics_backend_interp(envelope: Dict[str, Any], trait: str) -> str:
    """Concise 2-paragraph plain-language summary for genetics results."""
    meta  = envelope.get("meta", {})
    vc    = (envelope.get("tables", {}).get("variance_components") or [{}])[0]
    means = envelope.get("tables", {}).get("genotype_means", [])

    n_genos = meta.get("n_genotypes", "?")
    n_locs  = meta.get("n_locations", "?")
    h2      = vc.get("H2_broad")
    ga_pct  = vc.get("GA_percent")

    h2_pct = float(h2) * 100 if h2 and float(str(h2)) <= 1 else float(h2 or 0)
    h2_cat = ("very high" if h2_pct >= 70 else "high" if h2_pct >= 60 else
               "moderate" if h2_pct >= 30 else "low")
    ga_cat = ("excellent" if ga_pct and float(ga_pct) > 10 else
               "good" if ga_pct and float(ga_pct) >= 5 else "low")

    para1 = (
        f"Variance component analysis of {trait} across {n_locs} location(s) "
        f"involving {n_genos} genotype(s) revealed {h2_cat} broad-sense heritability "
        f"(H\u00b2 = {_gfmt(h2_pct, 2)}%), indicating that genetic factors "
        f"{'strongly' if h2_pct >= 60 else 'moderately'} control the expression of this trait. "
    )
    if ga_pct:
        para1 += (
            f"Predicted genetic advance under 5% selection intensity was {_gfmt(ga_pct, 2)}% "
            f"of the population mean, reflecting a {ga_cat} expected response to selection. "
        )

    para2 = ""
    if means:
        best  = means[0]
        worst = means[-1]
        gid_b = best.get("Genotype") or best.get("genotype") or best.get(next(iter(best), ""), "—")
        gid_w = worst.get("Genotype") or worst.get("genotype") or worst.get(next(iter(worst), ""), "—")
        m_b   = best.get("mean")
        m_w   = worst.get("mean")
        para2 = (
            f"{gid_b} ranked highest for {trait} (mean = {_gfmt(m_b, 2)}), "
            f"while {gid_w} ranked lowest (mean = {_gfmt(m_w, 2)}). "
            "High heritability combined with significant genotype differences supports "
            "the use of direct phenotypic selection in early breeding generations. "
            f"It is recommended to advance {gid_b} to replicated multi-location trials "
            "for validation before commercial release."
        )
    return para1 + para2


def _generate_genetics_academic_interp(
    envelope: Dict[str, Any], trait: str, design: TrialDesign, alpha: float
) -> Dict[str, str]:
    """Dr. Fayeun's 8-section structured academic interpretation for genetics."""
    meta   = envelope.get("meta", {})
    tables = envelope.get("tables", {})
    vc     = (tables.get("variance_components") or [{}])[0]
    means  = tables.get("genotype_means", [])
    stab   = tables.get("stability", [])
    anova  = tables.get("combined_anova", [])

    n_genos = meta.get("n_genotypes", "?")
    n_locs  = meta.get("n_locations", "?")
    mode    = meta.get("mode", "multi_environment")

    h2      = vc.get("H2_broad")
    h2_pct  = float(h2) * 100 if h2 and float(str(h2)) <= 1 else float(h2 or 0)
    ga      = vc.get("GA")
    ga_pct  = vc.get("GA_percent")
    s2g     = vc.get("sigma2_g")
    s2e     = vc.get("sigma2_e")

    best_geno  = ((means[0].get("Genotype") or means[0].get("genotype", "—")) if means else "—")
    worst_geno = ((means[-1].get("Genotype") or means[-1].get("genotype", "—")) if means else "—")

    geno_sig = loc_sig = gl_sig = False
    for rec in anova:
        src = str(rec.get("source", "")).lower()
        p   = rec.get("PR(>F)") or rec.get("p_value")
        if p is None:
            continue
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if "genotype" in src:
            geno_sig = pf < alpha
        elif "location" in src or "environ" in src:
            loc_sig = pf < alpha
        elif "×" in src or "x" in src or ("g" in src and "l" in src):
            gl_sig = pf < alpha

    sec1 = (
        f"This study employed a Randomized Complete Block Design (RCBD) multi-environment trial "
        f"({mode}) to evaluate {n_genos} genotype(s) for the trait '{trait}' across {n_locs} "
        f"location(s). The primary objectives were to: (i) estimate variance components and "
        f"broad-sense heritability; (ii) assess genotype \u00d7 location (G\u00d7L) interaction "
        f"effects; and (iii) identify superior, stable genotypes for potential variety release. "
        f"The multi-environment approach provides a rigorous basis for understanding genotype "
        f"adaptability and the relative influence of genetic versus environmental factors."
    )

    sec2 = (
        f"The combined ANOVA revealed "
        f"{'highly significant' if geno_sig else 'non-significant'} genotypic variation for "
        f"{trait} (p {'< 0.001' if geno_sig else '> 0.05'}), indicating that "
        f"{'substantial genetic diversity exists among the tested genotypes' if geno_sig else 'limited genetic differentiation was observed'}. "
        f"Location effects were {'significant' if loc_sig else 'non-significant'}, "
        f"{'reflecting the diverse agro-ecological conditions across sites' if loc_sig else 'suggesting environmental uniformity across sites'}. "
    )
    if gl_sig:
        sec2 += (
            "The significant G\u00d7L interaction indicates that genotype rankings were "
            "inconsistent across environments, necessitating site-specific variety recommendations "
            "and stability analysis to identify broadly adapted genotypes."
        )
    else:
        sec2 += (
            "The non-significant G\u00d7L interaction supports the identification of broadly "
            "adapted genotypes that perform consistently across all tested environments."
        )

    h2_label = ("very high" if h2_pct >= 70 else "high" if h2_pct >= 60 else
                 "moderate" if h2_pct >= 30 else "low")
    sec3 = (
        f"The estimated broad-sense heritability (H\u00b2 = {_gfmt(h2_pct, 2)}%) indicates "
        f"{h2_label} genetic control over the expression of {trait}. "
        f"Genetic variance (σ\u00b2g = {_gfmt(s2g, 4)}) "
        f"{'dominated' if s2g and s2e and float(str(s2g)) > float(str(s2e)) else 'was comparable to'} "
        f"environmental variance (σ\u00b2e = {_gfmt(s2e, 4)}). "
    )
    if ga_pct:
        ga_flt = float(str(ga_pct))
        ga_cat = ("excellent (>10% of mean)" if ga_flt > 10 else
                  "good (5\u201310% of mean)" if ga_flt >= 5 else "low (<5% of mean)")
        sec3 += (
            f"The predicted genetic advance under 5% selection intensity was "
            f"{_gfmt(ga, 4)} units ({_gfmt(ga_pct, 2)}% of the population mean), "
            f"classified as {ga_cat}. "
            f"This {'supports effective improvement through direct phenotypic selection' if ga_flt >= 5 else 'suggests marker-assisted or recurrent selection may be necessary to accelerate breeding progress'}."
        )

    stable_gs = [r.get("Genotype") or r.get("genotype", "") for r in stab
                 if str(r.get("classification", "")).startswith("stable")] if stab else []
    sec4 = ""
    if stab:
        sec4 = (
            "Stability analysis was conducted using the Eberhart and Russell (1966) model. "
            "Genotypes with regression coefficients (bi) close to unity (bi \u2248 1.0) and "
            "minimal deviation variance (S\u00b2di \u2248 0) were classified as stable. "
        )
        sec4 += (
            f"The following genotype(s) were identified as broadly stable: "
            f"{', '.join(str(g) for g in stable_gs[:4])}. "
            "These are recommended for wide-adaptation release."
            if stable_gs else
            "No genotypes were classified as broadly stable. "
            "Location-specific recommendations are advised."
        )
    else:
        sec4 = (
            "Stability analysis requires data from at least 2 locations. "
            f"With {n_locs} location(s) in this dataset, "
            f"{'stability parameters were estimated' if str(n_locs).isdigit() and int(str(n_locs)) >= 2 else 'stability analysis could not be performed'}."
        )

    sec5 = (
        f"The {h2_label} heritability observed for {trait} has important implications for the "
        f"breeding programme. "
        f"{'High heritability (H\u00b2 \u2265 60%) indicates that a large proportion of phenotypic variation is attributable to genetic differences, making direct phenotypic selection highly effective.' if h2_pct >= 60 else 'Moderate heritability suggests that environmental factors contribute substantially to phenotypic variation, and multi-environment testing is essential before final selection decisions.'} "
        f"The superior performance of {best_geno} for {trait} may reflect favourable allele "
        f"combinations at loci controlling the physiological pathways underlying this trait."
    )

    sec6 = (
        f"Based on the analysis of {trait}, the following breeding recommendations are proposed:\n"
        f"1. ADVANCE {best_geno}: Ranked highest — promote to replicated multi-location trials "
        f"or use as a parent in crossing programmes.\n"
        f"2. DISCARD {worst_geno}: Lowest performer — remove from the breeding pipeline.\n"
        f"3. Selection Strategy: Given the {h2_label} heritability, "
        f"{'direct phenotypic selection in F\u2082\u2013F\u2083 generations is recommended' if h2_pct >= 60 else 'progeny testing across multiple environments before selection is recommended'}.\n"
        f"4. Multi-environment Validation: Validate across at least 5 locations over 2\u20133 seasons before variety release."
    )

    sec7 = (
        f"This study was conducted across {n_locs} location(s), which "
        f"{'provides a reasonable basis for multi-environment inferences' if str(n_locs).isdigit() and int(str(n_locs)) >= 3 else 'limits generalisability; at least 3\u20135 locations are recommended for variety release'}. "
        "Future studies should integrate genomic data to enable marker-assisted selection (MAS) "
        "and genomic selection (GS), which can substantially accelerate genetic gains. "
        "Economic modelling incorporating input costs and market prices should also be conducted "
        "before recommending specific varieties to farmers."
    )

    sec8 = (
        f"In conclusion, the multi-environment trial demonstrated "
        f"{'significant' if geno_sig else 'limited'} genetic variation for {trait} among the "
        f"tested genotypes, with broad-sense heritability of {_gfmt(h2_pct, 2)}%. "
        f"The genotype {best_geno} consistently performed best and is recommended for advancement. "
        f"{'The high heritability and satisfactory genetic advance support direct phenotypic selection as an efficient strategy for this trait.' if h2_pct >= 60 and ga_pct and float(str(ga_pct)) >= 5 else 'Marker-assisted or recurrent selection strategies may be required to achieve meaningful genetic gain.'} "
        "These findings contribute to evidence-based decision-making in plant breeding and "
        "provide a scientific foundation for variety development aimed at improving food security."
    )

    return {
        "Section 1 \u2014 Experimental Overview":              sec1,
        "Section 2 \u2014 Statistical Results & Model Fit":    sec2,
        "Section 3 \u2014 Variance Components & Heritability": sec3,
        "Section 4 \u2014 Stability Analysis":                 sec4,
        "Section 5 \u2014 Biological & Breeding Implications": sec5,
        "Section 6 \u2014 Practical Recommendations":          sec6,
        "Section 7 \u2014 Limitations & Future Work":          sec7,
        "Section 8 \u2014 Conclusion":                         sec8,
    }


def build_genetics_publication_tables(
    envelope: Dict[str, Any],
    trait: str,
    design: TrialDesign,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Build publication-ready tables for a genetics trial envelope.

    Keys returned:
      report_header               metadata section
      variance_components_table   σ²g / σ²e / σ²gl / σ²p with % of σ²p  (Table 1)
      heritability_ga_table       H², GA, GA%, GCV, PCV, ECV             (Table 2)
      combined_anova_table        Source / df / SS / MS / F / p           (Table 3)
      genotype_means_table        Rank / Genotype / Mean / Recommendation  (Table 4)
      stability_table             bi / S²di / ASV / Classification        (Table 5)
      backend_interpretation      concise 2-paragraph summary
      academic_interpretation     Dr. Fayeun's 8-section report dict
    """
    from datetime import datetime as _dt

    meta   = envelope.get("meta", {})
    tables = envelope.get("tables", {})
    now    = _dt.now()
    pub: Dict[str, Any] = {}

    n_genos = meta.get("n_genotypes", "—")
    n_locs  = meta.get("n_locations", "—")
    mode    = meta.get("mode", "multi_environment")

    # ── REPORT HEADER ──────────────────────────────────────────────────
    pub["report_header"] = {
        "title": "VivaSense\u2122 \u2014 Plant Breeding Genetics Analysis Report",
        "generated": now.strftime("%m/%d/%Y, %I:%M:%S %p"),
        "software": "VivaSense V2.2",
        "sections": [
            {"label": "Trait",               "value": trait},
            {"label": "Design",              "value": f"RCBD Multi-environment ({mode})"},
            {"label": "Genotype Column",     "value": design.genotype_col},
            {"label": "Location Column",     "value": design.location_col or "\u2014"},
            {"label": "Number of Genotypes", "value": str(n_genos)},
            {"label": "Number of Locations", "value": str(n_locs)},
            {"label": "Analysis Date",       "value": now.strftime("%m/%d/%Y, %I:%M:%S %p")},
            {"label": "Analysis ID",         "value": meta.get("analysis_id", "\u2014")},
            {"label": "Software",            "value": "VivaSense V2.2"},
        ],
    }

    # ── TABLE 1: VARIANCE COMPONENTS ──────────────────────────────────
    vc_recs = tables.get("variance_components", [])
    vc_row  = vc_recs[0] if vc_recs else {}
    s2g  = vc_row.get("sigma2_g")
    s2e  = vc_row.get("sigma2_e")
    s2gl = vc_row.get("sigma2_gl")
    s2p  = vc_row.get("sigma2_p")
    gm   = vc_row.get("grand_mean")
    gcv  = vc_row.get("GCV")
    pcv  = vc_row.get("PCV")
    ecv  = vc_row.get("ECV")

    def _pct_p(v):
        if v is None or s2p is None or float(str(s2p) or "0") == 0:
            return "\u2014"
        return f"{float(str(v)) / float(str(s2p)) * 100:.1f}"

    def _vc_i(comp, v):
        if v is None:
            return "\u2014"
        fv = float(str(v))
        sp = float(str(s2p)) if s2p else 1
        if comp == "g":
            return ("Very High" if fv / sp > 0.7 else "High" if fv / sp > 0.5 else
                    "Moderate" if fv / sp > 0.3 else "Low")
        if comp == "e":
            return "Minimal" if fv < 0.05 else "Low" if fv < 1 else "Moderate" if fv < 5 else "High"
        if comp == "gl":
            return "Low" if fv < 0.5 else "Moderate" if fv < 2 else "High"
        return "\u2014"

    pub["variance_components_table"] = {
        "title": f"Variance Components Analysis for {trait}",
        "table_number": "Table 1",
        "headers": ["Component", "Symbol", "Value", "% of \u03c3\u00b2p", "Interpretation"],
        "rows": [
            ["Genetic Variance",             "\u03c3\u00b2g",  _gfmt(s2g, 4),  _pct_p(s2g),  _vc_i("g", s2g)],
            ["Environmental Variance",       "\u03c3\u00b2e",  _gfmt(s2e, 4),  _pct_p(s2e),  _vc_i("e", s2e)],
            ["G \u00d7 Location Interaction", "\u03c3\u00b2gl", _gfmt(s2gl, 4), _pct_p(s2gl), _vc_i("gl", s2gl)],
            ["Phenotypic Variance (Total)",  "\u03c3\u00b2p",  _gfmt(s2p, 4),  "100.0",      "Sum of all components"],
        ],
        "grand_mean": {"label": f"Grand Mean (\u03bc) for {trait}", "value": _gfmt(gm, 2)},
        "cv_section": {
            "headers": ["Coefficient", "Value", "Description"],
            "rows": [
                ["GCV (Genetic CV)",      f"{_gfmt(gcv, 2)}%", "Genetic variation as % of mean"],
                ["PCV (Phenotypic CV)",   f"{_gfmt(pcv, 2)}%", "Total phenotypic variation as % of mean"],
                ["ECV (Environmental CV)", f"{_gfmt(ecv, 2)}%", "Environmental variation as % of mean"],
            ],
        },
        "footnotes": [
            "\u03c3\u00b2g = genetic, \u03c3\u00b2e = environmental, \u03c3\u00b2gl = G\u00d7Location interaction variance.",
            "\u03c3\u00b2p = phenotypic variance (total observable variation).",
            "GCV/PCV/ECV = Genotypic/Phenotypic/Environmental Coefficient of Variation.",
        ],
    }

    # ── TABLE 2: HERITABILITY & GENETIC ADVANCE ────────────────────────
    h2     = vc_row.get("H2_broad")
    h2_pct = vc_row.get("H2_broad_pct") or (float(str(h2)) * 100 if h2 and float(str(h2)) <= 1 else h2)
    ga     = vc_row.get("GA")
    ga_pct = vc_row.get("GA_percent")
    h2_flt = float(str(h2_pct)) if h2_pct else 0
    ga_flt = float(str(ga_pct)) if ga_pct else 0
    h2_i   = ("Very High (\u226570%) \u2014 direct selection highly effective" if h2_flt >= 70 else
               "High (60\u201370%) \u2014 direct selection effective" if h2_flt >= 60 else
               "Moderate (30\u201360%) \u2014 multi-env. testing recommended" if h2_flt >= 30 else
               "Low (<30%) \u2014 consider MAS or recurrent selection")
    ga_i   = ("Excellent (>10%) \u2014 rapid selection response" if ga_flt > 10 else
               "Good (5\u201310%) \u2014 moderate selection response" if ga_flt >= 5 else
               "Low (<5%) \u2014 limited response to selection")

    pub["heritability_ga_table"] = {
        "title": f"Heritability and Genetic Advance Estimates for {trait}",
        "table_number": "Table 2",
        "headers": ["Parameter", "Symbol", "Value", "Unit", "Formula", "Interpretation"],
        "rows": [
            ["Broad-sense Heritability",    "H\u00b2",   f"{_gfmt(h2_flt, 2)}%", "%",
             "\u03c3\u00b2g / \u03c3\u00b2p \u00d7 100",  h2_i],
            ["Genetic Advance (5% sel.)",   "GA",    _gfmt(ga, 4),   "trait units",
             "k \u00d7 \u221a(\u03c3\u00b2p \u00d7 H\u00b2)",  "Predicted gain per selection cycle"],
            ["Genetic Advance % of Mean",   "GA%",   f"{_gfmt(ga_flt, 2)}%", "%",
             "(GA / Grand Mean) \u00d7 100",          ga_i],
            ["Selection Intensity (k)",     "k",     "2.06",         "\u2014",
             "At 5% selection intensity",             "Standard tabulated constant"],
        ],
        "interpretation_scale": {
            "heritability": [
                "0\u201330%:   Low \u2014 environmental factors dominate",
                "30\u201360%:  Moderate \u2014 multi-environment testing needed",
                "60\u2013100%: High \u2014 direct phenotypic selection effective",
            ],
            "genetic_advance": [
                "< 5%:  Low \u2014 limited selection progress expected",
                "5\u201310%: Good \u2014 reasonable response to selection",
                "> 10%: Excellent \u2014 rapid genetic progress possible",
            ],
        },
        "footnotes": [
            "H\u00b2 = broad-sense heritability (additive + dominance + epistatic variance).",
            "GA = k \u00d7 \u221a\u03c3\u00b2p \u00d7 H\u00b2  (k = 2.06 at 5% selection intensity).",
            "GA% = (GA / Grand Mean) \u00d7 100.",
        ],
    }

    # ── TABLE 3: COMBINED ANOVA ────────────────────────────────────────
    anova_recs = tables.get("combined_anova", [])
    anova_rows: List[List[str]] = []
    ss_tot = df_tot = 0.0
    for rec in anova_recs:
        src = rec.get("source", "?")
        df_ = rec.get("df")
        ss  = rec.get("SS") or rec.get("sum_sq")
        ms  = rec.get("MS") or rec.get("mean_sq")
        f   = rec.get("F") or rec.get("F_value")
        p   = rec.get("PR(>F)") or rec.get("p_value")
        disp = rec.get("PR(>F)_display", "")
        if not disp:
            disp = "< 0.001" if (p is not None and float(str(p)) < 0.001) else _gfmt(p, 4)
        if ms is None and ss is not None and df_ and float(str(df_)) > 0:
            ms = float(str(ss)) / float(str(df_))
        if df_:  df_tot += float(str(df_))
        if ss:   ss_tot += float(str(ss))
        anova_rows.append([
            src,
            str(int(float(str(df_)))) if df_ is not None else "\u2014",
            _gfmt(ss, 4), _gfmt(ms, 4),
            _gfmt(f, 3) if f else "\u2014",
            disp, _gsig(p),
        ])
    if anova_rows:
        anova_rows.append(["Total", str(int(df_tot)), _gfmt(ss_tot, 4), "\u2014", "\u2014", "\u2014", ""])
    pub["combined_anova_table"] = {
        "title": f"Combined ANOVA for {trait} Across {n_locs} Location(s)",
        "table_number": "Table 3",
        "headers": ["Source of Variation", "df", "Sum of Squares (SS)",
                    "Mean Square (MS)", "F-value", "p-value", "Sig."],
        "rows": anova_rows,
        "footnotes": [
            "G\u00d7L = Genotype \u00d7 Location interaction.",
            "Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant.",
            "Type II ANOVA.  Block effects included.",
        ],
    }

    # ── TABLE 4: GENOTYPE MEANS & RECOMMENDATIONS ──────────────────────
    means_recs = tables.get("genotype_means", [])
    g_col      = design.genotype_col
    n_means    = len(means_recs)
    means_rows: List[List[str]] = []
    for rank, rec in enumerate(means_recs, 1):
        gid  = rec.get(g_col) or rec.get("genotype") or rec.get("Genotype", "\u2014")
        mean = rec.get("mean") or rec.get("Mean")
        rec_text = (
            "SELECT \u2014 Best performer; advance to next generation" if rank == 1 else
            "ADVANCE \u2014 Competitive; conduct further evaluation" if rank == 2 else
            "DISCARD \u2014 Lowest performer; remove from programme" if rank == n_means else
            "EVALUATE \u2014 Intermediate; retain for specific environments"
        )
        means_rows.append([str(rank), str(gid), _gfmt(mean, 2), "\u2014", "\u2014", rec_text])
    pub["genotype_means_table"] = {
        "title": f"Mean {trait} by Genotype and Breeding Recommendations",
        "table_number": "Table 4",
        "headers": ["Rank", "Genotype", "Mean", "SD", "SE", "Recommendation"],
        "rows": means_rows,
        "footnotes": [
            "Ranked from highest to lowest mean performance.",
            "SELECT: Advance to replicated trials or use as crossing parent.",
            "ADVANCE: Promising \u2014 continue evaluation across environments.",
            "EVALUATE: Intermediate \u2014 retain for niche environments.",
            "DISCARD: Lowest performance \u2014 remove to conserve resources.",
        ],
    }

    # ── TABLE 5: STABILITY (Eberhart & Russell) ────────────────────────
    stab_recs = tables.get("stability", [])
    stab_rows: List[List[str]] = []
    for rec in stab_recs:
        gid  = rec.get("Genotype") or rec.get("genotype", "\u2014")
        mean = rec.get("grand_mean") or rec.get("mean")
        bi   = rec.get("bi") or rec.get("regression_coefficient")
        s2di = rec.get("S2di") or rec.get("s2di") or rec.get("stability_variance")
        asv  = rec.get("ASV") or rec.get("asv")
        cls  = rec.get("classification", "\u2014")
        bi_f = float(str(bi)) if bi else None
        bi_note = ("Stable (bi \u2248 1)" if bi_f and abs(bi_f - 1.0) < 0.15 else
                   "Responsive to good envs" if bi_f and bi_f > 1 else
                   "Stable in poor envs" if bi_f else "\u2014")
        stab_rows.append([str(gid), _gfmt(mean, 2), _gfmt(bi, 3), _gfmt(s2di, 4),
                          _gfmt(asv, 3), str(cls), bi_note])
    pub["stability_table"] = {
        "title": f"Stability Parameters (Eberhart & Russell, 1966) for {trait}",
        "table_number": "Table 5",
        "headers": ["Genotype", "Mean", "bi", "S\u00b2di", "ASV", "Classification", "bi Interpretation"],
        "rows": stab_rows,
        "interpretation_guide": [
            "bi \u2248 1.0: Average stability \u2014 suitable for all environments.",
            "bi > 1.0: Responsive \u2014 performs best in favourable/high-input environments.",
            "bi < 1.0: Stable in poor environments \u2014 suited to marginal conditions.",
            "S\u00b2di \u2248 0: Minimal unpredictable variation (stable).",
            "Lower ASV = more stable across environments.",
        ],
        "footnotes": [
            "Eberhart & Russell (1966): \u0232\u1d62\u2c7c = \u03bc\u1d62 + b\u1d62(I\u2c7c) + \u03b4\u1d62\u2c7c",
            "bi = regression coefficient of genotype mean on environment index.",
            "S\u00b2di = deviation mean square from regression.",
            "ASV = AMMI Stability Value (Purchase et al., 2000).",
        ],
    }

    # ── INTERPRETATIONS ────────────────────────────────────────────────
    pub["backend_interpretation"] = _generate_genetics_backend_interp(envelope, trait)
    pub["academic_interpretation"] = _generate_genetics_academic_interp(
        envelope, trait, design, alpha
    )

    return pub


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
    primary_trait_col: str = Form(
        "",
        description="Primary trait to feature in the main envelope; blank = first detected",
    ),
    traits: str = Form(
        "",
        description="Alias for trait_cols — accepts comma-separated string, "
                    "JSON array '[\"T1\",\"T2\"]', or repeated form fields. "
                    "Takes precedence over trait_cols when non-empty.",
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
        trait_cols_str=_resolve_trait_cols(trait_cols, traits),
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
    primary_trait = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )

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
@genetics_router.post("/variance_components")
async def genetics_variance_components(
    file: UploadFile = File(...),
    genotype_col: str = Form("Genotype"),
    location_col: str = Form("Location"),
    rep_col: str = Form("Rep"),
    trait_cols: str = Form(""),
    alpha: float = Form(0.05),
    primary_trait_col: str = Form("", description="Primary trait to feature; blank = first detected"),
    traits: str = Form("", description="Alias for trait_cols (comma-separated, JSON array, or repeated fields)"),
):
    """
    Variance components only (σ²g, σ²e, σ²gl, H², GA, GCV, PCV).
    Requires ≥2 locations for σ²gl; falls back to single-environment estimates.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(genotype_col, location_col, rep_col, _resolve_trait_cols(trait_cols, traits), "", "RCBD")
    cfg    = _build_config(alpha, 2)

    pipeline = GeneticsPipeline(config=cfg, design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    # Slim down to VC-relevant keys
    _VC_TABLE_NAMES = {"Variance Components", "Heritability & Genetic Advance",
                       "Combined ANOVA", "Genotype Means & Rankings"}
    _VC_FIG_NAMES   = {"Genotype Means Bar Chart"}

    def _vc_slim(env: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta":    env["meta"],
            "tables": {
                "variance_components": env["tables"]["variance_components"],
                "combined_anova":      env["tables"]["combined_anova"],
                "assumption_guidance": env["tables"]["assumption_guidance"],
            },
            "html_tables":         [t for t in env.get("html_tables", [])
                                    if t.get("name") in _VC_TABLE_NAMES],
            "publication_figures": [f for f in env.get("publication_figures", [])
                                    if f.get("name") in _VC_FIG_NAMES],
            "interpretation":  env["interpretation"],
            "strict_template": env["strict_template"],
            "intelligence":    env["intelligence"],
        }

    slim = _vc_slim(envelope)

    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                per_trait[t] = _vc_slim(_build_trial_envelope(raw, t, design, alpha))
            except Exception as exc:
                logger.warning("Could not build VC envelope for trait %s: %s", t, exc)
        slim["per_trait"] = per_trait

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
    primary_trait_col: str = Form("", description="Primary trait to feature; blank = first detected"),
    traits: str = Form("", description="Alias for trait_cols (comma-separated, JSON array, or repeated fields)"),
):
    """
    Stability parameters only (Eberhart-Russell bi, S²di, ASV).
    Requires ≥2 locations.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col, location_col])

    design = _build_design(genotype_col, location_col, rep_col, _resolve_trait_cols(trait_cols, traits), "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    _STAB_TABLE_NAMES = {"Stability Analysis", "Genotype Means & Rankings"}
    _STAB_FIG_NAMES   = {"Stability Scatter Plot", "Genotype Means Bar Chart"}

    def _stab_slim(env: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta":    env["meta"],
            "tables": {
                "stability":      env["tables"]["stability"],
                "genotype_means": env["tables"]["genotype_means"],
            },
            "plots":               {k: v for k, v in env["plots"].items()
                                    if any(x in k for x in ["stability", "bi"])},
            "html_tables":         [t for t in env.get("html_tables", [])
                                    if t.get("name") in _STAB_TABLE_NAMES],
            "publication_figures": [f for f in env.get("publication_figures", [])
                                    if f.get("name") in _STAB_FIG_NAMES],
            "interpretation":  env["interpretation"],
            "strict_template": env["strict_template"],
            "intelligence":    env["intelligence"],
        }

    slim = _stab_slim(envelope)

    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                per_trait[t] = _stab_slim(_build_trial_envelope(raw, t, design, alpha))
            except Exception as exc:
                logger.warning("Could not build stability envelope for trait %s: %s", t, exc)
        slim["per_trait"] = per_trait

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
    primary_trait_col: str = Form("", description="Primary trait to feature; blank = first detected"),
    traits: str = Form("", description="Alias for trait_cols (comma-separated, JSON array, or repeated fields)"),
):
    """
    AMMI model (ANOVA partition + IPCA scores + biplot data).
    Requires ≥3 locations.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col, location_col])

    design = _build_design(genotype_col, location_col, rep_col, _resolve_trait_cols(trait_cols, traits), "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, n_ammi_axes), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    _AMMI_TABLE_NAMES = {"Combined ANOVA", "Genotype Means & Rankings"}
    _AMMI_FIG_NAMES   = {"AMMI Biplot", "Genotype Means Bar Chart"}

    def _ammi_slim(env: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta":    env["meta"],
            "tables": {
                "ammi_ipca":               env["tables"].get("ammi_ipca", []),
                "ammi_explained_variance": env["tables"].get("ammi_explained_variance", []),
                "combined_anova":          env["tables"]["combined_anova"],
            },
            "plots":               {k: v for k, v in env["plots"].items() if "ammi" in k},
            "html_tables":         [t for t in env.get("html_tables", [])
                                    if t.get("name") in _AMMI_TABLE_NAMES],
            "publication_figures": [f for f in env.get("publication_figures", [])
                                    if f.get("name") in _AMMI_FIG_NAMES],
            "interpretation":  env["interpretation"],
            "strict_template": env["strict_template"],
            "intelligence":    env["intelligence"],
        }

    slim = _ammi_slim(envelope)

    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                per_trait[t] = _ammi_slim(_build_trial_envelope(raw, t, design, alpha))
            except Exception as exc:
                logger.warning("Could not build AMMI envelope for trait %s: %s", t, exc)
        slim["per_trait"] = per_trait

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
    primary_trait_col: str = Form("", description="Primary trait to feature; blank = first detected"),
    traits: str = Form("", description="Alias for trait_cols (comma-separated, JSON array, or repeated fields)"),
):
    """
    GGE biplot (which-won-where, ideal genotype, mega-environments).
    Requires ≥2 locations.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col, location_col])

    design = _build_design(genotype_col, location_col, rep_col, _resolve_trait_cols(trait_cols, traits), "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    _GGE_TABLE_NAMES = {"Genotype Means & Rankings"}
    _GGE_FIG_NAMES   = {"GGE Biplot (Which-Won-Where)", "Genotype Means Bar Chart"}

    def _gge_slim(env: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta":    env["meta"],
            "tables": {
                "gge_which_won_where": env["tables"].get("gge_which_won_where", []),
                "genotype_means":      env["tables"]["genotype_means"],
            },
            "plots":               {k: v for k, v in env["plots"].items() if "gge" in k},
            "html_tables":         [t for t in env.get("html_tables", [])
                                    if t.get("name") in _GGE_TABLE_NAMES],
            "publication_figures": [f for f in env.get("publication_figures", [])
                                    if f.get("name") in _GGE_FIG_NAMES],
            "interpretation":  env["interpretation"],
            "strict_template": env["strict_template"],
            "intelligence":    env["intelligence"],
        }

    slim = _gge_slim(envelope)

    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                per_trait[t] = _gge_slim(_build_trial_envelope(raw, t, design, alpha))
            except Exception as exc:
                logger.warning("Could not build GGE envelope for trait %s: %s", t, exc)
        slim["per_trait"] = per_trait

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
    primary_trait_col: str = Form("", description="Primary trait to feature; blank = first detected"),
    traits: str = Form("", description="Alias for trait_cols (comma-separated, JSON array, or repeated fields)"),
):
    """
    Phenotypic + genotypic correlations, path analysis, Smith-Hazel selection index.
    Requires ≥2 trait columns.
    """
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(genotype_col, location_col, rep_col, _resolve_trait_cols(trait_cols, traits), "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    # Correlations/path/selection index are inherently multi-trait — always
    # embed full per-trait envelopes so the frontend can tab through traits.
    _CORR_TABLE_NAMES = {"Phenotypic Correlations", "Path Analysis", "Selection Index"}
    _CORR_FIG_NAMES   = {"Phenotypic Correlation Heatmap", "Genotype Means Bar Chart"}
    slim = {
        "meta":    envelope["meta"],
        "tables": {
            "correlations":    envelope["tables"].get("correlations", {}),
            "path_analysis":   envelope["tables"].get("path_analysis", {}),
            "selection_index": envelope["tables"].get("selection_index", {}),
        },
        "plots":               {k: v for k, v in envelope["plots"].items()
                                if any(x in k for x in ["heatmap", "path", "corr"])},
        "html_tables":         [t for t in envelope.get("html_tables", [])
                                if t.get("name") in _CORR_TABLE_NAMES],
        "publication_figures": [f for f in envelope.get("publication_figures", [])
                                if f.get("name") in _CORR_FIG_NAMES],
        "interpretation":  envelope["interpretation"],
        "strict_template": envelope["strict_template"],
        "intelligence":    envelope["intelligence"],
    }

    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                t_env = _build_trial_envelope(raw, t, design, alpha)
                per_trait[t] = {
                    "meta":           t_env["meta"],
                    "tables": {
                        "correlations":    t_env["tables"].get("correlations", {}),
                        "path_analysis":   t_env["tables"].get("path_analysis", {}),
                        "selection_index": t_env["tables"].get("selection_index", {}),
                    },
                    "interpretation": t_env["interpretation"],
                    "intelligence":   t_env["intelligence"],
                }
            except Exception as exc:
                logger.warning("Could not build correlations envelope for trait %s: %s", t, exc)
        slim["per_trait"] = per_trait

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
    primary_trait_col: str = Form("", description="Primary trait to feature; blank = first detected"),
    traits: str = Form("", description="Alias for trait_cols (comma-separated, JSON array, or repeated fields)"),
):
    """PCA + hierarchical clustering (Ward) + k-means on genotype means."""
    df = await _load_file(file)
    require_cols(df, [genotype_col])

    design = _build_design(genotype_col, location_col, rep_col, _resolve_trait_cols(trait_cols, traits), "", "RCBD")
    pipeline = GeneticsPipeline(config=_build_config(alpha, 2), design=design)
    raw = pipeline.run_trial_analysis(df, filename=file.filename or "upload")

    if raw.get("status") in ("validation_error", "error"):
        raise HTTPException(status_code=400, detail=raw.get("errors", "Analysis failed."))

    trait_list = raw.get("metadata", {}).get("trait_cols", [])
    primary = (
        primary_trait_col if primary_trait_col in trait_list
        else (trait_list[0] if trait_list else "trait")
    )
    envelope = _build_trial_envelope(raw, primary, design, alpha)

    _MV_TABLE_NAMES = {"PCA Loadings Matrix", "Cluster Membership"}
    _MV_FIG_NAMES   = {"PCA Biplot", "Hierarchical Dendrogram", "PCA Scree Plot"}

    def _mv_slim(env: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta":    env["meta"],
            "tables": {
                "multivariate": env["tables"].get("multivariate", {}),
            },
            "plots":               {k: v for k, v in env["plots"].items()
                                    if any(x in k for x in ["pca", "scree", "dendrogram", "cluster"])},
            "html_tables":         [t for t in env.get("html_tables", [])
                                    if t.get("name") in _MV_TABLE_NAMES],
            "publication_figures": [f for f in env.get("publication_figures", [])
                                    if f.get("name") in _MV_FIG_NAMES],
            "interpretation":  env["interpretation"],
            "strict_template": env["strict_template"],
            "intelligence":    env["intelligence"],
        }

    slim = _mv_slim(envelope)

    if len(trait_list) > 1:
        per_trait: Dict[str, Any] = {}
        for t in trait_list:
            try:
                per_trait[t] = _mv_slim(_build_trial_envelope(raw, t, design, alpha))
            except Exception as exc:
                logger.warning("Could not build multivariate envelope for trait %s: %s", t, exc)
        slim["per_trait"] = per_trait

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
