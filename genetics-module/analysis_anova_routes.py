"""
VivaSense – ANOVA Analysis Module

POST /analysis/anova

Returns per-trait:
  • ANOVA table (source, df, SS, MS, F, p)
  • Descriptive statistics
  • Assumption tests (Shapiro-Wilk, Levene — if available from R)
  • Mean separation (Tukey HSD or LSD)
  • Interpretation text
  • Data warnings (balance / completeness issues)

The R engine (vivasense_genetics.R) performs all computation.
Python re-uses the dataset context stored by POST /upload/dataset,
builds flat observation records, dispatches to R, and slices the result.

Analysis results are cached in dataset_cache so that a subsequent call
to /analysis/genetic-parameters for the same dataset_token + traits
does not trigger a second R subprocess call.
"""

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from genetics_schemas import GeneticsResponse
from multitrait_upload_routes import build_observations, check_balance, read_file
from module_schemas import AnovaModuleResponse, AnovaTraitResult, ModuleRequest
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


def compute_cv_from_anova(
    anova_table,
    grand_mean: Optional[float],
) -> Optional[float]:
    """
    Compute CV% from the ANOVA error mean square.

    CV% = sqrt(MSE) / grand_mean × 100

    Tries common residual/error row names in the ANOVA source column
    (R typically uses "Residuals"; other conventions are also handled).
    Returns None when the table is absent, grand_mean is zero, MSE is
    negative (variance component issue), or no error term is found.
    """
    if anova_table is None or grand_mean is None or grand_mean == 0:
        return None
    if not hasattr(anova_table, "source") or not hasattr(anova_table, "ms"):
        return None

    error_terms = ["Residuals", "residuals", "Residual", "residual", "error", "Error"]
    for term in error_terms:
        try:
            idx = anova_table.source.index(term)
            mse = anova_table.ms[idx]
            if mse is not None and mse >= 0:
                return float((mse ** 0.5) / grand_mean * 100)
        except (ValueError, IndexError, TypeError):
            continue
    return None


def _generate_split_plot_interpretation(
    trait: str,
    summary: Dict[str, Optional[float]],
    precision_level: Optional[str],
    cv_interpretation_flag: Optional[str],
    main_plot_significant: Optional[bool],
    subplot_significant: Optional[bool],
    interaction_significant: Optional[bool],
    n_reps: Optional[int],
) -> str:
    """
    Domain-neutral ANOVA interpretation for generic split-plot RCBD.

    Uses role-based language (main-plot factor, subplot factor) with no
    reference to genotypes, breeding, or selection.
    """
    sections: List[tuple] = []

    # ── 1. Overview ────────────────────────────────────────────────────────────
    overview = [
        "This analysis used a split-plot randomised complete block design (RCBD), "
        "in which the main-plot factor was assigned to larger experimental units "
        "within each replication and the subplot factor was assigned within main plots."
    ]
    if n_reps:
        overview.append(f"The experiment had {n_reps} complete replication(s).")
    if summary.get("grand_mean") is not None:
        overview.append(
            f"The overall mean of {trait} across all experimental units was "
            f"{summary['grand_mean']:.2f}."
        )
    if cv_interpretation_flag == "cv_available" and summary.get("cv_percent") is not None:
        cv = summary["cv_percent"]
        precision_word = "good" if cv < 10 else "moderate" if cv <= 20 else "low"
        overview.append(
            f"The coefficient of variation (CV) was {cv:.1f}%, indicating "
            f"{precision_word} experimental precision."
        )
    sections.append(("Overview", " ".join(overview)))

    # ── 2. Descriptive Summary ─────────────────────────────────────────────────
    desc = []
    if (
        summary.get("min") is not None
        and summary.get("max") is not None
        and summary.get("range") is not None
        and summary.get("grand_mean") is not None
    ):
        variability = (
            "substantial"
            if summary["range"] > summary["grand_mean"] * 0.5
            else "moderate"
        )
        desc.append(
            f"{trait} ranged from {summary['min']:.2f} to {summary['max']:.2f} "
            f"(range = {summary['range']:.2f}), indicating {variability} variability "
            "across treatment combinations."
        )
    if precision_level == "good":
        desc.append("Experimental precision was good, supporting reliable inference.")
    elif precision_level == "moderate":
        desc.append("Experimental precision was moderate; results should be interpreted with care.")
    elif precision_level == "low":
        desc.append(
            "Experimental precision was low. High unexplained variability may reduce "
            "confidence in treatment comparisons."
        )
    sections.append(("Descriptive Summary", " ".join(desc) if desc else "Descriptive statistics were not available."))

    # ── 3. Main-Plot Factor Effect ─────────────────────────────────────────────
    if main_plot_significant is True:
        mp_text = (
            f"The main-plot factor had a significant effect on {trait} (p < 0.05), "
            "indicating that levels of the whole-plot treatment produced meaningfully "
            "different responses."
        )
    elif main_plot_significant is False:
        mp_text = (
            f"The main-plot factor did not have a significant effect on {trait}, "
            "suggesting that the whole-plot treatment levels produced similar responses."
        )
    else:
        mp_text = f"The significance of the main-plot factor effect on {trait} could not be determined."
    sections.append(("Main-Plot Factor Effect", mp_text))

    # ── 4. Subplot Factor Effect ───────────────────────────────────────────────
    if subplot_significant is True:
        sub_text = (
            f"The subplot factor had a significant effect on {trait} (p < 0.05), "
            "indicating differential responses across subplot treatment levels."
        )
    elif subplot_significant is False:
        sub_text = (
            f"The subplot factor did not have a significant effect on {trait}, "
            "suggesting that subplot treatment levels produced similar responses."
        )
    else:
        sub_text = f"The significance of the subplot factor effect on {trait} could not be determined."
    sections.append(("Subplot Factor Effect", sub_text))

    # ── 5. Interaction Effect ──────────────────────────────────────────────────
    if interaction_significant is True:
        int_text = (
            f"A significant main-plot × subplot interaction was detected for {trait} "
            "(p < 0.05), indicating that the effect of the subplot factor depends on "
            "which level of the main-plot factor is applied. Treatment combinations "
            "should be evaluated jointly rather than interpreting main effects alone."
        )
    elif interaction_significant is False:
        int_text = (
            f"No significant main-plot × subplot interaction was detected for {trait}, "
            "suggesting that the effects of the two treatment factors are additive "
            "and can be interpreted independently."
        )
    else:
        int_text = (
            f"The main-plot × subplot interaction for {trait} could not be evaluated."
        )
    sections.append(("Main-Plot × Subplot Interaction", int_text))

    # ── 6. Risk and Limitations ────────────────────────────────────────────────
    risks = []
    if precision_level == "low":
        risks.append(
            "The low experimental precision introduces uncertainty in comparisons "
            "and may be due to heterogeneous experimental units or insufficient replication."
        )
    if interaction_significant is True:
        risks.append(
            "The significant interaction complicates interpretation of marginal "
            "factor means; treatment-combination means should be used for conclusions."
        )
    if not risks:
        risks.append("No major experimental limitations were identified.")
    sections.append(("Risk and Limitations", " ".join(risks)))

    # ── 7. Recommendation ─────────────────────────────────────────────────────
    recs = []
    if interaction_significant is True:
        recs.append(
            "Use treatment-combination means (main_plot × sub_plot cells) for "
            "decision-making rather than marginal factor means."
        )
    if main_plot_significant is True or subplot_significant is True:
        recs.append(
            "Apply mean separation tests within the appropriate error stratum "
            "(whole-plot error for main-plot comparisons, subplot error for "
            "subplot and interaction comparisons)."
        )
    if precision_level == "low":
        recs.append(
            "Increase replication or improve experimental control to reduce "
            "unexplained variability in future trials."
        )
    recs.append(
        "Integrate these ANOVA results with treatment-level means and practical "
        "relevance thresholds before drawing applied conclusions."
    )
    sections.append(("Recommendation", " ".join(recs)))

    return "\n\n".join(f"{heading}\n{content}" for heading, content in sections)


def generate_anova_interpretation(
    trait: str,
    summary: Dict[str, Optional[float]],
    precision_level: Optional[str],
    cv_interpretation_flag: Optional[str],
    genotype_significant: Optional[bool],
    environment_significant: Optional[bool],
    gxe_significant: Optional[bool],
    ranking_caution: Optional[bool],
    selection_feasible: Optional[bool],
    mean_separation: Optional[Any],
    n_genotypes: Optional[int],
    n_environments: Optional[int],
    n_reps: Optional[int],
    environment_mode: str = "single",
    design_type: Optional[str] = None,
    # Split-plot specific significance flags (only used when design_type == "split_plot_rcbd")
    main_plot_significant: Optional[bool] = None,
    subplot_significant: Optional[bool] = None,
    interaction_significant: Optional[bool] = None,
) -> str:
    """
    Generate context-aware ANOVA interpretation following VivaSense standards.

    For design_type="split_plot_rcbd" dispatches to a domain-neutral split-plot
    interpretation using role-based language (main-plot factor / subplot factor).
    All other designs use the genetics-aware interpretation path.

    environment_mode="single"
        Sections: Overview, Descriptive Interpretation, Genotype Effect,
                  Mean Performance and Ranking, Breeding Interpretation,
                  Risk and Limitations, Recommendation.
        Environment Effect and G×E Interaction sections are omitted.
        No environment-stability or broad-adaptation claims.

    environment_mode="multi"
        All nine sections including Environment Effect and G×E Interaction.
    """
    # ── Dispatch: generic split-plot uses its own domain-neutral path ──────────
    if design_type == "split_plot_rcbd":
        return _generate_split_plot_interpretation(
            trait=trait,
            summary=summary,
            precision_level=precision_level,
            cv_interpretation_flag=cv_interpretation_flag,
            main_plot_significant=main_plot_significant,
            subplot_significant=subplot_significant,
            interaction_significant=interaction_significant,
            n_reps=n_reps,
        )

    is_multi = environment_mode == "multi"
    # List of (heading, content) tuples built up below
    sections: List[tuple] = []

    # ── 1. Overview ────────────────────────────────────────────────────────────
    overview = []
    if n_genotypes and n_reps:
        if is_multi and n_environments and n_environments > 1:
            overview.append(
                f"This analysis evaluated {trait} across {n_genotypes} genotypes "
                f"tested in {n_environments} environments with {n_reps} replications "
                "per genotype-environment combination."
            )
        else:
            overview.append(
                f"This analysis evaluated {trait} across {n_genotypes} genotypes "
                f"with {n_reps} replications per genotype."
            )

    if summary.get("grand_mean") is not None:
        overview.append(
            f"The overall mean performance for {trait} was {summary['grand_mean']:.2f}."
        )

    if design_type == "split_plot_rcbd":
        overview.append(
            "The experiment was analysed using a split-plot RCBD structure, "
            "with whole plots and subplots accounted for in the model."
        )

    if cv_interpretation_flag == "cv_available" and summary.get("cv_percent") is not None:
        cv = summary["cv_percent"]
        precision_word = "good" if cv < 10 else "moderate" if cv <= 20 else "low"
        overview.append(
            f"The coefficient of variation (CV) was {cv:.1f}%, indicating "
            f"{precision_word} experimental precision."
        )
    else:
        overview.append(
            "Experimental precision could not be assessed due to insufficient "
            "data for CV calculation."
        )

    sections.append(("Overview", " ".join(overview)))

    # ── 2. Descriptive Interpretation ─────────────────────────────────────────
    desc = []
    if summary.get("grand_mean") is not None:
        desc.append(
            f"The grand mean of {summary['grand_mean']:.2f} represents the average "
            f"{trait} performance across all experimental units."
        )

    if (
        summary.get("min") is not None
        and summary.get("max") is not None
        and summary.get("range") is not None
    ):
        variability = (
            "substantial"
            if summary["range"] > summary["grand_mean"] * 0.5
            else "moderate"
        )
        desc.append(
            f"Performance ranged from {summary['min']:.2f} to {summary['max']:.2f}, "
            f"with a total range of {summary['range']:.2f}, indicating {variability} "
            "variability among experimental units."
        )

    if precision_level == "good":
        desc.append(
            "The experimental precision was good, suggesting reliable and "
            "reproducible results."
        )
    elif precision_level == "moderate":
        desc.append(
            "The experimental precision was moderate, indicating acceptable but "
            "not optimal experimental control."
        )
    elif precision_level == "low":
        desc.append(
            "The experimental precision was low, suggesting high variability that "
            "warrants cautious interpretation of the results."
        )
    elif cv_interpretation_flag == "cv_unavailable":
        desc.append(
            "Experimental precision could not be assessed, limiting confidence "
            "in the results."
        )

    sections.append(("Descriptive Interpretation", " ".join(desc)))

    # ── 3. Genotype Effect ─────────────────────────────────────────────────────
    if genotype_significant is True:
        geno_text = (
            f"Significant genetic variation was detected for {trait} (p < 0.05), "
            f"indicating that genotypes differ in their performance and that "
            f"selection for improved {trait} is feasible."
        )
    elif genotype_significant is False:
        geno_text = (
            f"No significant genetic variation was detected for {trait}, suggesting "
            "that the genotypes tested do not differ sufficiently to justify "
            "selection based on this trait."
        )
    else:
        geno_text = (
            f"The significance of genetic variation for {trait} could not be determined."
        )
    sections.append(("Genotype Effect", geno_text))

    # ── 4. Environment Effect (multi only) ────────────────────────────────────
    if is_multi:
        if environment_significant is True:
            env_text = (
                f"Significant environmental variation was observed for {trait}, "
                "indicating that growing conditions substantially influence performance "
                "and that results may not be transferable across environments."
            )
        elif environment_significant is False:
            env_text = (
                f"No significant environmental variation was detected for {trait}, "
                "suggesting relatively consistent performance across the tested conditions."
            )
        else:
            env_text = (
                f"The significance of environmental variation for {trait} "
                "could not be determined."
            )
        sections.append(("Environment Effect", env_text))

        # ── 5. G×E Interaction (multi only) ───────────────────────────────────
        if gxe_significant is True:
            gxe_text = (
                f"A significant genotype \u00d7 environment interaction was detected "
                f"for {trait}, indicating that genotype performance is not consistent "
                "across environments. This suggests that no single genotype is "
                "universally superior, and selection strategies should account for "
                "environmental stability."
            )
        elif gxe_significant is False:
            gxe_text = (
                f"No significant genotype \u00d7 environment interaction was detected "
                f"for {trait}, suggesting relatively stable genotype performance across "
                "the tested environments."
            )
        else:
            gxe_text = (
                f"The presence of genotype \u00d7 environment interaction for {trait} "
                "could not be determined."
            )
        sections.append(("G\u00d7E Interaction", gxe_text))

    # ── 6 (single: 4). Mean Performance and Ranking ───────────────────────────
    ranking = []
    if mean_separation and hasattr(mean_separation, "genotype") and mean_separation.genotype:
        try:
            top_genotype = mean_separation.genotype[0]
            top_mean = mean_separation.mean[0]
            ranking.append(
                f"Based on overall means, {top_genotype} exhibited the highest "
                f"{trait} performance ({top_mean:.2f})."
            )
        except (IndexError, TypeError):
            ranking.append(
                "Mean separation analysis was available but could not be summarised."
            )
    else:
        ranking.append("Detailed mean separation analysis was not available.")

    # Ranking caution only meaningful in multi-environment context
    if is_multi and ranking_caution is True:
        ranking.append(
            "However, due to significant genotype \u00d7 environment interaction, "
            "ranking based on overall means should be interpreted cautiously, as "
            "performance may vary across environments."
        )

    sections.append(("Mean Performance and Ranking", " ".join(ranking)))

    # ── 7 (single: 5). Breeding Interpretation ────────────────────────────────
    breeding = []
    if selection_feasible is True:
        breeding.append(
            f"The results suggest that selection for improved {trait} is feasible."
        )
        if is_multi:
            if gxe_significant is False:
                breeding.append(
                    "Given the absence of significant genotype \u00d7 environment "
                    "interaction, breeding efforts can focus on broad adaptation "
                    "across environments."
                )
            else:
                breeding.append(
                    "However, due to significant genotype \u00d7 environment "
                    "interaction, breeding strategies should prioritise stability "
                    "analysis and environment-specific selection."
                )
    else:
        breeding.append(
            f"The lack of significant genetic variation indicates that selection "
            f"for {trait} may not be effective with the current germplasm."
        )

    breeding.append(
        "The observed variability and experimental precision should guide the "
        "design of future experiments and breeding trials."
    )
    sections.append(("Breeding Interpretation", " ".join(breeding)))

    # ── 8 (single: 6). Risk and Limitations ───────────────────────────────────
    risks = []
    if is_multi and gxe_significant is True:
        risks.append(
            "The significant genotype \u00d7 environment interaction represents a "
            "major limitation, as it complicates genotype evaluation and selection."
        )
    if precision_level == "low":
        risks.append(
            "The low experimental precision introduces uncertainty in the results "
            "and suggests potential issues with experimental control or replication."
        )
    if is_multi and environment_significant is True:
        risks.append(
            "Strong environmental influence may limit the generalisability of these "
            "results to other locations or conditions."
        )
    if not risks:
        risks.append("No major experimental limitations were identified in this analysis.")
    sections.append(("Risk and Limitations", " ".join(risks)))

    # ── 9 (single: 7). Recommendation ─────────────────────────────────────────
    recs = []
    if is_multi and gxe_significant is True:
        recs.append(
            "Conduct stability analysis (e.g., AMMI or GGE biplot) to identify "
            "genotypes with consistent performance across environments."
        )
    if selection_feasible is True:
        if is_multi:
            recs.append(
                "Consider advancing promising genotypes to further evaluation, "
                "with appropriate caution regarding environmental interactions."
            )
        else:
            recs.append(
                "Consider advancing promising genotypes to further evaluation "
                "in additional environments to validate their performance."
            )
    if precision_level == "low":
        recs.append(
            "Improve experimental design by increasing replication or enhancing "
            "environmental control to reduce variability."
        )
    recs.append(
        "Integrate these ANOVA results with genetic parameter estimates "
        "(heritability, genetic coefficient of variation) for comprehensive "
        "trait evaluation."
    )
    sections.append(("Recommendation", " ".join(recs)))

    return "\n\n".join(f"{heading}\n{content}" for heading, content in sections)


def compute_descriptive_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """Compute numeric descriptive statistics for a trait series.

    This uses raw observation-level data and ignores missing values.
    If insufficient observations exist, variance/SD/SE are returned as None.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)

    if n == 0:
        return {
            "grand_mean": None,
            "standard_deviation": None,
            "variance": None,
            "standard_error": None,
            "cv_percent": None,
            "min": None,
            "max": None,
            "range": None,
        }

    grand_mean = float(clean.mean())
    min_val = float(clean.min())
    max_val = float(clean.max())
    range_val = max_val - min_val

    if n >= 2:
        variance = float(clean.var(ddof=1))
        standard_deviation = float(variance ** 0.5)
        standard_error = float(standard_deviation / (n ** 0.5))
    else:
        variance = None
        standard_deviation = None
        standard_error = None

    cv_percent = None
    if grand_mean != 0 and standard_deviation is not None:
        cv_percent = float((standard_deviation / grand_mean) * 100)

    return {
        "grand_mean": grand_mean,
        "standard_deviation": standard_deviation,
        "variance": variance,
        "standard_error": standard_error,
        "cv_percent": cv_percent,
        "min": min_val,
        "max": max_val,
        "range": range_val,
    }


def compute_per_genotype_stats(
    df: pd.DataFrame, trait_column: str, genotype_column: str
) -> List[Dict[str, Optional[float]]]:
    """Compute per-genotype descriptive statistics for the requested trait."""
    if genotype_column not in df.columns:
        return []

    grouped = df[[genotype_column, trait_column]].copy()
    grouped[trait_column] = pd.to_numeric(grouped[trait_column], errors="coerce")

    stats: List[Dict[str, Optional[float]]] = []
    for genotype, group in grouped.groupby(genotype_column, sort=True):
        clean = group[trait_column].dropna()
        n = len(clean)

        if n == 0:
            stats.append(
                {
                    "genotype": genotype,
                    "mean": None,
                    "sd": None,
                    "cv_percent": None,
                }
            )
            continue

        mean_val = float(clean.mean())
        sd_val = None
        cv_percent = None

        if n >= 2:
            variance = float(clean.var(ddof=1))
            sd_val = float(variance ** 0.5)
            if mean_val != 0:
                cv_percent = float((sd_val / mean_val) * 100)

        stats.append(
            {
                "genotype": genotype,
                "mean": mean_val,
                "sd": sd_val,
                "cv_percent": cv_percent,
            }
        )

    return stats


def classify_precision_level(cv_percent: Optional[float]) -> str:
    """Classify experimental precision based on coefficient of variation.

    Thresholds (aligned with VivaSense ANOVA spec):
      < 10  → good
      10–20 → moderate   (20 inclusive)
      > 20  → low
    """
    if cv_percent is None:
        return "low"  # No CV available, assume low precision
    if cv_percent < 10.0:
        return "good"
    elif cv_percent <= 20.0:
        return "moderate"
    else:
        return "low"


def get_cv_interpretation_flag(cv_percent: Optional[float]) -> str:
    """Return flag indicating if CV is available for interpretation."""
    return "cv_available" if cv_percent is not None else "cv_unavailable"


def _is_term_significant(anova_table, term: str) -> Optional[bool]:
    """Return True/False if term is in the ANOVA table, else None."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return None
    try:
        idx = anova_table.source.index(term)
        p_val = anova_table.p_value[idx]
        if p_val is None:
            return None
        return float(p_val) < 0.05
    except (ValueError, IndexError):
        return None


def is_main_plot_significant(anova_table) -> Optional[bool]:
    """Check if the main-plot factor effect is significant (split-plot ANOVA)."""
    return _is_term_significant(anova_table, "main_plot")


def is_subplot_significant(anova_table) -> Optional[bool]:
    """Check if the subplot factor effect is significant (split-plot ANOVA)."""
    return _is_term_significant(anova_table, "sub_plot")


def is_interaction_significant(anova_table) -> Optional[bool]:
    """Check if the main_plot × sub_plot interaction is significant."""
    result = _is_term_significant(anova_table, "main_plot:sub_plot")
    if result is None:
        result = _is_term_significant(anova_table, "sub_plot:main_plot")
    return result


def is_genotype_effect_significant(anova_table) -> bool:
    """Check if genotype effect is significant (p < 0.05)."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return False
    try:
        idx = anova_table.source.index("genotype")
        p_val = anova_table.p_value[idx]
        return p_val is not None and p_val < 0.05
    except (ValueError, IndexError):
        return False


def is_environment_effect_significant(anova_table) -> bool:
    """Check if environment effect is significant (p < 0.05)."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return False
    try:
        idx = anova_table.source.index("environment")
        p_val = anova_table.p_value[idx]
        return p_val is not None and p_val < 0.05
    except (ValueError, IndexError):
        return False


def is_gxe_effect_significant(anova_table) -> bool:
    """Check if genotype x environment interaction is significant (p < 0.05)."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return False
    # Look for common GxE terms
    gxe_terms = ["genotype:environment", "environment:genotype", "GxE", "gxe"]
    for term in gxe_terms:
        try:
            idx = anova_table.source.index(term)
            p_val = anova_table.p_value[idx]
            if p_val is not None and p_val < 0.05:
                return True
        except (ValueError, IndexError):
            continue
    return False


@router.post(
    "/analysis/anova",
    response_model=AnovaModuleResponse,
    summary="Run ANOVA analysis for selected traits",
)
async def analysis_anova(request: ModuleRequest):
    """
    For each requested trait column, run the R genetics engine and return
    the ANOVA-specific slice of the result:

      anova_table, descriptive_stats, assumption_tests,
      mean_separation, interpretation, data_warnings

    Requires a dataset_token issued by POST /upload/dataset.
    If another module already ran R for a trait under the same token,
    the result is read from cache — no duplicate R call.
    """
    import app_genetics  # lazy: r_engine assigned on startup
    if app_genetics.r_engine is None:
        raise HTTPException(status_code=503, detail="R genetics engine not ready")
    r_engine = app_genetics.r_engine

    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset token '{request.dataset_token}' not found. "
                "Re-upload via POST /upload/dataset to get a new token."
            ),
        )

    try:
        file_bytes = base64.b64decode(ctx["base64_content"])
        df = read_file(file_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc

    missing = [c for c in request.trait_columns if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Trait columns not found in dataset: {missing}"
        )

    mode           = ctx["mode"]
    env_col        = ctx["environment_column"] if mode == "multi" else None
    geno_col       = ctx["genotype_column"]
    rep_col        = ctx["rep_column"]        # may be None for CRD datasets
    factor_col     = ctx.get("factor_column") if mode == "single" else None
    main_plot_col  = ctx.get("main_plot_column")
    sub_plot_col   = ctx.get("sub_plot_column")
    design_type    = ctx.get("design_type")
    random_env     = ctx["random_environment"]
    # CRD: no explicit rep column AND single-environment mode
    crd_mode       = (rep_col is None) and (mode == "single")

    trait_results: Dict[str, AnovaTraitResult] = {}
    failed_traits: List[str] = []

    MAX_CONCURRENT_R_PROCESSES = 4
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_R_PROCESSES)

    async def process_trait(trait: str):
        try:
            # ── Cache check ──────────────────────────────────────────────────
            cached: Optional[GeneticsResponse] = dataset_cache.get_analysis(
                request.dataset_token, trait
            )

            if cached is None:
                async with semaphore:
                    balance_warnings = check_balance(
                        df,
                        geno_col,
                        rep_col,
                        trait,
                        env_col,
                        factor_col=factor_col,
                        design_type=design_type,
                        main_plot_col=main_plot_col,
                        sub_plot_col=sub_plot_col,
                    )
                    observations = build_observations(
                        df,
                        geno_col,
                        rep_col,
                        trait,
                        env_col,
                        factor_col=factor_col,
                        design_type=design_type,
                        main_plot_col=main_plot_col,
                        sub_plot_col=sub_plot_col,
                    )

                    result_dict = await asyncio.to_thread(
                        r_engine.run_analysis,
                        data=observations,
                        mode=mode,
                        trait_name=trait,
                        random_environment=random_env,
                        crd_mode=crd_mode,
                    )

                if result_dict.get("status") == "ERROR":
                    r_errors = result_dict.get("errors") or {}
                    raise RuntimeError(
                        result_dict.get("interpretation")
                        or next(iter(r_errors.values()), None)
                        or "R analysis returned ERROR"
                    )

                cached = GeneticsResponse(**result_dict)
                dataset_cache.put_analysis(request.dataset_token, trait, cached)
            else:
                balance_warnings = []

            res = cached.result
            if res is None:
                raise RuntimeError("R returned status OK but result object is empty")

            trait_descriptive_stats = compute_descriptive_stats(df[trait])
            # Per-genotype stats only make sense when a genotype column exists
            per_genotype_stats = (
                compute_per_genotype_stats(df, trait, geno_col)
                if geno_col and design_type != "split_plot_rcbd"
                else []
            )

            # Prefer ANOVA-derived CV (sqrt(MSE)/grand_mean*100) over the
            # raw-observation SD-based CV when the ANOVA table is available.
            anova_cv = compute_cv_from_anova(res.anova_table, res.grand_mean)
            if anova_cv is not None:
                trait_descriptive_stats["cv_percent"] = anova_cv

            # Build summary from descriptive stats
            summary = {
                "grand_mean": trait_descriptive_stats["grand_mean"],
                "cv_percent": trait_descriptive_stats["cv_percent"],
                "min": trait_descriptive_stats["min"],
                "max": trait_descriptive_stats["max"],
                "range": trait_descriptive_stats["range"],
                "standard_error": trait_descriptive_stats["standard_error"],
            }

            # Classify precision and flags
            precision_level = classify_precision_level(trait_descriptive_stats["cv_percent"])
            cv_interpretation_flag = get_cv_interpretation_flag(trait_descriptive_stats["cv_percent"])
            genotype_significant = is_genotype_effect_significant(res.anova_table)
            environment_significant = is_environment_effect_significant(res.anova_table)
            gxe_significant = is_gxe_effect_significant(res.anova_table)
            # ranking_caution follows directly from GxE significance
            ranking_caution = gxe_significant
            selection_feasible = genotype_significant

            # Split-plot specific significance flags (only populated for split_plot_rcbd)
            mp_significant  = is_main_plot_significant(res.anova_table)  if design_type == "split_plot_rcbd" else None
            sub_significant = is_subplot_significant(res.anova_table)    if design_type == "split_plot_rcbd" else None
            int_significant = is_interaction_significant(res.anova_table) if design_type == "split_plot_rcbd" else None

            # Generate ANOVA interpretation — design-type-aware
            anova_interpretation = generate_anova_interpretation(
                trait=trait,
                summary=summary,
                precision_level=precision_level,
                cv_interpretation_flag=cv_interpretation_flag,
                genotype_significant=genotype_significant,
                environment_significant=environment_significant,
                gxe_significant=gxe_significant,
                ranking_caution=ranking_caution,
                selection_feasible=selection_feasible,
                mean_separation=res.mean_separation,
                n_genotypes=res.n_genotypes,
                n_environments=res.n_environments,
                n_reps=res.n_reps,
                environment_mode=mode,
                design_type=design_type,
                main_plot_significant=mp_significant,
                subplot_significant=sub_significant,
                interaction_significant=int_significant,
            )
            logger.info(
                "ANOVA interpretation generated for trait '%s': %d characters",
                trait, len(anova_interpretation) if anova_interpretation else 0
            )

            result_obj = AnovaTraitResult(
                trait=trait,
                status="success",
                grand_mean=res.grand_mean,
                n_genotypes=res.n_genotypes,
                n_reps=res.n_reps,
                n_environments=res.n_environments,
                anova_table=res.anova_table,
                descriptive_stats=trait_descriptive_stats,
                per_genotype_stats=per_genotype_stats,
                summary=summary,
                precision_level=precision_level,
                cv_interpretation_flag=cv_interpretation_flag,
                ranking_caution=ranking_caution,
                selection_feasible=selection_feasible,
                genotype_significant=genotype_significant,
                environment_significant=environment_significant,
                gxe_significant=gxe_significant,
                assumption_tests=res.assumption_tests,
                mean_separation=res.mean_separation,
                interpretation=anova_interpretation,
                data_warnings=balance_warnings,
            )
            return trait, "success", result_obj

        except Exception as exc:
            logger.warning("ANOVA: trait '%s' failed — %s", trait, exc)
            result_obj = AnovaTraitResult(
                trait=trait,
                status="failed",
                error=str(exc),
            )
            return trait, "failed", result_obj

    # Execute all trait analyses concurrently
    tasks = [process_trait(trait) for trait in request.trait_columns]
    results = await asyncio.gather(*tasks)

    for trait, status, result_obj in results:
        trait_results[trait] = result_obj
        if status == "failed":
            failed_traits.append(trait)
        else:
            logger.info(
                "ANOVA response: trait '%s' has interpretation = %s",
                trait, "YES" if result_obj.interpretation else "NO"
            )

    logger.info(
        "ANOVA endpoint returning %d successful traits, %d failed traits",
        len(trait_results) - len(failed_traits), len(failed_traits)
    )

    return AnovaModuleResponse(
        dataset_token=request.dataset_token,
        mode=mode,
        trait_results=trait_results,
        failed_traits=failed_traits,
    )
