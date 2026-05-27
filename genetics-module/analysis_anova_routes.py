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
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from genetics_schemas import GeneticsResponse
from multitrait_upload_routes import build_observations, check_balance, read_file
from module_schemas import AnovaModuleResponse, AnovaTraitResult, ModuleRequest, AnalysisContext
import dataset_cache
from analysis_utils import compute_descriptive_stats, compute_per_genotype_stats, classify_precision_level
from academic_interpretation import detect_analysis_domain
import math

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

FAILED_TRAIT_CV_MESSAGE = "CV% unavailable due to failed ANOVA estimation."


def _get_ms_error_a(anova_table) -> Optional[float]:
    if anova_table is None or not hasattr(anova_table, "source") or not hasattr(anova_table, "ms"):
        return None
    # Match labels like "Rep:MainPlot", "Error A", or "Rep × A"
    error_a_labels = ["Rep:MainPlot", "Error A", "Rep × A", "rep:main_plot"]
    for label in error_a_labels:
        try:
            idx = anova_table.source.index(label)
            return float(anova_table.ms[idx]) if anova_table.ms[idx] is not None else None
        except (ValueError, IndexError, TypeError):
            continue
    return None


def _get_ms_error_b(anova_table) -> Optional[float]:
    if anova_table is None or not hasattr(anova_table, "source") or not hasattr(anova_table, "ms"):
        return None
    # Match labels like "Residual", "Error B"
    error_b_labels = ["Error B", "Residuals", "residuals", "Residual", "residual", "error", "Error"]
    for label in error_b_labels:
        try:
            idx = anova_table.source.index(label)
            return float(anova_table.ms[idx]) if anova_table.ms[idx] is not None else None
        except (ValueError, IndexError, TypeError):
            continue
    return None


def _sanitize_cv_percent(cv: Optional[float]) -> Optional[float]:
    if cv is None:
        return None
    try:
        value = abs(float(cv))
    except (TypeError, ValueError):
        return None
    return value


def _precision_label_from_cv(cv: Optional[float]) -> Optional[str]:
    cv_val = _sanitize_cv_percent(cv)
    if cv_val is None:
        return None
    if cv_val < 10:
        return "good"
    if cv_val < 20:
        return "moderate"
    return "low"


def _map_precision_level(precision_level: Optional[str], cv: Optional[float]) -> Optional[str]:
    if precision_level in {"good", "moderate", "low"}:
        return precision_level
    if precision_level == "high" or precision_level == "good":
        return "good"
    if precision_level == "acceptable" or precision_level == "moderate":
        return "moderate"
    if precision_level == "caution" or precision_level == "low":
        return "low"
    return classify_precision_level(cv)


def _cv_tier_narrative(cv: Optional[float], domain: str = "general") -> Optional[str]:
    cv_val = _sanitize_cv_percent(cv)
    if cv_val is None:
        return None
    if cv_val < 10:
        base = "Residual variability was relatively low, suggesting high experimental precision within the scope of this design."
    elif cv_val < 20:
        from domain_guard import is_plant_breeding_domain
        base = (
            "Experimental variability appeared acceptable for genotype comparison under the evaluated conditions."
            if is_plant_breeding_domain(domain)
            else "Experimental variability appeared acceptable for treatment comparison under the evaluated conditions."
        )
    else:
        base = "Residual variability was comparatively high, and findings should therefore be interpreted cautiously."
    if cv_val < 1:
        base += " Residual variability was extremely low relative to the trait mean. Verify raw data consistency and experimental realism."
    return base


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

    mse = _get_ms_error_b(anova_table)
    if mse is not None and mse >= 0 and grand_mean != 0:
        return float((mse ** 0.5) / grand_mean * 100)
    return None


def _fmt_factor_label(raw: Optional[str], fallback: str) -> str:
    """Format a column name (e.g. 'main_plot', 'sub_plot', 'irrigation') for prose."""
    if not raw:
        return fallback
    return raw.replace("_", " ").title()


def _generate_split_plot_interpretation(
    trait: str,
    summary: Dict[str, Optional[float]],
    precision_level: Optional[str],
    cv_interpretation_flag: Optional[str],
    main_plot_significant: Optional[bool],
    subplot_significant: Optional[bool],
    interaction_significant: Optional[bool],
    n_reps: Optional[int],
    cv_a: Optional[float] = None,
    cv_b: Optional[float] = None,
    mean_separation: Optional[Any] = None,
    main_plot_mean_separation: Optional[Any] = None,
    mp_label: Optional[str] = None,
    sp_label: Optional[str] = None,
) -> str:
    """
    Scientifically rigorous split-plot RCBD interpretation.

    Implements the Interaction Priority Rule: when the A×B interaction is
    significant, it is reported first and main effects are flagged as
    conditional rather than unconditional. Uses dual CV (CV_A for whole-plot
    precision, CV_B for subplot precision) when both are available.

    mp_label / sp_label: actual factor names (e.g. "Irrigation", "Variety").
    Falls back to generic "main-plot factor" / "subplot factor" if not provided.
    """
    # Resolve actual factor labels for scientific specificity
    MP = _fmt_factor_label(mp_label, "main-plot factor")
    SP = _fmt_factor_label(sp_label, "subplot factor")

    sections: List[tuple] = []

    # ── 1. Overview ────────────────────────────────────────────────────────────
    overview = [
        f"This analysis used a split-plot randomised complete block design (RCBD), "
        f"with {MP} assigned to main plots (whole plots) and {SP} assigned to subplots "
        f"within each whole plot. "
        "Two levels of restricted randomisation were applied. "
        "This design structure creates two error terms: Error A (whole-plot error, Rep × main plot) "
        f"for testing {MP}, and Error B (subplot error) for testing "
        f"{SP} and the {MP} × {SP} interaction."
    ]
    if n_reps:
        overview.append(f"The experiment comprised {n_reps} complete block(s).")
    if summary.get("grand_mean") is not None:
        overview.append(
            f"The overall grand mean for {trait} across all treatment combinations "
            f"was {summary['grand_mean']:.2f}."
        )
    sections.append(("Overview", " ".join(overview)))

    # ── 2. Statistical Model ────────────────────────────────────────────────────
    model_desc = [
        f"The split-plot ANOVA model for {trait} partitions variability into: "
        f"Replication (block effect), {MP} (A), whole-plot error (Rep × {MP}), "
        f"{SP} (B), {MP} × {SP} interaction, and subplot error (Residual). "
        f"{MP} is tested against Rep × {MP} (Error A), while "
        f"{SP} and the {MP} × {SP} interaction are tested against the residual error (Error B). "
        "This two-tier error structure reflects the nested randomisation of the design."
    ]
    sections.append(("Statistical Model", " ".join(model_desc)))

    # ── 3. Experimental Precision (Dual CV) ────────────────────────────────────
    precision_parts = []
    cv_a = _sanitize_cv_percent(cv_a)
    cv_b = _sanitize_cv_percent(cv_b)

    if cv_a is not None and cv_b is not None:
        prec_a = _precision_label_from_cv(cv_a)
        prec_b = _precision_label_from_cv(cv_b)
        precision_parts.append(
            f"The whole-plot coefficient of variation (CV_A = √(MS_Error_A)/mean × 100) "
            f"was {cv_a:.2f}%, indicating {prec_a} precision for main-plot comparisons."
        )
        precision_parts.append(
            f"The subplot coefficient of variation (CV_B = √(MS_Error_B)/mean × 100) "
            f"was {cv_b:.2f}%, indicating {prec_b} precision for subplot and interaction comparisons."
        )
        cv_b_narrative = _cv_tier_narrative(cv_b)
        if cv_b_narrative:
            precision_parts.append(cv_b_narrative)
        if prec_a != prec_b:
            precision_parts.append(
                "The two CVs reflect the separate error strata of the split-plot design. "
                "It is normal for CV_A and CV_B to differ due to the different experimental unit sizes; "
                "whole-plot error typically has larger variability than subplot error."
            )
    elif cv_b is not None:
        prec_b = _precision_label_from_cv(cv_b)
        precision_parts.append(
            f"The subplot coefficient of variation (CV_B) was {cv_b:.2f}%, "
            f"indicating {prec_b} precision for subplot-level comparisons."
        )
        cv_b_narrative = _cv_tier_narrative(cv_b)
        if cv_b_narrative:
            precision_parts.append(cv_b_narrative)
    elif cv_a is not None:
        prec_a = _precision_label_from_cv(cv_a)
        precision_parts.append(
            f"The whole-plot coefficient of variation (CV_A) was {cv_a:.2f}%, "
            f"indicating {prec_a} precision for main-plot comparisons."
        )
        cv_a_narrative = _cv_tier_narrative(cv_a)
        if cv_a_narrative:
            precision_parts.append(cv_a_narrative)
    elif cv_interpretation_flag == "cv_available" and summary.get("cv_percent") is not None:
        cv = _sanitize_cv_percent(summary["cv_percent"])
        prec = _precision_label_from_cv(cv)
        precision_parts.append(f"The coefficient of variation was {cv:.2f}% ({prec} experimental precision).")
        cv_narrative = _cv_tier_narrative(cv)
        if cv_narrative:
            precision_parts.append(cv_narrative)
    else:
        precision_parts.append("CV% was unavailable, so residual precision could not be interpreted for this trait.")
    sections.append(("Experimental Precision", " ".join(precision_parts)))

    # ── 4–6. Treatment Effects — Interaction Priority Rule ─────────────────────
    # When A×B interaction is significant, report it first and frame main effects
    # as conditional. When interaction is non-significant, report main effects first.
    if interaction_significant is True:
        int_text = (
            f"A statistically significant {MP} × {SP} interaction was detected for {trait} (p < 0.05), "
            f"indicating that the effect of {SP} depends on the level of {MP}, and vice versa. "
            "When the interaction is significant, interpretation of marginal (main) effects in isolation "
            "is insufficient and can be misleading. Treatment-combination cell means "
            f"(each unique {MP} × {SP} combination) should be the primary basis for biological "
            "and agronomic conclusions. Inspect the interaction plot and cell mean table to characterise "
            "the nature of the interaction."
        )
        sections.append((f"{MP} × {SP} Interaction Effect (Primary)", int_text))

        if main_plot_significant is True:
            mp_text = (
                f"{MP} showed a significant marginal effect on {trait} (p < 0.05). "
                f"However, because the {MP} × {SP} interaction is also significant, this effect should "
                f"be interpreted as an average across {SP} levels and may not represent the true "
                f"response pattern at any specific {SP} level. {MP} marginal means are conditional "
                f"on the {SP} level."
            )
        elif main_plot_significant is False:
            mp_text = (
                f"{MP} did not show a significant marginal effect on {trait} (p ≥ 0.05). "
                f"However, the significant {MP} × {SP} interaction indicates that differential responses "
                f"to {MP} exist when examined at specific {SP} levels, even though the overall "
                "average effect is not significant."
            )
        else:
            mp_text = f"The significance of the {MP} effect could not be determined."
        sections.append((f"{MP} Effect (Conditional on Interaction)", mp_text))

        if subplot_significant is True:
            sub_text = (
                f"{SP} showed a significant marginal effect on {trait} (p < 0.05). "
                f"As with the {MP} effect, this should be interpreted as an average across "
                f"{MP} levels and is conditional on the significant interaction. The true "
                f"{SP} effect varies depending on which {MP} level is examined."
            )
        elif subplot_significant is False:
            sub_text = (
                f"{SP} did not show a significant marginal effect on {trait} (p ≥ 0.05). "
                f"Despite this, the significant {MP} × {SP} interaction confirms that {SP} responses "
                f"are not uniform across all {MP} levels — specific {MP} × {SP} combinations "
                "produce differential effects."
            )
        else:
            sub_text = f"The significance of the {SP} effect could not be determined."
        sections.append((f"{SP} Effect (Conditional on Interaction)", sub_text))

    else:
        # No significant interaction — additive effects; main effects interpretable independently
        if interaction_significant is False:
            sections.append((
                f"{MP} × {SP} Interaction Effect",
                f"No statistically significant {MP} × {SP} interaction was detected for {trait} (p ≥ 0.05), "
                f"indicating that {MP} and {SP} have additive effects. Each factor can be interpreted "
                "independently — the effect of one factor does not depend on the level of the other. "
                "Marginal means for each factor provide adequate summaries of the treatment effects."
            ))

        if main_plot_significant is True:
            mp_text = (
                f"{MP} had a statistically significant effect on {trait} (p < 0.05), "
                f"indicating that different levels of {MP} produced meaningfully different responses. "
                f"Given the absence of a significant interaction, this effect is consistent across all {SP} levels."
            )
        elif main_plot_significant is False:
            mp_text = (
                f"{MP} did not have a statistically significant effect on {trait} "
                f"(p ≥ 0.05), suggesting that the evaluated levels of {MP} produced similar responses on average."
            )
        else:
            mp_text = f"The significance of the {MP} effect on {trait} could not be determined."
        sections.append((f"{MP} Effect", mp_text))

        if subplot_significant is True:
            sub_text = (
                f"{SP} had a statistically significant effect on {trait} (p < 0.05), "
                f"indicating that different levels of {SP} produced meaningfully different responses. "
                f"Given the absence of a significant interaction, this effect is consistent across all {MP} levels."
            )
        elif subplot_significant is False:
            sub_text = (
                f"{SP} did not have a statistically significant effect on {trait} "
                f"(p ≥ 0.05), suggesting that the evaluated levels of {SP} produced similar responses on average."
            )
        else:
            sub_text = f"The significance of the {SP} effect on {trait} could not be determined."
        sections.append((f"{SP} Effect", sub_text))

    # ── 7. Mean Separation Summary ─────────────────────────────────────────────
    sep_parts = []
    if (
        main_plot_mean_separation is not None
        and hasattr(main_plot_mean_separation, "genotype")
        and main_plot_mean_separation.genotype
    ):
        try:
            top_mp      = main_plot_mean_separation.genotype[0]
            top_mp_mean = main_plot_mean_separation.mean[0]
            top_mp_grp  = main_plot_mean_separation.group[0] if main_plot_mean_separation.group else "—"
            sep_parts.append(
                f"{MP} mean separation (Fisher LSD, α = 0.05, tested against Error A): "
                f"'{top_mp}' had the highest mean {trait} ({top_mp_mean:.2f}, group '{top_mp_grp}'). "
                f"These marginal means average across all {SP} levels."
            )
        except (IndexError, TypeError):
            sep_parts.append(f"{MP} mean separation was performed but could not be summarised.")

    if (
        mean_separation is not None
        and hasattr(mean_separation, "genotype")
        and mean_separation.genotype
    ):
        try:
            top_sub      = mean_separation.genotype[0]
            top_sub_mean = mean_separation.mean[0]
            top_sub_grp  = mean_separation.group[0] if mean_separation.group else "—"
            sep_parts.append(
                f"{SP} mean separation (Fisher LSD, α = 0.05, tested against Error B): "
                f"'{top_sub}' had the highest mean {trait} ({top_sub_mean:.2f}, group '{top_sub_grp}'). "
                f"These marginal means average across all {MP} levels."
            )
        except (IndexError, TypeError):
            sep_parts.append(f"{SP} mean separation was performed but could not be summarised.")

    if interaction_significant is True:
        sep_parts.append(
            f"When the {MP} × {SP} interaction is significant, cell means (each unique {MP} × {SP} "
            "combination) are more informative than marginal means. Consult the interaction plot "
            "and cell mean table to identify the optimal treatment combinations for this trait."
        )

    if not sep_parts:
        sep_parts.append("Mean separation results were not available.")
    sections.append(("Mean Separation Summary", " ".join(sep_parts)))

    # ── 8. Risk and Limitations ────────────────────────────────────────────────
    risks = []
    cv_a_val = cv_a
    cv_b_val = cv_b or summary.get("cv_percent")
    if (cv_a_val is not None and cv_a_val > 20) or (cv_b_val is not None and cv_b_val > 20):
        risks.append(
            "One or both CVs exceed 20%, indicating elevated experimental variability. "
            "This may reflect heterogeneity in experimental units, environmental gradients, "
            "or insufficient replication. High variability reduces statistical power and "
            "increases the risk of Type II errors (failing to detect real treatment differences)."
        )
    if interaction_significant is True:
        risks.append(
            f"The significant {MP} × {SP} interaction complicates interpretation of marginal factor means. "
            "Conclusions based solely on main effects can be misleading when the interaction is present. "
            f"Treatment-combination cell means (all {MP} × {SP} combinations) and the interaction plot "
            "must be the primary basis for applied conclusions and management recommendations."
        )
    if not risks:
        risks.append(
            "Results should be interpreted within the scope of this experiment, "
            "including the evaluated environment and replication structure. "
            "The design structure and precision levels support reliable inference within these constraints."
        )
    sections.append(("Risk and Limitations", " ".join(risks)))

    # ── 9. Recommendation ─────────────────────────────────────────────────────
    recs = []
    if interaction_significant is True:
        recs.append(
            f"Examine {MP} × {SP} treatment-combination cell means rather than marginal factor means — "
            "main effects alone are insufficient when the interaction is significant. "
            "Interpret the interaction plot alongside treatment-combination cell means to identify "
            f"which specific {MP} × {SP} combinations produce the strongest response for {trait}."
        )
    recs.append(
        "Use Error A (whole-plot error) for pairwise comparisons among main-plot factor levels "
        "and Error B (subplot error) for comparisons among subplot factor levels and for all "
        "interaction contrasts. This correctly reflects the two-stratum error structure of the "
        "split-plot design."
    )
    if (cv_a_val is not None and cv_a_val > 20) or (cv_b_val is not None and cv_b_val > 20):
        recs.append(
            "Consider increasing replication, improving blocking strategies, or refining "
            "experimental technique to reduce unexplained variability in future trials. "
            "Lower CVs will increase precision and statistical power."
        )
    recs.append(
        f"Integrate these ANOVA results with domain-specific knowledge about {trait} and practical "
        "relevance thresholds (minimum meaningful difference) before drawing applied conclusions "
        "or making management recommendations."
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
    # Split-plot specific parameters (only used when design_type == "split_plot_rcbd")
    main_plot_significant: Optional[bool] = None,
    subplot_significant: Optional[bool] = None,
    interaction_significant: Optional[bool] = None,
    cv_a: Optional[float] = None,
    cv_b: Optional[float] = None,
    main_plot_mean_separation: Optional[Any] = None,
    domain: str = "general",
    mp_label: Optional[str] = None,
    sp_label: Optional[str] = None,
) -> str:
    """
    Generate context-aware ANOVA interpretation following VivaSense standards.

    For design_type="split_plot_rcbd" dispatches to a domain-neutral split-plot
    interpretation using actual factor names when mp_label/sp_label are supplied,
    falling back to role-based language (main-plot factor / subplot factor).
    All other designs use the genetics-aware interpretation path.

    environment_mode="single"
        Sections: Overview, Descriptive Interpretation, Genotype Effect,
                  Mean Performance and Ranking, Experimental Interpretation,
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
            cv_a=cv_a,
            cv_b=cv_b,
            mean_separation=mean_separation,
            main_plot_mean_separation=main_plot_mean_separation,
            mp_label=mp_label,
            sp_label=sp_label,
        )

    is_multi = environment_mode == "multi"
    # Domain-aware terminology
    if domain == "plant_breeding":
        _term = "genotype"
        _terms = "genotypes"
        _effect_label = "Genotype Effect"
        _section_label = "Breeding Interpretation"
    else:
        _term = "treatment"
        _terms = "treatments"
        _effect_label = "Treatment Effect"
        _section_label = "Research Interpretation"
    # List of (heading, content) tuples built up below
    sections: List[tuple] = []

    # ── 1. Overview ────────────────────────────────────────────────────────────
    overview = []
    if n_genotypes and n_reps:
        if is_multi and n_environments and n_environments > 1:
            overview.append(
                f"This analysis evaluated {trait} across {n_genotypes} {_terms} "
                f"tested in {n_environments} environments with {n_reps} replications "
                f"per {_term}-environment combination."
            )
        else:
            overview.append(
                f"This analysis evaluated {trait} across {n_genotypes} {_terms} "
                f"with {n_reps} replications per {_term}."
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
        cv = _sanitize_cv_percent(summary["cv_percent"])
        if cv is not None:
            precision_word = _precision_label_from_cv(cv)
            overview.append(
                f"The coefficient of variation (CV) was {cv:.2f}%, indicating "
                f"{precision_word} experimental precision."
            )
        else:
            overview.append(
                "CV% was unavailable, so residual precision could not be interpreted for this trait."
            )
    else:
        overview.append(
            "CV% was unavailable, so residual precision could not be interpreted for this trait."
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
        if cv_interpretation_flag == "cv_available" and summary.get("grand_mean") is not None:
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
        else:
            desc.append(
                f"Performance ranged from {summary['min']:.2f} to {summary['max']:.2f}, "
                f"with a total range of {summary['range']:.2f}."
            )

    if cv_interpretation_flag == "cv_available" and summary.get("cv_percent") is not None:
        cv_narrative = _cv_tier_narrative(summary.get("cv_percent"), domain=domain)
        if cv_narrative:
            desc.append(cv_narrative)
    elif cv_interpretation_flag == "cv_unavailable":
        desc.append(
            "CV% was unavailable, so residual precision could not be interpreted for this trait."
        )

    sections.append(("Descriptive Interpretation", " ".join(desc)))

    # ── 3. Genotype Effect ─────────────────────────────────────────────────────
    # ── 3. Genotype / Treatment Effect ──────────────────────────────────────
    if domain == "plant_breeding":
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
    else:
        if genotype_significant is True:
            geno_text = (
                f"Significant differences among {_terms} were detected for {trait} "
                f"(p < 0.05), indicating that the tested {_terms} vary in their "
                f"effect on {trait}."
            )
        elif genotype_significant is False:
            geno_text = (
                f"No significant differences among {_terms} were detected for {trait}, "
                f"suggesting that the tested {_terms} do not differ sufficiently in "
                "their effect on this variable."
            )
        else:
            geno_text = (
                f"The significance of the {_term} effect on {trait} could not be determined."
            )
    sections.append((_effect_label, geno_text))

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
            if domain == "plant_breeding":
                gxe_text = (
                    f"A significant genotype \u00d7 environment interaction was detected "
                    f"for {trait}, indicating that genotype performance is not consistent "
                    "across environments. This suggests that no single genotype is "
                    "universally superior, and selection strategies should account for "
                    "environmental stability."
                )
            else:
                gxe_text = (
                    f"A significant treatment \u00d7 environment interaction was detected "
                    f"for {trait}, indicating that treatment effects are not consistent "
                    "across environments. Treatment performance should be evaluated "
                    "within the specific environmental context tested."
                )
        elif gxe_significant is False:
            if domain == "plant_breeding":
                gxe_text = (
                    f"No significant genotype \u00d7 environment interaction was detected "
                    f"for {trait}, suggesting relatively stable genotype performance across "
                    "the tested environments."
                )
            else:
                gxe_text = (
                    f"No significant treatment \u00d7 environment interaction was detected "
                    f"for {trait}, suggesting relatively consistent treatment effects across "
                    "the tested environments."
                )
        else:
            gxe_text = (
                f"The presence of interaction between treatments and environments for {trait} "
                "could not be determined."
            )
        sections.append(("G\u00d7E Interaction" if domain == "plant_breeding" else "Treatment \u00d7 Environment Interaction", gxe_text))

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
    # ── 7 (single: 5). Breeding / Research Interpretation ────────────────────
    breeding = []
    if domain == "plant_breeding":
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
    else:
        if selection_feasible is True:
            breeding.append(
                f"Significant differences among {_terms} indicate that the tested "
                f"levels vary meaningfully in their effect on {trait}."
            )
            if is_multi:
                if gxe_significant is False:
                    breeding.append(
                        f"Given the absence of significant {_term} \u00d7 environment "
                        "interaction, results are consistent across the tested environments."
                    )
                else:
                    breeding.append(
                        f"Due to significant {_term} \u00d7 environment interaction, "
                        f"the effect of {_terms} varies across environments; "
                        "environment-specific recommendations should be considered."
                    )
        else:
            breeding.append(
                f"No significant differences among {_terms} indicate that the tested "
                f"levels do not differ sufficiently in their effect on {trait}."
            )
        breeding.append(
            "The observed variability and experimental precision should guide the "
            "design of future experiments."
        )
    sections.append((_section_label, " ".join(breeding)))

    # ── 8 (single: 6). Risk and Limitations ───────────────────────────────────
    risks = []
    if is_multi and gxe_significant is True:
        if domain == "plant_breeding":
            risks.append(
                "The significant genotype \u00d7 environment interaction represents a "
                "major limitation, as it complicates genotype evaluation and "
                "environment-specific inference."
            )
        else:
            risks.append(
                "The significant treatment \u00d7 environment interaction represents a "
                "major limitation, as treatment effects differ across environments and "
                "broad generalisation of results requires caution."
            )
    if cv_interpretation_flag == "cv_available" and precision_level == "low":
        risks.append(
            "The caution-level experimental precision introduces uncertainty in the results "
            "and suggests potential issues with experimental control or replication."
        )
    if is_multi and environment_significant is True:
        risks.append(
            "Strong environmental influence may limit the generalisability of these "
            "results to other locations or conditions."
        )
    # Always acknowledge experimental scope — never claim no limitations
    risks.append(
        "Results should be interpreted within the scope of this experiment, "
        "including the evaluated environment and replication structure."
    )
    sections.append(("Risk and Limitations", " ".join(risks)))

    # ── 9 (single: 7). Recommendation ─────────────────────────────────────────
    recs = []
    if is_multi and gxe_significant is True:
        if domain == "plant_breeding":
            recs.append(
                "Conduct stability analysis (e.g., AMMI or GGE biplot) to identify "
                "genotypes with consistent performance across environments."
            )
        else:
            recs.append(
                "Conduct stability analysis to identify treatments with consistent "
                "performance across the tested environments."
            )
    if selection_feasible is True:
        if domain == "plant_breeding":
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
        else:
            recs.append(
                "Further evaluation across additional environments and management conditions "
                "may help validate treatment consistency."
            )
    if cv_interpretation_flag == "cv_available" and precision_level == "low":
        recs.append(
            "Improve experimental design by increasing replication or enhancing "
            "environmental control to reduce variability."
        )
    recs.append(
        "Integrate these ANOVA results with genetic parameter estimates "
        "(heritability, genetic coefficient of variation) for comprehensive "
        "trait evaluation."
    ) if domain == "plant_breeding" else recs.append(
        "Integrate these ANOVA results with other relevant analyses for "
        "comprehensive trait evaluation."
    )
    sections.append(("Recommendation", " ".join(recs)))

    return "\n\n".join(f"{heading}\n{content}" for heading, content in sections)


def get_cv_interpretation_flag(cv_percent: Optional[float]) -> str:
    """Return flag indicating if CV is available for interpretation."""
    return "cv_available" if cv_percent is not None else "cv_unavailable"


def _is_term_significant(anova_table, term: str) -> Optional[bool]:
    """Return True/False if term is in the ANOVA table, else None."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return None
    try:
        # Strip whitespace from source labels — R rownames can have trailing spaces
        stripped_sources = [str(s).strip() for s in anova_table.source]
        idx = stripped_sources.index(term.strip())
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
async def analysis_anova(request: ModuleRequest, http_request: Request):
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
        df, _ = read_file(file_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc

    analysis_domain = detect_analysis_domain(list(df.columns), "anova")

    missing = [c for c in request.trait_columns if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Trait columns not found in dataset: {missing}"
        )

    mode           = ctx["mode"]
    if mode == "multi":
        mode_header = (http_request.headers.get("X-VivaSense-Mode") or "free").lower().strip()
        if mode_header != "pro":
            return JSONResponse(
                status_code=403,
                content={
                    "error": "PRO_FEATURE",
                    "message": "Upgrade to access this feature",
                },
            )
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
                trait_descriptive_stats["cv_percent"] = _sanitize_cv_percent(anova_cv)

            trait_descriptive_stats["cv_percent"] = _sanitize_cv_percent(
                trait_descriptive_stats.get("cv_percent")
            )

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
            precision_level_raw = classify_precision_level(trait_descriptive_stats["cv_percent"])
            precision_level = _map_precision_level(
                precision_level_raw,
                trait_descriptive_stats["cv_percent"],
            )
            cv_interpretation_flag = get_cv_interpretation_flag(trait_descriptive_stats["cv_percent"])
            genotype_significant = is_genotype_effect_significant(res.anova_table)
            environment_significant = is_environment_effect_significant(res.anova_table)
            gxe_significant = is_gxe_effect_significant(res.anova_table)
            # ranking_caution follows directly from GxE significance
            ranking_caution = gxe_significant
            selection_feasible = genotype_significant

            # Split-plot specific significance flags and dual CV
            is_sp = design_type == "split_plot_rcbd"
            mp_significant  = is_main_plot_significant(res.anova_table)   if is_sp else None
            sub_significant = is_subplot_significant(res.anova_table)     if is_sp else None
            int_significant = is_interaction_significant(res.anova_table) if is_sp else None

            # Extract MS error values and calculate per-stratum CV% for split-plot
            ms_error_a = _get_ms_error_a(res.anova_table)
            ms_error_b = _get_ms_error_b(res.anova_table)
            cv_main_plot_pct = None
            cv_sub_plot_pct = None

            if is_sp and res.grand_mean and res.grand_mean != 0:
                if ms_error_a is not None and ms_error_a >= 0:
                    cv_main_plot_pct = round((math.sqrt(ms_error_a) / res.grand_mean) * 100, 2)
                if ms_error_b is not None and ms_error_b >= 0:
                    cv_sub_plot_pct = round((math.sqrt(ms_error_b) / res.grand_mean) * 100, 2)

            analysis_ctx = AnalysisContext(
                is_single_environment=True,
                environment_count=1,
                design_type=design_type
            )

            # Extract dual CV from variance_components for split-plot
            if is_sp and isinstance(res.variance_components, dict):
                vc = res.variance_components
                cv_a = _sanitize_cv_percent(float(vc["cv_A"])) if vc.get("cv_A") is not None else None
                cv_b = _sanitize_cv_percent(float(vc["cv_B"])) if vc.get("cv_B") is not None else None
                # Override summary CV with the more precise CV_B for split-plot
                if cv_b is not None:
                    summary["cv_percent"] = cv_b
                    trait_descriptive_stats["cv_percent"] = cv_b
            else:
                cv_a = None
                cv_b = None

            # Extract actual factor labels for named interpretation (split-plot only)
            _mp_lbl = getattr(res.main_plot_mean_separation, "treatment_label", None) if is_sp else None
            _sp_lbl = getattr(res.mean_separation, "treatment_label", None) if is_sp else None

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
                cv_a=cv_a,
                cv_b=cv_b,
                main_plot_mean_separation=res.main_plot_mean_separation if is_sp else None,
                domain=analysis_domain,
                mp_label=_mp_lbl,
                sp_label=_sp_lbl,
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
                main_plot_mean_separation=res.main_plot_mean_separation if is_sp else None,
                interaction_means=res.interaction_means if is_sp and hasattr(res, 'interaction_means') else None,
                cv_a=cv_a,
                cv_b=cv_b,
                interpretation=anova_interpretation,
                data_warnings=balance_warnings,
                design_type=design_type,
                cv_main_plot_pct=cv_main_plot_pct,
                cv_sub_plot_pct=cv_sub_plot_pct,
                ms_error_a=ms_error_a,
                ms_error_b=ms_error_b,
                analysis_context=analysis_ctx,
            )
            return trait, "success", result_obj

        except Exception as exc:
            logger.warning("ANOVA: trait '%s' failed — %s", trait, exc)
            result_obj = AnovaTraitResult(
                trait=trait,
                status="failed",
                cv_interpretation_flag="cv_unavailable",
                interpretation=FAILED_TRAIT_CV_MESSAGE,
                data_warnings=[FAILED_TRAIT_CV_MESSAGE],
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

    analysis_context = AnalysisContext(
        is_single_environment=True,
        environment_count=1,
        design_type=design_type
    )

    return AnovaModuleResponse(
        dataset_token=request.dataset_token,
        mode=mode,
        trait_results=trait_results,
        failed_traits=failed_traits,
        analysis_context=analysis_context,
    )
