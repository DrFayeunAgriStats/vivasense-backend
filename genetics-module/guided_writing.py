"""
VivaSense Academic Mentor — Layer C: Guided Writing Support
===========================================================

Pure Python template engine.  No AI, no R, no I/O.
Takes a parsed analysis result dict and returns a GuidedWritingBlock.

Design principle (restored from V4.4):
  • All sentence starters have ___ blanks for EVERY numerical value.
  • The student must fill every blank from their own analysis output.
  • Pre-filling numbers defeats the academic purpose.
  • 'values_to_fill' names WHAT each blank needs — not the value itself.
  • 'hint' tells the student WHERE to find it.

Modules covered:
  anova               — significance, means, Tukey, assumption
  genetic_parameters  — heritability, GAM, GCV/PCV, breeding implication
  correlation         — pairwise r, co-selection, causation caution
  heatmap             — pattern summary, dominant pair, scope
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from academic_schemas import GuidedWritingBlock, SentenceStarter

logger = logging.getLogger(__name__)

# ── Fixed text constants ───────────────────────────────────────────────────────

_SCOPE_STATEMENT = (
    "These results apply to this experiment and should be interpreted "
    "within this context. Single-experiment results cannot support general "
    "management recommendations."
)

_SUPERVISOR_PROMPT = (
    "Discuss these findings with your supervisor before finalising your "
    "write-up. — Dr. Fayeun, VivaSense Academic Mentor"
)

_RESEARCH_REFERRAL = (
    "For structured support with writing your results section, visit "
    "Field-to-Insight Academy: www.fieldtoinsightacademy.com.ng"
)


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def build_guided_writing(
    module_type: str,
    trait: Optional[str],
    analysis_result: Dict[str, Any],
) -> GuidedWritingBlock:
    """
    Build a GuidedWritingBlock for the given module and trait result.

    Parameters
    ----------
    module_type     : "anova" | "genetic_parameters" | "correlation" | "heatmap"
    trait           : trait name (for anova / genetic_parameters)
    analysis_result : the single-trait result dict (already extracted from
                      the module response by academic_interpretation.py)
    """
    dispatch = {
        "anova":               _build_anova_writing,
        "genetic_parameters":  _build_gp_writing,
        "correlation":         _build_correlation_writing,
        "heatmap":             _build_heatmap_writing,
    }
    builder = dispatch.get(module_type)
    if builder is None:
        logger.warning("guided_writing: unknown module_type '%s'", module_type)
        return _empty_block(module_type, trait)

    return builder(trait or "the trait", analysis_result)


# ============================================================================
# ANOVA MODULE
# ============================================================================

def _build_anova_writing(
    trait: str,
    result: Dict[str, Any],
) -> GuidedWritingBlock:
    """Sentence starters and examiner checkpoint for the ANOVA module."""

    # Dispatch to design-specific builder when appropriate
    if result.get("design_type") == "split_plot_rcbd":
        return _build_split_plot_anova_writing(trait, result)

    starters: List[SentenceStarter] = []

    n_genos = result.get("n_genotypes")
    n_envs = result.get("n_environments")
    exp_str = "in this experiment"
    if n_genos is not None:
        if n_envs is not None and n_envs > 1:
            exp_str = f"evaluated across {n_genos} genotypes and {n_envs} environments"
        else:
            exp_str = f"evaluated across {n_genos} genotypes"

    # ── Starter 1: Significance ──────────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Significance statement",
        template=(
            "An analysis of variance for {trait} {exp_str} showed that the genotype effect was ___ "
            "(F = ___, p = ___, η² = ___)."
        ).format(trait=trait, exp_str=exp_str),
        values_to_fill=[
            "write 'significant' (p < 0.05) or 'not significant' (p ≥ 0.05)",
            "F-value for Genotype (from ANOVA table)",
            "p-value for Genotype (from ANOVA table)",
            "η² = SS_Genotype ÷ SS_Total (calculate from ANOVA table)",
        ],
        hint="ANOVA table → Genotype row",
    ))

    # ── Starter 2: Means comparison ──────────────────────────────────────────
    ms = result.get("mean_separation") or {}
    n_genos = len(ms.get("genotype", [])) if ms else 0
    geno_phrase = (
        f"across the {n_genos} genotypes tested"
        if n_genos > 0
        else "among the genotypes tested"
    )

    starters.append(SentenceStarter(
        purpose="Means comparison",
        template=(
            "The highest mean {trait} {geno_phrase} was recorded in ___ "
            "(mean = ___), while the lowest was recorded in ___ (mean = ___)."
        ).format(trait=trait, geno_phrase=geno_phrase),
        values_to_fill=[
            "name of top-ranked genotype (rank 1 in Mean Separation table)",
            "mean value of top-ranked genotype",
            "name of bottom-ranked genotype (last rank)",
            "mean value of bottom-ranked genotype",
        ],
        hint="Mean Separation table → sorted by mean descending",
    ))

    # ── Starter 3: Tukey group reference ────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Statistical grouping",
        template=(
            "Means were separated using ___ at α = ___. "
            "Genotypes were assigned to ___ distinct group(s); "
            "___ and ___ belonged to group ___ (highest), "
            "while ___ was in group ___ (lowest)."
        ),
        values_to_fill=[
            "name of post-hoc test (Tukey HSD or LSD — from Mean Separation table header)",
            "alpha level (0.05 unless stated otherwise)",
            "number of distinct Tukey group letters",
            "name of a top-group genotype",
            "name of another top-group genotype (or omit if only one)",
            "top group letter (usually 'a')",
            "name of bottom-ranked genotype",
            "bottom group letter (e.g. 'b', 'c', or 'd')",
        ],
        hint="Mean Separation table → Group column",
    ))

    # ── Starter 4: Assumption check ──────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Assumption check",
        template=(
            "The Shapiro-Wilk test ___ a departure from normality (W = ___, p = ___). "
            "The Levene test ___ a violation of homogeneity of variance "
            "(statistic = ___, p = ___)."
        ),
        values_to_fill=[
            "write 'indicated' if p < 0.05, or 'did not indicate' if p ≥ 0.05",
            "W statistic from Assumption Tests table",
            "p-value from Shapiro-Wilk test",
            "write 'indicated' if p < 0.05, or 'did not indicate' if p ≥ 0.05",
            "test statistic from Levene test",
            "p-value from Levene test",
        ],
        hint="Assumption Tests table",
    ))

    # ── Examiner checkpoint ───────────────────────────────────────────────────
    checkpoint = [
        "F-value and p-value reported for the genotype effect",
        "Tukey group letters cited for all genotypes discussed",
        "η² effect size reported alongside p-value",
        "Assumption test results (Shapiro-Wilk, Levene) referenced",
        "At least one scope phrase present: 'in this experiment' or 'among the levels tested'",
    ]

    # ── Caution notes (data-driven) ───────────────────────────────────────────
    caution = _build_anova_caution(result)

    return GuidedWritingBlock(
        module_type="anova",
        trait=trait,
        sentence_starters=starters,
        examiner_checkpoint=checkpoint,
        scope_statement=_SCOPE_STATEMENT,
        caution_note=caution,
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )


def _build_anova_caution(result: Dict[str, Any]) -> Optional[str]:
    """Generate contextual caution notes from analysis data."""
    notes: List[str] = []

    n_reps = result.get("n_reps")
    n_genos = result.get("n_genotypes")

    if n_reps is not None and n_reps < 3 and n_genos is not None:
        n_pairs = (n_genos * (n_genos - 1)) // 2
        notes.append(
            f"⚠ Low-replication note: This experiment has {n_genos} genotypes "
            f"with only {n_reps} replicate(s) each. The post-hoc test evaluates "
            f"{n_pairs} pairwise comparisons under these conditions and has limited "
            "power to detect individual differences even when the overall F-test is "
            "significant. Interpret mean separations cautiously."
        )

    # Check assumption tests for normality violation
    at = result.get("assumption_tests") or {}
    shapiro = at.get("shapiro_wilk") or at.get("Shapiro-Wilk") or {}
    if isinstance(shapiro, dict):
        p_sw = shapiro.get("p_value") or shapiro.get("p.value") or shapiro.get("p")
        if p_sw is not None and p_sw < 0.05:
            if n_reps is not None and n_genos is not None and n_reps * n_genos <= 15:
                notes.append(
                    "⚠ Normality note: The Shapiro-Wilk test indicated a departure from "
                    "normality. With a small balanced design, ANOVA can be moderately "
                    "robust to this violation. Confirm with your supervisor whether a "
                    "non-parametric alternative (Kruskal-Wallis) should also be reported."
                )
            else:
                notes.append(
                    "⚠ Normality note: The Shapiro-Wilk test indicated a departure from "
                    "normality. Consider reporting the W statistic and p-value in your "
                    "write-up and consulting your supervisor about a non-parametric "
                    "alternative (Kruskal-Wallis test)."
                )

    data_warnings = result.get("data_warnings") or []
    for w in data_warnings:
        notes.append(f"⚠ Design note: {w}")

    return "\n\n".join(notes) if notes else None


# ============================================================================
# SPLIT-PLOT RCBD — domain-neutral ANOVA writing support
# ============================================================================

def _build_split_plot_anova_writing(
    trait: str,
    result: Dict[str, Any],
) -> GuidedWritingBlock:
    """
    Sentence starters and examiner checkpoint for split-plot RCBD ANOVA.
    Uses domain-neutral language (main-plot factor / subplot factor).
    No genotype, breeding, Tukey, or mean-separation language.
    """
    starters: List[SentenceStarter] = []

    # ── Starter 1: Design and testing structure ──────────────────────────────
    starters.append(SentenceStarter(
        purpose="Design and error-strata description",
        template=(
            "A split-plot RCBD was used with ___ replicate blocks. "
            "The main-plot factor (___ levels) was randomised within blocks "
            "and tested against the whole-plot error; the subplot factor "
            "(___ levels) and the main-plot × subplot interaction were "
            "tested against the subplot residual."
        ),
        values_to_fill=[
            "number of replicates (n_reps from Descriptive Statistics)",
            "name and number of levels for the main-plot factor",
            "name and number of levels for the subplot factor",
        ],
        hint="Design section of the ANOVA output — two error strata shown",
    ))

    # ── Starter 2: Main-plot factor effect ───────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Main-plot factor significance statement",
        template=(
            "The main-plot factor had a ___ effect on {trait} "
            "(F₁ = ___, df = ___, p = ___), indicating that ___ differed "
            "___ among the main-plot treatment levels tested."
        ).format(trait=trait),
        values_to_fill=[
            "write 'significant' (p < 0.05) or 'non-significant' (p ≥ 0.05)",
            "F-value for main-plot factor (whole-plot error stratum of ANOVA table)",
            "numerator degrees of freedom for main-plot factor",
            "p-value for main-plot factor",
            "trait name",
            "write 'significantly' or 'non-significantly' — matching significance above",
        ],
        hint="ANOVA table → main-plot row (whole-plot error stratum)",
    ))

    # ── Starter 3: Subplot factor effect ─────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Subplot factor significance statement",
        template=(
            "The subplot factor had a ___ effect on {trait} "
            "(F₂ = ___, df = ___, p = ___), based on the subplot residual "
            "as the error term."
        ).format(trait=trait),
        values_to_fill=[
            "write 'significant' (p < 0.05) or 'non-significant' (p ≥ 0.05)",
            "F-value for subplot factor (subplot error stratum)",
            "numerator degrees of freedom for subplot factor",
            "p-value for subplot factor",
        ],
        hint="ANOVA table → subplot row (subplot residual stratum)",
    ))

    # ── Starter 4: Interaction ────────────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Main-plot × subplot interaction statement",
        template=(
            "The main-plot × subplot interaction ___ significant for {trait} "
            "(F = ___, p = ___). "
            "This ___ that the effect of the subplot factor depended on "
            "the level of the main-plot factor."
        ).format(trait=trait),
        values_to_fill=[
            "write 'was' or 'was not'",
            "F-value for interaction (subplot residual stratum)",
            "p-value for interaction",
            "write 'indicates' (significant) or 'does not indicate' (non-significant)",
        ],
        hint="ANOVA table → interaction row (subplot residual stratum)",
    ))

    # ── Starter 5: Experimental precision ────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Experimental precision (CV%)",
        template=(
            "The coefficient of variation was ___%, reflecting ___ experimental "
            "precision for {trait} under the conditions of this study."
        ).format(trait=trait),
        values_to_fill=[
            "CV% from Descriptive Statistics",
            "write 'good' (CV ≤ 10%), 'moderate' (10–20%), or 'low' (> 20%)",
        ],
        hint="Descriptive Statistics → CV% row",
    ))

    checkpoint = [
        "F-values reported for main-plot factor, subplot factor, and their interaction",
        "Correct error stratum stated for each F-test (whole-plot error vs. subplot residual)",
        "Design identified as split-plot RCBD with restricted randomisation",
        "Number of replicates stated",
        "CV% reported and classified",
        "No Tukey group letters or mean-separation table cited — not applicable here",
        "At least one scope phrase present: 'in this experiment' or 'among the levels tested'",
    ]

    caution = _build_split_plot_caution(result)

    return GuidedWritingBlock(
        module_type="anova",
        trait=trait,
        sentence_starters=starters,
        examiner_checkpoint=checkpoint,
        scope_statement=_SCOPE_STATEMENT,
        caution_note=caution,
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )


def _build_split_plot_caution(result: Dict[str, Any]) -> Optional[str]:
    """Contextual caution notes for split-plot RCBD."""
    notes: List[str] = []

    n_reps = result.get("n_reps")
    if n_reps is not None and n_reps < 3:
        notes.append(
            f"⚠ Low-replication note: This experiment uses only {n_reps} replicate "
            "block(s). With few blocks, the whole-plot error has limited degrees of "
            "freedom, reducing power to detect main-plot factor differences. "
            "Interpret the whole-plot F-test cautiously."
        )

    data_warnings = result.get("data_warnings") or []
    for w in data_warnings:
        notes.append(f"⚠ Design note: {w}")

    return "\n\n".join(notes) if notes else None


# ============================================================================
# GENETIC PARAMETERS MODULE
# ============================================================================

def _build_gp_writing(
    trait: str,
    result: Dict[str, Any],
) -> GuidedWritingBlock:
    """Sentence starters and examiner checkpoint for Genetic Parameters module."""

    starters: List[SentenceStarter] = []

    n_genos = result.get("n_genotypes")
    n_envs = result.get("n_environments")
    exp_str = "in this experiment"
    if n_genos is not None:
        if n_envs is not None and n_envs > 1:
            exp_str = f"evaluated across {n_genos} genotypes and {n_envs} environments"
        else:
            exp_str = f"evaluated across {n_genos} genotypes"

    # ── Starter 1: Heritability ──────────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Heritability statement",
        template=(
            "For {trait} {exp_str}, the estimated broad-sense heritability was ___ "
            "(h² = ___), indicating ___ genetic control."
        ).format(trait=trait, exp_str=exp_str),
        values_to_fill=[
            "heritability classification: 'high' (h² ≥ 0.60), 'moderate' (0.30–0.59), or 'low' (< 0.30)",
            "h² value from Genetic Parameters table",
            "same classification word as above",
        ],
        hint="Genetic Parameters → Heritability (H²) row",
    ))

    # ── Starter 2: Joint H² + GAM ────────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Joint heritability and genetic advance",
        template=(
            "The genetic advance as percent of mean (GAM = ___%) was ___, "
            "and together with h² = ___, this combination suggests that "
            "direct phenotypic selection ___ expected to produce "
            "___ progress for {trait} in this experiment."
        ).format(trait=trait),
        values_to_fill=[
            "GAM% value from Genetic Parameters table",
            "GAM classification: 'high' (≥ 10%), 'moderate' (5–9.99%), or 'low' (< 5%)",
            "h² value",
            "write 'is' if both h² and GAM are moderate or high, 'is not' if both are low",
            "write 'substantial' / 'moderate' / 'limited' matching the GAM class",
        ],
        hint="Genetic Advance Estimates table → GAM%",
    ))

    # ── Starter 3: GCV vs PCV ────────────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="GCV vs PCV comparison",
        template=(
            "The genotypic coefficient of variation (GCV = ___%) was "
            "___ the phenotypic coefficient of variation (PCV = ___%), "
            "indicating that environmental effects ___ influenced "
            "trait expression for {trait} in this experiment."
        ).format(trait=trait),
        values_to_fill=[
            "GCV value",
            "write 'close to' (difference ≤ 2%), 'moderately lower than' (2–7%), or 'substantially lower than' (> 7%)",
            "PCV value",
            "write 'minimally', 'appreciably', or 'substantially' matching the GCV–PCV gap",
        ],
        hint="Genetic Advance Estimates table → GCV and PCV rows",
    ))

    # ── Starter 4: Breeding implication ──────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Breeding implication",
        template=(
            "Based on the h² and GAM values observed in this experiment, "
            "direct phenotypic selection for {trait} ___ likely to be "
            "___ under the environmental conditions of this study."
        ).format(trait=trait),
        values_to_fill=[
            "write 'is' or 'is not' based on combined h² and GAM levels",
            "write 'effective' / 'moderately effective' / 'ineffective' matching the parameter levels",
        ],
        hint="Interpretation section of the Genetic Parameters output",
    ))

    checkpoint = [
        "h² value and classification (high/moderate/low) both stated",
        "GAM% and its class stated jointly with h² — not reported alone",
        "GCV and PCV both cited and compared — not listed separately without comparison",
        "Breeding implication scoped to 'this environment' or 'this experiment'",
        "Any negative variance component warning cited if present in the output",
    ]

    caution = _build_gp_caution(result)

    return GuidedWritingBlock(
        module_type="genetic_parameters",
        trait=trait,
        sentence_starters=starters,
        examiner_checkpoint=checkpoint,
        scope_statement=_SCOPE_STATEMENT,
        caution_note=caution,
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )


def _build_gp_caution(result: Dict[str, Any]) -> Optional[str]:
    notes: List[str] = []

    vc = result.get("variance_components") or {}
    sigma2_g = vc.get("sigma2_genotype")
    if sigma2_g is not None and sigma2_g < 0:
        notes.append(
            "⚠ Negative genotypic variance detected (σ²G < 0). "
            "This can occur with small datasets or low genotypic variation. "
            "Treat heritability and genetic advance estimates with caution. "
            "Report this in your write-up and consult your supervisor."
        )

    h2_dict = result.get("heritability") or {}
    h2 = h2_dict.get("h2_broad_sense") if isinstance(h2_dict, dict) else None
    if h2 is not None and h2 > 1.0:
        notes.append(
            "⚠ Heritability estimate > 1.0 detected. This is statistically "
            "impossible and indicates a model estimation issue. Do not report "
            "this value without supervisor consultation."
        )

    data_warnings = result.get("data_warnings") or []
    for w in data_warnings:
        notes.append(f"⚠ Design note: {w}")

    return "\n\n".join(notes) if notes else None


# ============================================================================
# CORRELATION MODULE
# ============================================================================

def _build_correlation_writing(
    trait: str,
    result: Dict[str, Any],
) -> GuidedWritingBlock:
    """Sentence starters and examiner checkpoint for the Correlation module."""

    trait_names = result.get("trait_names") or []
    n_traits = len(trait_names)
    method = result.get("method", "Pearson")
    
    n_obs = result.get("n_observations")
    obs_str = f"across {n_obs} genotype means" if n_obs else "among the genotypes tested"

    starters: List[SentenceStarter] = []

    # ── Starter 1: Pairwise correlation ──────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Pairwise correlation statement",
        template=(
            "A ___ phenotypic correlation was observed between ___ and ___ "
            "{obs_str} (r = ___, p = ___)."
        ).format(obs_str=obs_str),
        values_to_fill=[
            "direction + strength: 'strong positive' (r ≥ 0.70), 'moderate positive' (0.40–0.69), "
            "'weak' (< 0.40), 'moderate negative', or 'strong negative'",
            "name of first trait (from Correlation matrix)",
            "name of second trait",
            "r-value from r_matrix (e.g. r_matrix[i][j])",
            "p-value from p_matrix (e.g. p_matrix[i][j])",
        ],
        hint=(
            f"Correlation matrix (r_matrix) — {method} r-values. "
            "Use upper triangle only (i < j) to avoid duplication."
        ),
    ))

    # ── Starter 2: Co-selection implication ──────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Co-selection implication (positive correlation)",
        template=(
            "The positive phenotypic correlation between ___ and ___ "
            "(r = ___, p = ___) suggests that selection for ___ "
            "may produce concurrent gains in ___ among the genotypes tested "
            "in this experiment."
        ),
        values_to_fill=[
            "first trait",
            "second trait",
            "r-value",
            "p-value",
            "the primary selection target (trait with the higher breeding priority)",
            "the secondary trait that would benefit",
        ],
        hint="Use only for significant positive correlations (p < 0.05, r ≥ 0.40)",
    ))

    # ── Starter 3: Causation caution ─────────────────────────────────────────
    starters.append(SentenceStarter(
        purpose="Causation caution (mandatory)",
        template=(
            "It should be noted that correlation does not imply causation; "
            "the observed association between ___ and ___ "
            "reflects co-variation among genotype means in this experiment "
            "and does not establish a directional biological relationship."
        ),
        values_to_fill=[
            "first trait in the pair discussed",
            "second trait",
        ],
        hint="Include this sentence whenever a significant correlation is reported",
    ))

    checkpoint = [
        "r-value and p-value both reported for every pair discussed",
        "Causation language ('causes', 'leads to', 'drives') absent from the write-up",
        "Scope limited to 'among the genotypes tested' — no generalisation to populations",
        "Strong pairs (|r| ≥ 0.70, p < 0.05) specifically identified",
        "Statement that 'correlation does not imply causation' included",
    ]

    return GuidedWritingBlock(
        module_type="correlation",
        trait=None,
        sentence_starters=starters,
        examiner_checkpoint=checkpoint,
        scope_statement=_SCOPE_STATEMENT,
        caution_note=(
            "⚠ Causation caution: Phenotypic correlations describe co-variation "
            "among genotype means. They do not establish physiological or causal "
            "relationships between traits. Never use causal language in your write-up."
        ),
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )


# ============================================================================
# HEATMAP MODULE
# ============================================================================

def _build_heatmap_writing(
    trait: str,
    result: Dict[str, Any],
) -> GuidedWritingBlock:
    """Sentence starters for the Heatmap module (lightweight version of Correlation)."""

    labels = result.get("labels") or []
    method = result.get("method", "Pearson")
    
    n_obs = result.get("n_observations")
    obs_str = f"across {n_obs} genotype means" if n_obs else "among the genotypes tested"

    starters: List[SentenceStarter] = []

    starters.append(SentenceStarter(
        purpose="Heatmap pattern summary",
        template=(
            "The {method} correlation heatmap for ___ traits revealed that "
            "___ and ___ showed the strongest positive association "
            "(r ≈ ___), while ___ and ___ showed the strongest "
            "negative association (r ≈ ___) {obs_str}."
        ).format(method=method.capitalize(), obs_str=obs_str),
        values_to_fill=[
            f"number of traits ({len(labels)} in this analysis)",
            "trait with highest off-diagonal r value — from heatmap or matrix",
            "its correlated trait",
            "peak positive r value",
            "trait pair with most negative r",
            "most negative r value",
        ],
        hint="Heatmap matrix — look for darkest green (positive) and darkest red (negative) cells",
    ))

    starters.append(SentenceStarter(
        purpose="Scope statement for heatmap",
        template=(
            "The patterns shown in the heatmap reflect phenotypic co-variation "
            "among genotype means in this experiment and should not be interpreted "
            "as causal relationships among the traits."
        ),
        values_to_fill=[],
        hint="Include this in any write-up that references the heatmap",
    ))

    checkpoint = [
        "Heatmap described in terms of r-values, not visual impressions only",
        "Strongest positive and negative pairs specifically identified",
        "Causation language absent from the heatmap description",
        "Scope limited to genotypes in this experiment",
        "Legend/colour scale explained in the figure caption",
    ]

    return GuidedWritingBlock(
        module_type="heatmap",
        trait=None,
        sentence_starters=starters,
        examiner_checkpoint=checkpoint,
        scope_statement=_SCOPE_STATEMENT,
        caution_note=(
            "⚠ Heatmap caution: Colour patterns are a visual aid. "
            "Always cite the actual r-values and p-values from the "
            "correlation table in your written results — not just the colours."
        ),
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )


# ============================================================================
# EMPTY FALLBACK
# ============================================================================

def _empty_block(module_type: str, trait: Optional[str]) -> GuidedWritingBlock:
    return GuidedWritingBlock(
        module_type=module_type,
        trait=trait,
        sentence_starters=[],
        examiner_checkpoint=[],
        scope_statement=_SCOPE_STATEMENT,
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )
