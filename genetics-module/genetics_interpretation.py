"""
Genetics Interpretation Engine (No circular dependencies)

Generates academic-grade genetic parameters interpretation.
Decoupled from routes modules to avoid circular imports.
"""

from typing import Optional, Tuple


def generate_genetics_interpretation(
    trait_name: str,
    h2: Optional[float],
    gam: Optional[float],
    gcv: Optional[float],
    pcv: Optional[float],
    gxe_significant: bool = False,
    environment_significant: bool = False,
) -> Tuple[str, str]:
    """
    Generate academic-grade genetics interpretation following VivaSense standards.
    
    Returns:
        (interpretation_text, breeding_implication_text)
    """
    h2_class = _classify_heritability(h2)
    gam_class = _classify_gam(gam)

    # ── Joint h² + GAM interpretation ───────────────────────────────────
    if h2_class == "not_computed":
        interpretation = (
            f"Heritability could not be estimated for '{trait_name}'. "
            "Data limitations prevent reliable genetic interpretation."
        )
    elif h2_class == "high" and gam_class == "high":
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) indicates HIGH genetic control "
            f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is HIGH, "
            "suggesting substantial expected response to direct selection. "
            "Additive gene effects are likely important; direct phenotypic selection should be effective."
        )
    elif h2_class == "high" and gam_class == "moderate":
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) indicates HIGH genetic control "
            f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is MODERATE, "
            "indicating a meaningful selection response. Direct phenotypic selection should yield steady "
            "genetic progress; both additive and non-additive gene effects likely contribute."
        )
    elif h2_class == "high" and gam_class == "low":
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) indicates HIGH genetic control, "
            f"yet the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW for '{trait_name}'. "
            "This dissociation suggests that while phenotypic variation is substantially genetic, the "
            "expected response to selection is limited. Non-additive gene effects or strong inbreeding "
            "depression may be responsible."
        )
    elif h2_class == "moderate" and gam_class == "high":
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) indicates MODERATE genetic control "
            f"of '{trait_name}', with the genetic advance as percent of mean (GAM = {gam:.2f}%) being HIGH. "
            "Useful selection response is achievable despite environmental influence. "
            "Both genetic and environmental management should be considered."
        )
    elif h2_class == "moderate" and gam_class == "moderate":
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) and genetic advance as percent "
            f"of mean (GAM = {gam:.2f}%) both indicate MODERATE genetic control for '{trait_name}'. "
            "Selection may be useful, though environmental factors remain important. "
            "Progress should be steady but not rapid."
        )
    elif h2_class == "moderate" and gam_class == "low":
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) suggests MODERATE genetic control "
            f"of '{trait_name}', but the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW. "
            "Direct phenotypic selection may be slow. Consider investigating additive effects more carefully "
            "or combining selection with environmental optimization."
        )
    else:  # low h2
        interpretation = (
            f"The estimated broad-sense heritability (h² = {h2:.3f}) indicates LOW genetic control of "
            f"'{trait_name}' under the present environment. Phenotypic variation is dominated by environmental "
            "factors and/or measurement variation. Direct phenotypic selection is unlikely to be reliable; "
            "focus on improving growing conditions and management practices."
        )

    # ── GCV vs PCV addendum ──────────────────────────────────────────────
    if gcv is not None and pcv is not None:
        try:
            diff = float(pcv) - float(gcv)
            env_active = gxe_significant or environment_significant
            if diff <= 2:
                if env_active:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is similar to the PCV ({pcv:.2f}%), "
                        "indicating limited variance inflation between the genetic and phenotypic coefficients of variation. "
                        "However, the ANOVA results indicate significant environmental effects or genotype × environment "
                        "interaction for this trait, suggesting that environmental conditions may still influence trait "
                        "expression and alter genotype rankings across environments. "
                        "The GCV–PCV comparison alone should not be taken as evidence of negligible environmental effects."
                    )
                else:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is similar to the PCV ({pcv:.2f}%), "
                        "indicating limited variance inflation between the genetic and phenotypic coefficients of variation "
                        "in this experiment. Environmental effects on this trait appear modest under the conditions tested."
                    )
            elif diff <= 7:
                if env_active:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is moderately lower than the PCV ({pcv:.2f}%), "
                        "suggesting appreciable environmental influence on trait expression. "
                        "This is consistent with the ANOVA evidence of significant environmental or "
                        "genotype × environment effects."
                    )
                else:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is moderately lower than the PCV ({pcv:.2f}%), "
                        "suggesting appreciable but not dominant environmental influence."
                    )
            else:
                interpretation += (
                    f" The GCV ({gcv:.2f}%) is substantially lower than the PCV ({pcv:.2f}%), "
                    "indicating that environmental factors strongly affect trait expression."
                )
        except (TypeError, ValueError):
            pass

    # ── Breeding recommendation ──────────────────────────────────────────
    if h2_class == "not_computed":
        breeding_implication = (
            "Heritability could not be reliably estimated. Redesign or expand the experiment "
            "to improve precision before making selection decisions."
        )
    elif h2_class == "high":
        breeding_implication = (
            "Strong genetic basis for the trait. Direct phenotypic selection should be effective "
            "in this environment. Prioritize identification and advancement of high-value individuals "
            "for next-generation breeding."
        )
    elif h2_class == "moderate":
        breeding_implication = (
            "Moderate genetic basis. Direct selection is possible but should be combined with "
            "attention to environmental standardization. Consider multi-environment evaluation "
            "to assess stability of selected genotypes."
        )
    else:  # low
        breeding_implication = (
            "Weak genetic basis under present conditions. Direct selection will be unreliable. "
            "Prioritize improvement of growing conditions, management practices, and measurement "
            "precision before intensifying selection."
        )

    return interpretation, breeding_implication


def _classify_heritability(h2) -> str:
    """Classify heritability into low, moderate, high, or not_computed."""
    if h2 is None:
        return "not_computed"
    try:
        h2 = float(h2)
    except (TypeError, ValueError):
        return "not_computed"
    if h2 < 0.30:
        return "low"
    if h2 < 0.60:
        return "moderate"
    return "high"


def _classify_gam(gam_percent) -> str:
    """Classify genetic advance as % of mean into low, moderate, high, or not_computed."""
    if gam_percent is None:
        return "not_computed"
    try:
        gam_percent = float(gam_percent)
    except (TypeError, ValueError):
        return "not_computed"
    if gam_percent < 5:
        return "low"
    if gam_percent < 10:
        return "moderate"
    return "high"
