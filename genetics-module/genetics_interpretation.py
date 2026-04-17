"""
Genetics Interpretation Engine (No circular dependencies)

Generates VALIDATED genetic parameters interpretation sections.
Returns strict GeneticsInterpretationSections objects, never freeform prose.
"""

from typing import Optional, Tuple
from interpretation_sections import GeneticsInterpretationSections


def generate_genetics_interpretation_sections(
    trait_name: str,
    h2: Optional[float],
    gam: Optional[float],
    gcv: Optional[float],
    pcv: Optional[float],
    gxe_significant: bool = False,
    environment_significant: bool = False,
    n_observations: Optional[int] = None,
) -> GeneticsInterpretationSections:
    """
    Generate VALIDATED genetic parameters interpretation as strict section objects.
    
    All fields populated deterministically from Python logic:
    - h² classification (high/moderate/low)
    - GAM classification (high/moderate/low)
    - GCV/PCV relationship
    - Risk flags (GxE, small n, etc.)
    
    NO freeform generation. NO template placeholders. NO bypassed validators.
    
    Returns:
        GeneticsInterpretationSections with all required fields populated
    """
    
    h2_class = _classify_heritability(h2)
    gam_class = _classify_gam(gam)
    
    # ── Section 1: Overview ──────────────────────────────────────────────
    overview_parts = [f"Genetic parameters were estimated for {trait_name}."]
    if n_observations is not None and n_observations < 10:
        overview_parts.append("The sample size is limited, indicating preliminary estimates.")
    overview = " ".join(overview_parts)
    
    # ── Section 2: Heritability Interpretation ───────────────────────────
    if h2_class == "not_computed":
        heritability_interp = (
            f"Broad-sense heritability (h²) could not be reliably estimated for {trait_name}. "
            "Data limitations prevent interpretation of genetic control."
        )
    elif h2_class == "high":
        heritability_interp = (
            f"Broad-sense heritability is estimated at h² = {h2:.3f} (high), "
            f"indicating that the majority of observed phenotypic variation in {trait_name} "
            "is attributable to genetic differences among genotypes under these conditions."
        )
    elif h2_class == "moderate":
        heritability_interp = (
            f"Broad-sense heritability is estimated at h² = {h2:.3f} (moderate), "
            f"indicating that both genetic and environmental factors contribute substantially "
            f"to phenotypic variation in {trait_name}."
        )
    else:  # low
        heritability_interp = (
            f"Broad-sense heritability is estimated at h² = {h2:.3f} (low), "
            f"indicating that environmental factors and/or measurement variation dominate "
            f"the phenotypic variation in {trait_name}."
        )
    
    # ── Section 3: Genetic Advance Interpretation ────────────────────────
    if gam_class == "not_computed":
        genetic_advance_interp = (
            f"Genetic advance for {trait_name} could not be estimated due to data limitations."
        )
    elif gam_class == "high":
        genetic_advance_interp = (
            f"The genetic advance as a percent of the mean (GAM) is {gam:.2f}% (high), "
            f"suggesting that direct phenotypic selection should produce substantial response "
            f"in {trait_name} in the next generation."
        )
    elif gam_class == "moderate":
        genetic_advance_interp = (
            f"The genetic advance as a percent of the mean (GAM) is {gam:.2f}% (moderate), "
            f"indicating that phenotypic selection should produce meaningful but not rapid "
            f"response in {trait_name}."
        )
    else:  # low
        genetic_advance_interp = (
            f"The genetic advance as a percent of the mean (GAM) is {gam:.2f}% (low), "
            f"indicating that selection response for {trait_name} under these conditions "
            f"is expected to be limited."
        )
    
    # ── Section 4: Variance Relationship Interpretation ─────────────────
    variance_interp_parts = []
    if gcv is not None and pcv is not None:
        diff = float(pcv) - float(gcv)
        variance_interp_parts.append(
            f"The genotypic coefficient of variation (GCV = {gcv:.2f}%) and "
            f"phenotypic coefficient of variation (PCV = {pcv:.2f}%) indicate"
        )
        
        if diff <= 2:
            variance_interp_parts.append(
                "limited variance inflation between genetic and phenotypic components."
            )
            env_active = gxe_significant or environment_significant
            if env_active:
                variance_interp_parts.append(
                    "However, ANOVA results show significant environmental or G×E effects, "
                    "so environmental factors still influence trait expression and may alter rankings."
                )
            else:
                variance_interp_parts.append(
                    "Environmental effects on this trait appear modest under the tested conditions."
                )
        elif diff <= 7:
            variance_interp_parts.append(
                "moderate variance inflation, suggesting appreciable environmental influence."
            )
        else:
            variance_interp_parts.append(
                "substantial variance inflation, indicating strong environmental effects on trait expression."
            )
    else:
        variance_interp_parts.append(
            "Coefficients of variation (GCV, PCV) were not computed or data unavailable."
        )
    
    variance_interp = " ".join(variance_interp_parts)
    
    # ── Section 5: Breeding Interpretation ───────────────────────────────
    if h2_class == "not_computed":
        breeding_interp = (
            "Genetic basis could not be established. Expand or redesign the experiment "
            "before making selection decisions."
        )
    elif h2_class == "high":
        breeding_interp = (
            "Strong genetic basis justifies direct phenotypic selection. Prioritize "
            "identification and advancement of high-performing individuals."
        )
    elif h2_class == "moderate":
        breeding_interp = (
            "Moderate genetic basis allows direct selection, but should be combined with "
            "environmental standardization and multi-environment evaluation for stability assessment."
        )
    else:  # low
        breeding_interp = (
            "Weak genetic basis makes direct phenotypic selection unreliable. Focus on improving "
            "growing conditions, management practices, and measurement precision first."
        )
    
    # ── Section 6: Risk & Limitations ────────────────────────────────────
    risk_parts = []
    
    if n_observations is not None and n_observations < 10:
        risk_parts.append(
            f"The sample size ({n_observations} genotypes) is small, indicating these estimates "
            "are preliminary and should be treated with caution."
        )
    
    if gxe_significant:
        risk_parts.append(
            f"Significant genotype × environment interaction was detected for {trait_name}, "
            "indicating that genotype rankings and trait values may vary across environments. "
            "Multi-environment validation is essential before committing to selection decisions."
        )
    
    if environment_significant and not gxe_significant:
        risk_parts.append(
            f"Significant environmental effects were detected, indicating that trait expression "
            "is sensitive to growing conditions. Environmental standardization will increase precision."
        )
    
    risk_limitations = " ".join(risk_parts) if risk_parts else "No major limitations identified."
    
    # ── Section 7: Recommendation ────────────────────────────────────────
    if h2_class == "high":
        recommendation = (
            f"Proceed with direct phenotypic selection for {trait_name}. "
            "High heritability and meaningful expected response suggest selection will be effective."
        )
    elif h2_class == "moderate":
        recommendation = (
            f"Consider direct phenotypic selection for {trait_name} combined with environmental "
            f"optimization. Conduct multi-environment testing to verify genotype stability."
        )
    else:
        recommendation = (
            f"Do not rely on direct phenotypic selection for {trait_name} at present. "
            "Improve experimental conditions and measurement precision before selection."
        )
    
    return GeneticsInterpretationSections(
        overview=overview,
        heritability_interpretation=heritability_interp,
        genetic_advance_interpretation=genetic_advance_interp,
        variance_relationship_interpretation=variance_interp,
        breeding_interpretation=breeding_interp,
        risk_limitations=risk_limitations,
        recommendation=recommendation,
    )


def generate_genetics_interpretation(
    trait_name: str,
    h2: Optional[float],
    gam: Optional[float],
    gcv: Optional[float],
    pcv: Optional[float],
    gxe_significant: bool = False,
    environment_significant: bool = False,
    n_observations: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Generate academic-grade genetics interpretation following VivaSense standards.
    
    BACKWARD COMPATIBILITY FUNCTION: Calls generate_genetics_interpretation_sections
    and returns results as tuple (interpretation_text, breeding_implication_text).
    
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
