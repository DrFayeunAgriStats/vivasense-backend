"""
Genetics Interpretation Engine (No circular dependencies)

Generates VALIDATED genetic parameters interpretation sections.
Returns strict GeneticsInterpretationSections objects, never freeform prose.
"""

from typing import Optional, Tuple
from interpretation_sections import GeneticsInterpretationSections
from interpretation import InterpretationEngine


def _describe_gcv_pcv(gcv: float, pcv: float, trait_name: str) -> str:
    if gcv <= 0:
        return ""
    inflation_pct = ((pcv - gcv) / gcv) * 100

    if inflation_pct < 3:
        return (
            f"GCV ({gcv:.2f}%) and PCV ({pcv:.2f}%) are nearly identical "
            f"(difference: {inflation_pct:.1f}%), indicating negligible environmental "
            f"variance inflation. The genetic signal for {trait_name} is exceptionally "
            f"clean - direct phenotypic selection will closely track underlying genotypic value."
        )
    elif inflation_pct < 10:
        return (
            f"GCV ({gcv:.2f}%) is slightly lower than PCV ({pcv:.2f}%) "
            f"(inflation: {inflation_pct:.1f}%), suggesting a small but non-trivial "
            f"environmental contribution to phenotypic variance in {trait_name}. "
            f"Selection efficiency remains high given the elevated H2."
        )
    elif inflation_pct < 20:
        return (
            f"PCV ({pcv:.2f}%) is moderately higher than GCV ({gcv:.2f}%) "
            f"(inflation: {inflation_pct:.1f}%), indicating meaningful environmental "
            f"effects on {trait_name} expression. Selection in controlled or "
            f"multi-location environments is advisable."
        )
    else:
        return (
            f"PCV ({pcv:.2f}%) substantially exceeds GCV ({gcv:.2f}%) "
            f"(inflation: {inflation_pct:.1f}%), indicating strong environmental masking "
            f"of genetic differences for {trait_name}. Replicated multi-environment "
            f"evaluation is essential before selection decisions are made."
        )


def _describe_env_effects(f_env: float, p_env: float, f_gxe: float, p_gxe: float) -> str:
    env_sig = p_env is not None and p_env < 0.05
    gxe_sig = p_gxe is not None and p_gxe < 0.05

    if not env_sig:
        env_desc = "Environmental effects were non-significant for this trait"
    elif f_env > 1000:
        env_desc = (
            f"Strong environmental effects were detected (F = {f_env:,.3f}, p < 0.001), "
            f"indicating substantial variation in trait expression across environments"
        )
    elif f_env > 100:
        env_desc = (
            f"Moderate-to-strong environmental effects were detected "
            f"(F = {f_env:,.3f}, p < 0.001)"
        )
    elif f_env > 10:
        env_desc = (
            f"Moderate environmental effects were detected (F = {f_env:,.3f}, p < 0.001)"
        )
    else:
        env_desc = (
            f"Modest environmental effects were detected (F = {f_env:,.3f})"
        )

    if gxe_sig and f_gxe > 10:
        gxe_desc = (
            " Significant genotype-by-environment interaction was also detected "
            f"(GxE F = {f_gxe:,.3f}, p < 0.001), suggesting genotype rankings may not be "
            "consistent across all environments - stability analysis is recommended."
        )
    elif gxe_sig:
        gxe_desc = (
            f" A significant but modest GxE interaction was detected "
            f"(F = {f_gxe:,.3f}, p < 0.05)."
        )
    else:
        gxe_desc = (
            " GxE interaction was non-significant, indicating relatively consistent "
            "genotype performance across environments."
        )

    return env_desc + ". " + gxe_desc


def generate_genetics_interpretation_sections(
    trait_name: str,
    h2: Optional[float],
    gam: Optional[float],
    gcv: Optional[float],
    pcv: Optional[float],
    gxe_significant: bool = False,
    environment_significant: bool = False,
    n_observations: Optional[int] = None,
    anova_f_env: Optional[float] = None,
    anova_p_env: Optional[float] = None,
    anova_f_gxe: Optional[float] = None,
    anova_p_gxe: Optional[float] = None,
) -> GeneticsInterpretationSections:
    """
    Generate VALIDATED genetic parameters interpretation as strict section objects.
    
    All fields populated deterministically from Python logic:
    - H2 classification (high/moderate/low)
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
            f"Broad-sense heritability (H2) could not be reliably estimated for {trait_name}. "
            "Data limitations prevent interpretation of genetic control."
        )
    elif h2_class == "high":
        heritability_interp = (
            f"Broad-sense heritability is estimated at H2 = {h2:.3f} (high), "
            f"indicating that the majority of observed phenotypic variation in {trait_name} "
            "is attributable to genetic differences among genotypes under these conditions."
        )
    elif h2_class == "moderate":
        heritability_interp = (
            f"Broad-sense heritability is estimated at H2 = {h2:.3f} (moderate), "
            f"indicating that both genetic and environmental factors contribute substantially "
            f"to phenotypic variation in {trait_name}."
        )
    else:  # low
        heritability_interp = (
            f"Broad-sense heritability is estimated at H2 = {h2:.3f} (low), "
            f"indicating that environmental factors and/or measurement variation dominate "
            f"the phenotypic variation in {trait_name}."
        )
    
    # ── Section 3: Genetic Advance Interpretation ────────────────────────
    if gam_class == "not_computed":
        genetic_advance_interp = (
            f"Genetic advance for {trait_name} could not be estimated due to data limitations."
        )
    elif gam_class == "High":
        genetic_advance_interp = (
            f"The genetic advance as a percent of the mean (GAM) is {gam:.2f}% (high), "
            f"suggesting that direct phenotypic selection should produce substantial response "
            f"in {trait_name} in the next generation."
        )
    elif gam_class == "Medium":
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
            anova_f_env = float(anova_f_env) if anova_f_env is not None else 0.0
            anova_p_env = float(anova_p_env) if anova_p_env is not None else None
            anova_f_gxe = float(anova_f_gxe) if anova_f_gxe is not None else 0.0
            anova_p_gxe = float(anova_p_gxe) if anova_p_gxe is not None else None
            variance_interp_parts.append(
                _describe_env_effects(
                    f_env=anova_f_env,
                    p_env=anova_p_env,
                    f_gxe=anova_f_gxe,
                    p_gxe=anova_p_gxe,
                )
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
            f"Significant genotype-by-environment interaction was detected for {trait_name}, "
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
    anova_f_env: Optional[float] = None,
    anova_p_env: Optional[float] = None,
    anova_f_gxe: Optional[float] = None,
    anova_p_gxe: Optional[float] = None,
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

    # ── Joint H2 + GAM interpretation ───────────────────────────────────
    if h2_class == "not_computed":
        interpretation = (
            f"Heritability could not be estimated for '{trait_name}'. "
            "Data limitations prevent reliable genetic interpretation."
        )
    elif h2_class == "high" and gam_class == "High":
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) indicates HIGH genetic control "
            f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is HIGH, "
            "suggesting substantial expected response to direct selection. "
            "Additive gene effects are likely important; direct phenotypic selection should be effective."
        )
    elif h2_class == "high" and gam_class == "Medium":
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) indicates HIGH genetic control "
            f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is MODERATE, "
            "indicating a meaningful selection response. Direct phenotypic selection should yield steady "
            "genetic progress; both additive and non-additive gene effects likely contribute."
        )
    elif h2_class == "high" and gam_class == "Low":
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) indicates HIGH genetic control, "
            f"yet the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW for '{trait_name}'. "
            "This dissociation suggests that while phenotypic variation is substantially genetic, the "
            "expected response to selection is limited. Non-additive gene effects or strong inbreeding "
            "depression may be responsible."
        )
    elif h2_class == "moderate" and gam_class == "High":
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) indicates MODERATE genetic control "
            f"of '{trait_name}', with the genetic advance as percent of mean (GAM = {gam:.2f}%) being HIGH. "
            "Useful selection response is achievable despite environmental influence. "
            "Both genetic and environmental management should be considered."
        )
    elif h2_class == "moderate" and gam_class == "Medium":
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) and genetic advance as percent "
            f"of mean (GAM = {gam:.2f}%) both indicate MODERATE genetic control for '{trait_name}'. "
            "Selection may be useful, though environmental factors remain important. "
            "Progress should be steady but not rapid."
        )
    elif h2_class == "moderate" and gam_class == "Low":
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) suggests MODERATE genetic control "
            f"of '{trait_name}', but the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW. "
            "Direct phenotypic selection may be slow. Consider investigating additive effects more carefully "
            "or combining selection with environmental optimization."
        )
    else:  # low h2
        interpretation = (
            f"The estimated broad-sense heritability (H2 = {h2:.3f}) indicates LOW genetic control of "
            f"'{trait_name}' under the present environment. Phenotypic variation is dominated by environmental "
            "factors and/or measurement variation. Direct phenotypic selection is unlikely to be reliable; "
            "focus on improving growing conditions and management practices."
        )

    # ── GCV vs PCV addendum ──────────────────────────────────────────────
    if gcv is not None and pcv is not None:
        try:
            gcv_pcv_sentence = _describe_gcv_pcv(gcv, pcv, trait_name)
            anova_f_env = float(anova_f_env) if anova_f_env is not None else 0.0
            anova_p_env = float(anova_p_env) if anova_p_env is not None else None
            anova_f_gxe = float(anova_f_gxe) if anova_f_gxe is not None else 0.0
            anova_p_gxe = float(anova_p_gxe) if anova_p_gxe is not None else None
            env_sentence = _describe_env_effects(
                f_env=anova_f_env,
                p_env=anova_p_env,
                f_gxe=anova_f_gxe,
                p_gxe=anova_p_gxe,
            )
            if gcv_pcv_sentence and env_sentence:
                interpretation += f"\n\n{gcv_pcv_sentence}\n\n{env_sentence}"
            elif gcv_pcv_sentence:
                interpretation += f"\n\n{gcv_pcv_sentence}"
            elif env_sentence:
                interpretation += f"\n\n{env_sentence}"
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
    """Classify GAM using the shared InterpretationEngine thresholds."""
    return InterpretationEngine.classify_gam(gam_percent)
