"""
Genetics Interpretation Engine (No circular dependencies)

Generates VALIDATED genetic parameters interpretation sections.
Returns strict GeneticsInterpretationSections objects, never freeform prose.
"""

from collections import defaultdict
from typing import Optional, Tuple
from interpretation_sections import GeneticsInterpretationSections
from interpretation import InterpretationEngine


def build_breeding_synthesis(trait_results: list[dict]) -> str:
    """
    Rule-based breeding synthesis engine.
    Each dict in trait_results must have:
      trait_name, h2, gam_class, top_genotype, f_gxe, p_gxe,
      genotype_means (list of dicts: {genotype, mean, rank, group})
    """

    FORBIDDEN = [
        "best genotype", "ideal parent", "confirmed donor parent",
        "recommended for release"
    ]

    n_traits = len(trait_results)
    if n_traits == 0:
        return ""

    paragraphs = []
    genotype_scores = {}

    for trait in trait_results:
        trait_name = str(trait.get("trait_name") or "this trait")
        means = trait.get("genotype_means", [])

        genotype_significant = trait.get("genotype_significant")
        p_value_raw = trait.get("p_value")
        f_value_raw = trait.get("f_value")

        if genotype_significant is None and p_value_raw is not None:
            try:
                genotype_significant = float(p_value_raw) <= 0.05
            except (TypeError, ValueError):
                genotype_significant = None

        if genotype_significant is False:
            try:
                f_text = f"{float(f_value_raw):.3f}" if f_value_raw is not None else "—"
            except (TypeError, ValueError):
                f_text = "—"
            try:
                p_text = f"{float(p_value_raw):.3f}" if p_value_raw is not None else "—"
            except (TypeError, ValueError):
                p_text = "—"

            paragraphs.append(
                f"Genotypic variation for {trait_name} was not statistically significant "
                f"(F = {f_text}, p = {p_text}). Genetic advance estimates "
                f"are unreliable for this trait under the present experimental conditions. "
                f"Increase sample size or evaluate across multiple environments before "
                f"making selection decisions."
            )
            continue

        groups = []
        for m in means:
            grp = str(m.get("group") or "").strip()
            if grp:
                groups.append(grp)
        if groups and len(set(groups)) == 1:
            paragraphs.append(
                "No genotype showed statistically superior performance. All genotypes were assigned to the same mean separation group."
            )
            continue

        if not means:
            continue
        n_genos = len(means)
        top_cutoff = max(1, round(n_genos * 0.25))

        for geno_data in means:
            geno = str(geno_data.get("genotype") or "")
            if not geno:
                continue
            rank = geno_data.get("rank", n_genos)

            if rank <= top_cutoff:
                perf_score = 3
            elif rank <= n_genos * 0.75:
                perf_score = 2
            else:
                perf_score = 1

            p_gxe = trait.get("p_gxe")
            if p_gxe is not None and p_gxe < 0.001:
                sig_score = 3
            elif p_gxe is not None and p_gxe < 0.05:
                sig_score = 2
            else:
                sig_score = 1

            f_gxe = trait.get("f_gxe", 0) or 0
            if f_gxe < 5:
                stab_score = 3
            elif f_gxe < 15:
                stab_score = 2
            else:
                stab_score = 1

            trait_strength = (
                0.5 * perf_score +
                0.2 * sig_score +
                0.3 * stab_score
            )

            if geno not in genotype_scores:
                genotype_scores[geno] = []
            genotype_scores[geno].append({
                "trait": trait["trait_name"],
                "trait_strength": trait_strength,
                "perf_score": perf_score,
                "stab_score": stab_score,
            })

    high_h2_traits = [t["trait_name"] for t in trait_results
                      if (t.get("h2") or 0) >= 0.80]
    if len(high_h2_traits) == n_traits:
        paragraphs.append(
            f"All {n_traits} traits showed high entry-mean broad-sense heritability "
            f"(H2 >= 0.80), indicating that genetic differences among genotypes "
            f"are reliably expressed across the tested environments. "
            f"Direct phenotypic selection is expected to be effective "
            f"for all traits evaluated in this study."
        )
    elif high_h2_traits:
        low_h2 = [t["trait_name"] for t in trait_results
                  if t["trait_name"] not in high_h2_traits]
        paragraphs.append(
            f"High heritability was observed for "
            f"{', '.join(high_h2_traits)}, making these traits most "
            f"amenable to direct selection. "
            + (f"{', '.join(low_h2)} showed comparatively lower "
               f"heritability and may require replicated multi-environment "
               f"evaluation before selection decisions are finalised."
               if low_h2 else "")
        )

    high_gam = [t["trait_name"] for t in trait_results
                if t.get("gam_class") == "High"]
    medium_gam = [t["trait_name"] for t in trait_results
                  if t.get("gam_class") == "Medium"]
    if high_gam:
        paragraphs.append(
            f"High genetic advance (GAM > 10%) was observed for "
            f"{', '.join(high_gam)}, indicating substantial scope for "
            f"improvement through selection in this population."
        )
    if medium_gam:
        paragraphs.append(
            f"Moderate genetic advance (GAM 5-10%) was observed for "
            f"{', '.join(medium_gam)}, indicating a meaningful but "
            f"moderate selection response under the current population "
            f"structure and environments tested."
        )

    # --- CROSS-TRAIT GENOTYPE SYNTHESIS ---
    # Step 1: Aggregate scores per genotype across all traits
    genotype_profile = {}
    for geno, scores in genotype_scores.items():
        n_scored = len(scores)
        if n_scored == 0:
            continue
        n_strong = sum(1 for s in scores if s["trait_strength"] >= 2.2)
        avg_stab = sum(s["stab_score"] for s in scores) / n_scored
        strong_trait_names = [s["trait"] for s in scores
                              if s["trait_strength"] >= 2.2]
        all_trait_names = [s["trait"] for s in scores]
        genotype_profile[geno] = {
            "n_traits_total": n_scored,
            "n_traits_strong": n_strong,
            "avg_stab": avg_stab,
            "is_stable": avg_stab >= 2.0,
            "is_multi_trait": n_strong >= 2,
            "strong_traits": strong_trait_names,
            "all_traits": all_trait_names,
        }

    # Step 2: Classify each genotype and generate ONE paragraph per genotype
    for geno, profile in sorted(genotype_profile.items()):
        n_strong = profile["n_traits_strong"]
        n_total = profile["n_traits_total"]
        is_stable = profile["is_stable"]
        strong_traits = profile["strong_traits"]
        all_traits = profile["all_traits"]

        if n_strong == n_total and n_total >= 2:
            # Multi-trait elite - strong in ALL evaluated traits
            if is_stable:
                paragraphs.append(
                    f"{geno} consistently ranked among the top-performing "
                    f"genotypes across all {n_total} evaluated traits "
                    f"({', '.join(all_traits)}), with relatively consistent "
                    f"expression across environments. It is a promising "
                    f"candidate for inclusion in broad-based breeding programmes."
                )
            else:
                paragraphs.append(
                    f"{geno} consistently ranked among the top-performing "
                    f"genotypes across all {n_total} evaluated traits "
                    f"({', '.join(all_traits)}), indicating strong overall "
                    f"genetic potential. However, significant "
                    f"genotype-by-environment interaction across traits "
                    f"suggests that further stability assessment is required "
                    f"before deployment in breeding programmes."
                )

        elif n_strong >= 2:
            # Multi-trait strong - strong in most but not all traits
            if is_stable:
                paragraphs.append(
                    f"{geno} showed strong performance across multiple traits "
                    f"({', '.join(strong_traits)}), making it a promising "
                    f"candidate for crossing programmes targeting combined "
                    f"trait improvement."
                )
            else:
                paragraphs.append(
                    f"{geno} showed strong performance across multiple traits "
                    f"({', '.join(strong_traits)}), indicating broad genetic "
                    f"potential. Genotype-by-environment interaction warrants "
                    f"stability evaluation before selection decisions are finalised."
                )

    trait_specific_groups = defaultdict(list)

    for geno, profile in sorted(genotype_profile.items()):
        if profile["n_traits_strong"] == 1:
            trait_specific_groups[profile["strong_traits"][0]].append(geno)

    for trait_name, genos in sorted(trait_specific_groups.items()):
        if len(genos) == 1:
            paragraphs.append(
                f"{genos[0]} showed high performance for {trait_name}, "
                f"although genotype-by-environment interaction suggests "
                f"its use in breeding should account for specific adaptation."
            )
        else:
            geno_list = " and ".join(genos)
            paragraphs.append(
                f"{geno_list} both showed strong performance for "
                f"{trait_name}, though genotype-by-environment interaction "
                f"indicates that specific adaptation should be considered "
                f"in their use for breeding programmes."
            )

    unstable = [t["trait_name"] for t in trait_results
                if (t.get("f_gxe") or 0) > 15]
    if unstable:
        paragraphs.append(
            f"Substantial genotype-by-environment interaction was "
            f"detected for {', '.join(unstable)}. Stability analysis "
            f"(Eberhart-Russell or GGE biplot) is recommended before "
            f"finalising genotype selection for "
            f"{'this trait' if len(unstable) == 1 else 'these traits'}. "
            f"The nature of this interaction (crossover vs non-crossover) "
            f"requires further stability analysis such as GGE biplot or "
            f"Eberhart-Russell regression before genotype rankings can be "
            f"considered environment-stable."
        )

    paragraphs.append(
        "These conclusions are based on data from this specific experiment "
        "and should be interpreted within that context. Findings should be "
        "validated across additional environments and seasons before "
        "breeding decisions are finalised."
    )

    synthesis = "\n\n".join(paragraphs)
    lower_synthesis = synthesis.lower()
    for phrase in FORBIDDEN:
        if phrase in lower_synthesis:
            synthesis = synthesis.replace(phrase, "candidate")
    return synthesis


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
