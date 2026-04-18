"""
Trait Association Interpretation Engine (No circular dependencies)

Generates academic-grade trait association interpretation.
No imports from routes modules to avoid circular dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple


def _compute_risk_flags(n: int, analysis_unit: str, gxe_significant: bool) -> List[str]:
    """
    Compute risk flags for a trait association / correlation analysis.

    Defined here (not in routes) so it can be imported by any module —
    including trait_relationships_routes — without creating circular imports.
    """
    flags: List[str] = []

    if n < 10:
        flags.append("small_sample_size")

    if analysis_unit == "genotype_mean":
        flags.append("genotype_mean_based")

    if gxe_significant:
        flags.append("gxe_significant")

    # Always present: pairwise N is not tracked in the current engine
    flags.append("pairwise_n_not_tracked")

    return flags


def generate_trait_association_interpretation(
    n_traits: int,
    n_observations: int,
    n_significant_pairs: int,
    strongest_positive: Optional[Dict[str, Any]],
    strongest_negative: Optional[Dict[str, Any]],
    risk_flags: List[str],
    gxe_significant: bool,
    environment_context: str
) -> str:
    """
    Generate academic-grade trait association interpretation following VivaSense standards.

    Rules enforced:
    - strongest_positive / strongest_negative must be the strongest SIGNIFICANT pairs.
      Callers are responsible for filtering to p <= alpha before passing them in.
      Non-significant pairs must never be passed here — they will be described as
      biologically meaningful, which is scientifically incorrect.
    - If strongest_negative is None (no significant negative pair exists), the report
      explicitly says so rather than omitting the topic.
    - Trade-off language is only emitted when a significant negative pair is present.
    - If n_observations < 10 -> include "preliminary evidence".
    - If "pairwise_n_not_tracked" in risk_flags -> state confidence is limited.
    - If gxe_significant -> warn that relationships may vary across environments.
    - Do not recommend indirect selection based on correlations alone.
    """
    sections = []

    # 1. Overview
    overview = []
    overview.append(f"This analysis examined trait associations among {n_traits} traits")
    if environment_context == "multi_environment":
        overview.append("across multiple environments")
    overview.append(f"using {n_observations} genotype mean(s).")

    if n_observations < 10:
        overview.append("The sample size is limited, indicating preliminary evidence.")

    sections.append(" ".join(overview))

    # 2. Significant Associations
    if n_significant_pairs > 0:
        assoc = []
        assoc.append(
            f"Significant trait associations (p < 0.05) were detected for "
            f"{n_significant_pairs} pair(s)."
        )

        # Positive — only describe if a significant positive pair was found
        if strongest_positive:
            assoc.append(
                f"The strongest significant positive association was between "
                f"{strongest_positive.get('trait_1')} and {strongest_positive.get('trait_2')} "
                f"(r = {strongest_positive.get('r', 0):.2f}), "
                "suggesting these traits may tend to improve together."
            )
        else:
            assoc.append("No significant positive associations were detected.")

        # Negative — only describe if a significant negative pair was found.
        # Never describe a non-significant negative correlation as a trade-off.
        if strongest_negative:
            assoc.append(
                f"The strongest significant negative association was between "
                f"{strongest_negative.get('trait_1')} and {strongest_negative.get('trait_2')} "
                f"(r = {strongest_negative.get('r', 0):.2f}), "
                "which may reflect a trade-off in genetic or physiological control."
            )
        else:
            assoc.append("No significant negative associations were detected.")

        sections.append(" ".join(assoc))
    else:
        sections.append(
            "No significant trait associations were detected at the chosen significance level (p < 0.05). "
            "This suggests that the traits examined are largely independent in their genetic or environmental control."
        )

    # 3. Pairwise Sample Size Limitation
    if "pairwise_n_not_tracked" in risk_flags:
        sections.append(
            "However, the confidence that can be placed in these association estimates is limited by the lack "
            "of pairwise sample size tracking. Each pair may have been based on different numbers of complete observations. "
            "Conclusions should be interpreted cautiously and validated in independent datasets."
        )

    # 4. GxE Interaction
    if gxe_significant:
        sections.append(
            "The significant genotype × environment interaction detected in the ANOVA suggests that trait "
            "relationships may vary across environments. Therefore, trait associations observed in one environment "
            "may not hold in another. Selection strategies based on these correlations should account for this variability."
        )

    # 5. Indirect Selection Constraints
    constraint = []
    constraint.append(
        "Trait associations can support indirect selection only when combined with strong genetic evidence "
        "(e.g., high heritability of the correlated trait) and validation across target environments."
    )

    if "pairwise_n_not_tracked" in risk_flags or n_observations < 10:
        constraint.append(
            "At present, the limitations noted above preclude firm recommendations for indirect selection."
        )

    if gxe_significant:
        constraint.append(
            "Further stability analysis is needed before implementing environment-specific selection strategies."
        )

    sections.append(" ".join(constraint))

    return "\n\n".join(sections)


# ============================================================================
# DUAL-MODE CORRELATION INTERPRETATION
# ============================================================================

def _r_strength_label(r: Optional[float]) -> str:
    """Return a plain-English strength label for a correlation coefficient."""
    if r is None:
        return "N/A"
    abs_r = abs(r)
    if abs_r >= 0.90:
        return "very strong"
    if abs_r >= 0.70:
        return "strong"
    if abs_r >= 0.50:
        return "moderate"
    if abs_r >= 0.30:
        return "weak"
    return "very weak"


def build_comparison_table(
    trait_names: List[str],
    genotypic: Dict[str, Any],
    phenotypic: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build a structured pairwise comparison table for all off-diagonal trait pairs.

    Each entry contains both phenotypic and genotypic r, p, and adjusted p for
    the same pair, enabling side-by-side display in the UI.
    """
    n = len(trait_names)
    r_g = genotypic.get("r_matrix", [])
    p_g = genotypic.get("p_matrix", [])
    pa_g = genotypic.get("p_adj_matrix", [])
    ci_lo_g = genotypic.get("ci_lower_matrix", [])
    ci_hi_g = genotypic.get("ci_upper_matrix", [])

    r_p = phenotypic.get("r_matrix", [])
    p_p = phenotypic.get("p_matrix", [])
    pa_p = phenotypic.get("p_adj_matrix", [])
    ci_lo_p = phenotypic.get("ci_lower_matrix", [])
    ci_hi_p = phenotypic.get("ci_upper_matrix", [])

    table = []
    for i in range(n):
        for j in range(i + 1, n):
            def _safe(mat: list, row: int, col: int) -> Optional[float]:
                try:
                    v = mat[row][col]
                    return None if v is None else float(v)
                except (IndexError, TypeError):
                    return None

            rg = _safe(r_g, i, j)
            rp = _safe(r_p, i, j)
            pg = _safe(p_g, i, j)
            pp = _safe(p_p, i, j)
            pag = _safe(pa_g, i, j)
            pap = _safe(pa_p, i, j)
            cilo_g = _safe(ci_lo_g, i, j)
            cihi_g = _safe(ci_hi_g, i, j)
            cilo_p = _safe(ci_lo_p, i, j)
            cihi_p = _safe(ci_hi_p, i, j)

            sig_g = pg is not None and pg < 0.05
            sig_p = pp is not None and pp < 0.05

            if rg is not None and rp is not None:
                if rg * rp < 0:
                    agreement = "sign_discordant"
                elif sig_g and sig_p:
                    agreement = "both_significant"
                elif sig_g or sig_p:
                    agreement = "one_significant"
                else:
                    agreement = "both_non_significant"
            else:
                agreement = "data_insufficient"

            table.append({
                "trait_1": trait_names[i],
                "trait_2": trait_names[j],
                "phenotypic_r": rp,
                "phenotypic_p": pp,
                "phenotypic_p_adj": pap,
                "phenotypic_ci_lower": cilo_p,
                "phenotypic_ci_upper": cihi_p,
                "phenotypic_significant": sig_p,
                "genotypic_r": rg,
                "genotypic_p": pg,
                "genotypic_p_adj": pag,
                "genotypic_ci_lower": cilo_g,
                "genotypic_ci_upper": cihi_g,
                "genotypic_significant": sig_g,
                "agreement": agreement,
            })

    return table


def generate_dual_mode_correlation_interpretation(
    trait_names: List[str],
    genotypic: Dict[str, Any],
    phenotypic: Dict[str, Any],
    user_objective: str
) -> str:
    """
    Generate biometrically correct dual-mode correlation interpretation.

    Sections:
      1. Objective framing
      2. Statistical context (n, df, critical r per mode + critical r comparison)
      3. Warning system (low power, pseudo-replication, multiple testing)
      4. Mode distinction (what each mode means — no causation claims)
      5. Pairwise comparison (genotypic vs phenotypic per pair)
      6. Sign discordance alerts
      7. Objective-specific closing guidance

    Rules enforced:
    - Never claims raw data is "better for breeding".
    - Never implies genetic causation from correlation.
    - Genotypic mean correlation is explicitly distinguished from true genetic correlation.
    - Warns when n < 10 (low power).
    - Warns about pseudo-replication in phenotypic mode.
    - Warns about multiple testing when > 2 traits.
    """
    sections: List[str] = []
    n_traits = len(trait_names)

    n_geno  = genotypic.get("n_observations", 0)
    n_pheno = phenotypic.get("n_observations", 0)
    df_geno  = genotypic.get("df")
    df_pheno = phenotypic.get("df")
    crit_r_geno  = genotypic.get("critical_r")
    crit_r_pheno = phenotypic.get("critical_r")

    # ------------------------------------------------------------------
    # 1. Objective framing
    # ------------------------------------------------------------------
    objective_framing = {
        "Field understanding": (
            "The stated objective is field-level understanding, which focuses on how traits "
            "co-vary across experimental units. Phenotypic correlation is most directly relevant "
            "here, though it reflects genetic, environmental, and management effects simultaneously."
        ),
        "Genotype comparison": (
            "The stated objective is genotype comparison, which focuses on how average genotype "
            "performances correlate across traits. Genotypic mean correlation is more informative "
            "for this purpose as it reduces within-genotype replication noise."
        ),
        "Breeding decision": (
            "The stated objective is breeding decision support. This requires careful interpretation: "
            "genotypic mean correlation is NOT equivalent to true quantitative genetic correlation, "
            "which requires variance component estimation. Treat both estimates as preliminary "
            "evidence requiring corroboration from heritability analysis and multi-environment trials."
        ),
    }
    sections.append(
        f"Dual-mode correlation analysis was conducted for {n_traits} trait(s). "
        + objective_framing.get(user_objective, "")
    )

    # ------------------------------------------------------------------
    # 2. Statistical context block (n, df, critical r)
    # ------------------------------------------------------------------
    def _fmt_crit(crit: Optional[float]) -> str:
        return f"|r| ≥ {abs(crit):.3f}" if crit is not None else "N/A"

    def _fmt_df(df: Optional[int]) -> str:
        return str(df) if df is not None else "N/A"

    stat_lines = [
        "Statistical Context:",
        f"  Phenotypic mode : n = {n_pheno} observations,  "
        f"df = {_fmt_df(df_pheno)},  critical r (α = 0.05): {_fmt_crit(crit_r_pheno)}",
        f"  Genotypic mode  : n = {n_geno} genotype means, "
        f"df = {_fmt_df(df_geno)},  critical r (α = 0.05): {_fmt_crit(crit_r_geno)}",
    ]

    if crit_r_geno is not None and crit_r_pheno is not None:
        cg = abs(crit_r_geno)
        cp = abs(crit_r_pheno)
        diff = abs(cg - cp)
        if diff < 0.02:
            stat_lines.append(
                f"  Critical r Comparison: Both modes have similar significance thresholds "
                f"(phenotypic {cp:.3f}, genotypic {cg:.3f}), suggesting similar sample sizes."
            )
        elif cg > cp:
            stat_lines.append(
                f"  Critical r Comparison: Genotypic mode requires a higher minimum |r| "
                f"({cg:.3f}) than phenotypic mode ({cp:.3f}) to reach significance. "
                "This is expected: fewer genotype means are available than total plot observations, "
                "so the genotypic mode has lower statistical power."
            )
        else:
            stat_lines.append(
                f"  Critical r Comparison: Phenotypic mode requires a higher minimum |r| "
                f"({cp:.3f}) than genotypic mode ({cg:.3f}). "
                "This unusual pattern suggests the number of raw observations is smaller than "
                "the number of genotype means — review the data structure."
            )

    sections.append("\n".join(stat_lines))

    # ------------------------------------------------------------------
    # 3. Warning system
    # ------------------------------------------------------------------
    warnings: List[str] = []

    if n_geno < 10:
        warnings.append(
            f"Low power: Only {n_geno} genotype mean(s) available in genotypic mode (n < 10). "
            "Correlation estimates are unstable and confidence intervals are wide. "
            "Interpret all genotypic results with caution."
        )

    if n_pheno > n_geno:
        warnings.append(
            f"Possible pseudo-replication: Phenotypic mode uses all {n_pheno} observations, "
            "including replication. In replicated field trials each plot is not an independent "
            "biological unit — this inflates degrees of freedom and can artificially inflate "
            "statistical significance. Phenotypic r should be treated as descriptive."
        )

    if n_traits > 2:
        n_pairs = n_traits * (n_traits - 1) // 2
        warnings.append(
            f"Multiple testing: {n_pairs} pairwise comparisons conducted simultaneously. "
            "FDR-adjusted p-values are reported alongside raw p-values. "
            "Use adjusted p-values for strict significance inference to control the false "
            "discovery rate."
        )

    if warnings:
        sections.append(
            "Statistical Warnings:\n" + "\n".join(f"• {w}" for w in warnings)
        )

    # ------------------------------------------------------------------
    # 4. Mode distinction
    # ------------------------------------------------------------------
    sections.append(
        "Mode Distinction:\n"
        "• Phenotypic Correlation: Reflects co-variation among individual experimental units "
        "(plots or plants). Captures the joint effects of genetic variation, environmental "
        "heterogeneity, and management. Useful for understanding field-level co-variation and "
        "physiological relationships, but does not isolate genetic from environmental effects.\n"
        "• Genotypic Mean Correlation: Reflects co-variation among genotype average performances "
        "across replications. By averaging out within-genotype replication, it reduces "
        "environmental noise and is informative for genotype comparison. "
        "IMPORTANT: This is NOT a true quantitative genetic correlation — it does not partition "
        "genetic from residual variance using mixed models. Correlation does not imply genetic "
        "causation, and elevated r between genotype means does not confirm a shared genetic basis."
    )

    # ------------------------------------------------------------------
    # 5. Pairwise comparison narrative
    # ------------------------------------------------------------------
    comparison = build_comparison_table(trait_names, genotypic, phenotypic)
    if comparison:
        pair_lines = ["Pairwise Comparison (Genotypic vs Phenotypic r):"]
        discordant: List[Tuple[str, str, float, float]] = []

        for entry in comparison:
            t1, t2 = entry["trait_1"], entry["trait_2"]
            rg = entry["genotypic_r"]
            rp = entry["phenotypic_r"]
            pg = entry["genotypic_p"]
            pp = entry["phenotypic_p"]
            pag = entry["genotypic_p_adj"]
            pap = entry["phenotypic_p_adj"]
            sig_g = entry["genotypic_significant"]
            sig_p = entry["phenotypic_significant"]

            def _rv(v: Optional[float]) -> str:
                return f"{v:.3f}" if v is not None else "N/A"

            def _pv(v: Optional[float]) -> str:
                if v is None:
                    return "N/A"
                return f"{v:.4f}" if v >= 0.0001 else "< 0.0001"

            sig_g_label = "sig." if sig_g else "ns"
            sig_p_label = "sig." if sig_p else "ns"
            pair_lines.append(
                f"  {t1} × {t2}:\n"
                f"    Genotypic  r = {_rv(rg)} ({sig_g_label}), p = {_pv(pg)}, p_adj(FDR) = {_pv(pag)}\n"
                f"    Phenotypic r = {_rv(rp)} ({sig_p_label}), p = {_pv(pp)}, p_adj(FDR) = {_pv(pap)}"
            )

            if entry["agreement"] == "sign_discordant" and rg is not None and rp is not None:
                discordant.append((t1, t2, rg, rp))

        sections.append("\n".join(pair_lines))

        if discordant:
            disc_items = "; ".join(
                f"{t1} × {t2} (genotypic r = {rg:.3f}, phenotypic r = {rp:.3f})"
                for t1, t2, rg, rp in discordant
            )
            sections.append(
                "Sign Discordance Alert:\n"
                f"The following pair(s) have opposite signs in the two modes: {disc_items}. "
                "This means the traits correlate positively in one analysis unit and negatively "
                "in the other. Possible causes include environmental confounding, scale effects, "
                "or a Simpson's paradox-type aggregation artefact. "
                "Do not act on either estimate without independent verification."
            )

    # ------------------------------------------------------------------
    # 6. Objective-specific closing guidance
    # ------------------------------------------------------------------
    objective_closing = {
        "Field understanding": (
            "For field-level understanding, phenotypic correlations describe the co-variation "
            "observed in this trial. Differences between phenotypic and genotypic r indicate "
            "how much of the observed co-variation is attributable to within-genotype "
            "environmental effects versus consistent differences between genotypes."
        ),
        "Genotype comparison": (
            "For genotype comparison, genotypic mean correlations are more informative. "
            "They reflect aggregate performance differences between genotypes, filtering out "
            "replication noise. Treat these as descriptive rankings — not as genetic parameters. "
            "Confidence in any ranking increases with more replications and a larger genotype panel."
        ),
        "Breeding decision": (
            "For breeding decisions, both phenotypic and genotypic mean correlation are insufficient "
            "as stand-alone evidence. Heritability estimates, variance component analysis, and "
            "validation across target environments are necessary to confirm that observed "
            "associations have a heritable basis amenable to selection. "
            "Avoid implementing indirect selection strategies based solely on this analysis."
        ),
    }
    if user_objective in objective_closing:
        sections.append(
            f"Guidance for '{user_objective}':\n" + objective_closing[user_objective]
        )

    return "\n\n".join(sections)
