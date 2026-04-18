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
    between_genotype: Dict[str, Any],
    phenotypic: Dict[str, Any],
    genotypic_vc: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a structured pairwise comparison table for all off-diagonal trait pairs.

    Each entry contains phenotypic, between-genotype, and (optionally)
    variance-component genotypic r/p/CI for side-by-side display.
    """
    n = len(trait_names)

    def _mat(d: Optional[Dict[str, Any]], key: str) -> list:
        return d.get(key, []) if d else []

    r_bg  = _mat(between_genotype, "r_matrix")
    p_bg  = _mat(between_genotype, "p_matrix")
    pa_bg = _mat(between_genotype, "p_adj_matrix")
    ci_lo_bg = _mat(between_genotype, "ci_lower_matrix")
    ci_hi_bg = _mat(between_genotype, "ci_upper_matrix")

    r_p  = _mat(phenotypic, "r_matrix")
    p_p  = _mat(phenotypic, "p_matrix")
    pa_p = _mat(phenotypic, "p_adj_matrix")
    ci_lo_p = _mat(phenotypic, "ci_lower_matrix")
    ci_hi_p = _mat(phenotypic, "ci_upper_matrix")

    r_vc  = _mat(genotypic_vc, "r_matrix")
    p_vc  = _mat(genotypic_vc, "p_matrix")
    pa_vc = _mat(genotypic_vc, "p_adj_matrix")
    ci_lo_vc = _mat(genotypic_vc, "ci_lower_matrix")
    ci_hi_vc = _mat(genotypic_vc, "ci_upper_matrix")

    table = []
    for i in range(n):
        for j in range(i + 1, n):
            def _safe(mat: list, row: int, col: int) -> Optional[float]:
                try:
                    v = mat[row][col]
                    return None if v is None else float(v)
                except (IndexError, TypeError):
                    return None

            rbg = _safe(r_bg, i, j)
            rp  = _safe(r_p, i, j)
            rvc = _safe(r_vc, i, j)
            pbg = _safe(p_bg, i, j)
            pp  = _safe(p_p, i, j)
            pvc = _safe(p_vc, i, j)

            sig_bg = pbg is not None and pbg < 0.05
            sig_p  = pp  is not None and pp  < 0.05
            sig_vc = pvc is not None and pvc < 0.05

            if rbg is not None and rp is not None:
                if rbg * rp < 0:
                    agreement = "sign_discordant"
                elif sig_bg and sig_p:
                    agreement = "both_significant"
                elif sig_bg or sig_p:
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
                "phenotypic_p_adj": _safe(pa_p, i, j),
                "phenotypic_ci_lower": _safe(ci_lo_p, i, j),
                "phenotypic_ci_upper": _safe(ci_hi_p, i, j),
                "phenotypic_significant": sig_p,
                "between_genotype_r": rbg,
                "between_genotype_p": pbg,
                "between_genotype_p_adj": _safe(pa_bg, i, j),
                "between_genotype_ci_lower": _safe(ci_lo_bg, i, j),
                "between_genotype_ci_upper": _safe(ci_hi_bg, i, j),
                "between_genotype_significant": sig_bg,
                "genotypic_vc_r": rvc,
                "genotypic_vc_p": pvc,
                "genotypic_vc_p_adj": _safe(pa_vc, i, j),
                "genotypic_vc_ci_lower": _safe(ci_lo_vc, i, j),
                "genotypic_vc_ci_upper": _safe(ci_hi_vc, i, j),
                "genotypic_vc_significant": sig_vc,
                "agreement": agreement,
            })

    return table


def generate_dual_mode_correlation_interpretation(
    trait_names: List[str],
    between_genotype: Dict[str, Any],
    phenotypic: Dict[str, Any],
    user_objective: str,
    genotypic_vc: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate biometrically correct three-mode correlation interpretation.

    Modes:
      phenotypic       — all observations (field-level relationship)
      between_genotype — genotype means (association among genotype means; not a true genetic correlation)
      genotypic_vc     — variance-component based (relevant to breeding inference); optional

    Sections:
      1. Objective framing
      2. Statistical context (n, df, critical r per mode)
      3. Effective sample size clarification (phenotypic mode)
      4. Warning system (low power, pseudo-replication, multiple testing)
      5. Mode distinction with correct labels
      6. Pairwise comparison across all available modes
      7. Sign discordance alerts
      8. Breeding decision safeguards
      9. Objective-specific closing guidance
    """
    sections: List[str] = []
    n_traits = len(trait_names)

    n_bg   = between_genotype.get("n_observations", 0)
    n_pheno = phenotypic.get("n_observations", 0)
    n_vc    = genotypic_vc.get("n_observations", 0) if genotypic_vc else None

    df_bg    = between_genotype.get("df")
    df_pheno = phenotypic.get("df")
    df_vc    = genotypic_vc.get("df") if genotypic_vc else None

    crit_r_bg    = between_genotype.get("critical_r")
    crit_r_pheno = phenotypic.get("critical_r")
    crit_r_vc    = genotypic_vc.get("critical_r") if genotypic_vc else None

    # Keep local aliases for compatibility with downstream sections
    n_geno       = n_bg
    crit_r_geno  = crit_r_bg

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
            "performances correlate across traits. Between-genotype association (computed from "
            "genotype means) is most informative for this purpose as it reduces within-genotype "
            "replication noise. Note: this is NOT a true genetic parameter."
        ),
        "Breeding decision": (
            "The stated objective is breeding decision support. Variance-component-based genotypic "
            "correlation is the most appropriate mode for this purpose, as it estimates genetic "
            "co-variation using bivariate REML. Where not available, between-genotype association "
            "is reported as a proxy, but it is NOT equivalent to true quantitative genetic correlation. "
            "Treat all estimates as preliminary evidence requiring corroboration from heritability "
            "analysis and multi-environment trials."
        ),
    }
    sections.append(
        f"Three-mode correlation analysis was conducted for {n_traits} trait(s). "
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
        f"  Phenotypic mode            : n = {n_pheno} observations, "
        f"df = {_fmt_df(df_pheno)},  critical r (α = 0.05): {_fmt_crit(crit_r_pheno)}",
        f"  Between-genotype mode      : n = {n_bg} genotype means, "
        f"df = {_fmt_df(df_bg)},  critical r (α = 0.05): {_fmt_crit(crit_r_bg)}",
    ]
    if genotypic_vc is not None:
        stat_lines.append(
            f"  Genotypic (VC-based) mode  : n = {n_vc} genotypes, "
            f"df = {_fmt_df(df_vc)},  critical r (α = 0.05): {_fmt_crit(crit_r_vc)}"
        )
    else:
        stat_lines.append(
            "  Genotypic (VC-based) mode  : not available "
            "(requires sommer package and sufficient data per pair)"
        )

    if crit_r_bg is not None and crit_r_pheno is not None:
        cg = abs(crit_r_bg)
        cp = abs(crit_r_pheno)
        diff = abs(cg - cp)
        if diff < 0.02:
            stat_lines.append(
                f"  Critical r Comparison: Phenotypic ({cp:.3f}) and between-genotype ({cg:.3f}) "
                "modes have similar significance thresholds, suggesting similar effective sample sizes."
            )
        elif cg > cp:
            stat_lines.append(
                f"  Critical r Comparison: Between-genotype mode requires a higher minimum |r| "
                f"({cg:.3f}) than phenotypic mode ({cp:.3f}) to reach significance — expected, "
                "as fewer genotype means are available than total plot observations."
            )
        else:
            stat_lines.append(
                f"  Critical r Comparison: Phenotypic mode requires a higher minimum |r| "
                f"({cp:.3f}) than between-genotype mode ({cg:.3f}). "
                "This unusual pattern suggests the raw observation count is smaller than the "
                "number of genotype means — review the data structure."
            )

    sections.append("\n".join(stat_lines))

    # ------------------------------------------------------------------
    # 3. Effective Sample Size Clarification (Phenotypic Mode)
    # ------------------------------------------------------------------
    if n_pheno > n_bg:
        sections.append(
            f"Effective Sample Size Note (Phenotypic Mode):\n"
            f"Although n = {n_pheno} observations were used for phenotypic correlations, "
            f"these are structured within {n_bg} genotypes and replications. "
            f"The number of truly independent experimental units may be lower than {n_pheno}, "
            f"potentially affecting the reliability of significance tests."
        )

    # ------------------------------------------------------------------
    # 4. Warning system
    # ------------------------------------------------------------------
    warn_list: List[str] = []

    if n_bg < 10:
        warn_list.append(
            f"Low power: Only {n_bg} genotype mean(s) available for between-genotype mode (n < 10). "
            "Correlation estimates are unstable and confidence intervals are wide. "
            "Interpret all between-genotype results with caution."
        )

    if n_pheno > n_bg:
        warn_list.append(
            f"Possible pseudo-replication: Phenotypic mode uses all {n_pheno} observations, "
            "including replication. In replicated field trials each plot is not an independent "
            "biological unit — this inflates degrees of freedom and can artificially inflate "
            "statistical significance. Phenotypic r should be treated as descriptive."
        )

    if n_traits > 2:
        n_pairs = n_traits * (n_traits - 1) // 2
        warn_list.append(
            f"Multiple testing: {n_pairs} pairwise comparisons conducted simultaneously. "
            "FDR-adjusted p-values are reported alongside raw p-values. "
            "Only correlations remaining significant after FDR adjustment should be considered reliable "
            "for strict statistical inference to control the false discovery rate."
        )

    if warn_list:
        sections.append(
            "Statistical Warnings:\n" + "\n".join(f"• {w}" for w in warn_list)
        )

    # ------------------------------------------------------------------
    # 5. Mode distinction with correct scientific labels
    # ------------------------------------------------------------------
    vc_mode_line = (
        "• Genotypic Correlation (variance-component based): Estimated from bivariate REML "
        "mixed models (sommer). Partitions genetic variance and covariance using genotype as a "
        "random effect. The formula is rg = Covg(X,Y) / sqrt(Vg(X) × Vg(Y)). "
        "This is the correct parameter for breeding inference, though precision depends on "
        "the number of genotypes and replication structure."
        if genotypic_vc is not None
        else "• Genotypic Correlation (variance-component based): Not computed — "
        "sommer package unavailable or insufficient data for bivariate REML."
    )
    sections.append(
        "Correlation Type and Mode Distinction:\n"
        "• Phenotypic Correlation (field-level relationship): Reflects co-variation among individual "
        "experimental units (plots or plants). Captures the joint effects of genetic variation, "
        "environmental heterogeneity, and management. Useful for understanding field-level co-variation "
        "but does not isolate genetic from environmental effects.\n"
        "• Between-Genotype Association (from genotype means): Reflects co-variation among genotype "
        "average performances across replications. Reduces within-genotype replication noise and is "
        "informative for genotype comparison. IMPORTANT: This is NOT a true quantitative genetic "
        "correlation — it does not partition genetic from residual variance using mixed models. "
        "Elevated association between genotype means does not confirm a shared genetic basis.\n"
        + vc_mode_line
    )

    # ------------------------------------------------------------------
    # 6. Pairwise comparison narrative
    # ------------------------------------------------------------------
    comparison = build_comparison_table(trait_names, between_genotype, phenotypic, genotypic_vc)
    if comparison:
        pair_lines = ["Pairwise Comparison (all modes):"]
        discordant: List[Tuple[str, str, float, float]] = []
        wide_ci_pairs: List[str] = []
        stability_warnings: List[str] = []
        fdr_survivors: List[str] = []
        fdr_non_survivors: List[str] = []

        for entry in comparison:
            t1, t2 = entry["trait_1"], entry["trait_2"]
            rbg = entry["between_genotype_r"]
            rp  = entry["phenotypic_r"]
            rvc = entry["genotypic_vc_r"]
            pbg = entry["between_genotype_p"]
            pp  = entry["phenotypic_p"]
            pvc = entry["genotypic_vc_p"]
            pabg = entry["between_genotype_p_adj"]
            pap  = entry["phenotypic_p_adj"]
            pavc = entry["genotypic_vc_p_adj"]
            cilo_bg = entry["between_genotype_ci_lower"]
            cihi_bg = entry["between_genotype_ci_upper"]
            cilo_p  = entry["phenotypic_ci_lower"]
            cihi_p  = entry["phenotypic_ci_upper"]
            sig_bg = entry["between_genotype_significant"]
            sig_p  = entry["phenotypic_significant"]
            sig_vc = entry["genotypic_vc_significant"]

            def _rv(v: Optional[float]) -> str:
                return f"{v:.3f}" if v is not None else "N/A"

            def _pv(v: Optional[float]) -> str:
                if v is None:
                    return "N/A"
                return f"{v:.4f}" if v >= 0.0001 else "< 0.0001"

            sig_bg_label = "sig." if sig_bg else "ns"
            sig_p_label  = "sig." if sig_p  else "ns"
            sig_vc_label = "sig." if sig_vc else ("ns" if rvc is not None else "—")

            # Wide CI flags
            if cilo_bg is not None and cihi_bg is not None and (cihi_bg - cilo_bg) > 0.6:
                wide_ci_pairs.append(f"{t1} × {t2} (between-genotype)")
            if cilo_p is not None and cihi_p is not None and (cihi_p - cilo_p) > 0.6:
                wide_ci_pairs.append(f"{t1} × {t2} (phenotypic)")

            # Stability warning
            if n_bg < 10 and rbg is not None and abs(rbg) > 0.8:
                stability_warnings.append(f"{t1} × {t2} (between-genotype r = {rbg:.3f})")

            # FDR tracking
            if pabg is not None and pabg < 0.05:
                fdr_survivors.append(f"{t1} × {t2} (between-genotype)")
            elif sig_bg and (pabg is None or pabg >= 0.05):
                fdr_non_survivors.append(f"{t1} × {t2} (between-genotype)")
            if pap is not None and pap < 0.05:
                fdr_survivors.append(f"{t1} × {t2} (phenotypic)")
            elif sig_p and (pap is None or pap >= 0.05):
                fdr_non_survivors.append(f"{t1} × {t2} (phenotypic)")
            if pavc is not None and pavc < 0.05:
                fdr_survivors.append(f"{t1} × {t2} (genotypic VC)")
            elif sig_vc and (pavc is None or pavc >= 0.05):
                fdr_non_survivors.append(f"{t1} × {t2} (genotypic VC)")

            vc_line = (
                f"    Genotypic VC   r = {_rv(rvc)} ({sig_vc_label}), p = {_pv(pvc)}, p_adj(FDR) = {_pv(pavc)}"
                if rvc is not None else
                "    Genotypic VC   : not available"
            )
            pair_lines.append(
                f"  {t1} × {t2}:\n"
                f"    Phenotypic          r = {_rv(rp)} ({sig_p_label}), p = {_pv(pp)}, p_adj(FDR) = {_pv(pap)}\n"
                f"    Between-genotype    r = {_rv(rbg)} ({sig_bg_label}), p = {_pv(pbg)}, p_adj(FDR) = {_pv(pabg)}\n"
                + vc_line
            )

            if entry["agreement"] == "sign_discordant" and rbg is not None and rp is not None:
                discordant.append((t1, t2, rbg, rp))

        sections.append("\n".join(pair_lines))

        # Confidence Interval alerts
        if wide_ci_pairs:
            sections.append(
                "Confidence Interval Alert:\n"
                f"The following correlation(s) have wide confidence intervals (> 0.6), "
                f"indicating high uncertainty: {', '.join(wide_ci_pairs)}."
            )

        if n_bg < 10:
            sections.append(
                "Confidence Interval Note:\n"
                "Due to the small number of genotype means (n < 10), confidence intervals "
                "are expected to be wide, reflecting higher uncertainty in all modes."
            )

        if stability_warnings:
            sections.append(
                "Stability Warning:\n"
                "High correlation estimates with small genotype panels may be sensitive to "
                f"individual genotypes: {', '.join(stability_warnings)}. "
                "Validate with larger populations before acting on these estimates."
            )

        if n_traits > 2:
            fdr_summary = []
            if fdr_survivors:
                fdr_summary.append(f"Correlations surviving FDR adjustment: {', '.join(fdr_survivors)}")
            if fdr_non_survivors:
                fdr_summary.append(f"Correlations not surviving FDR: {', '.join(fdr_non_survivors)}")
            if fdr_summary:
                sections.append("Multiple Testing Summary (FDR):\n" + "\n".join(fdr_summary))

        if discordant:
            disc_items = "; ".join(
                f"{t1} × {t2} (between-genotype r = {rg:.3f}, phenotypic r = {rp:.3f})"
                for t1, t2, rg, rp in discordant
            )
            sections.append(
                "Sign Discordance Alert:\n"
                f"The following pair(s) have opposite signs between phenotypic and "
                f"between-genotype modes: {disc_items}. "
                "Possible causes include environmental confounding, scale effects, or "
                "Simpson's paradox-type aggregation artefact. "
                "Do not act on either estimate without independent verification."
            )

    # ------------------------------------------------------------------
    # 7. Objective-specific closing guidance
    # ------------------------------------------------------------------
    objective_closing = {
        "Field understanding": (
            "For field-level understanding, phenotypic correlations describe the co-variation "
            "observed in this trial. Differences between phenotypic and between-genotype associations "
            "indicate how much of the observed co-variation reflects within-genotype environmental "
            "effects versus consistent differences between genotypes."
        ),
        "Genotype comparison": (
            "For genotype comparison, between-genotype associations are most informative. "
            "They reflect aggregate performance differences between genotypes, filtering out "
            "replication noise. Treat these as descriptive rankings — not as genetic parameters. "
            "Confidence in any ranking increases with more replications and a larger genotype panel."
        ),
        "Breeding decision": (
            "For breeding decisions, variance-component-based genotypic correlations are the most "
            "appropriate parameter, as they estimate genetic co-variation independently of "
            "environmental effects. However, correlation alone is insufficient for selection decisions. "
            "Reliable breeding decisions require additional evidence such as heritability, "
            "genetic gain estimates, and validation across target environments. "
            "Avoid implementing indirect selection strategies based solely on this analysis."
        ),
    }

    # Breeding guardrail — triggered by objective or by any high correlation
    high_corr_guardrail = False
    if comparison:
        for entry in comparison:
            rbg = entry["between_genotype_r"]
            rp  = entry["phenotypic_r"]
            rvc = entry["genotypic_vc_r"]
            if any(v is not None and abs(v) > 0.6 for v in [rbg, rp, rvc]):
                high_corr_guardrail = True
                break

    if user_objective == "Breeding decision" or high_corr_guardrail:
        sections.append(
            "Breeding Decision Safeguard:\n"
            "Correlation alone is insufficient for selection decisions. "
            "Reliable breeding decisions require additional evidence such as heritability, "
            "variance component analysis, and validation across target environments. "
            "High correlations (|r| > 0.6) may suggest potential for indirect selection, "
            "but this requires independent confirmation of the genetic basis."
        )

    if user_objective in objective_closing:
        sections.append(
            f"Guidance for '{user_objective}':\n" + objective_closing[user_objective]
        )

    return "\n\n".join(sections)
