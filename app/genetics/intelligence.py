"""
genetics/intelligence.py
=========================
Genetics-specific analogues of V2.2's intelligence-block builders:

  build_executive_insight()  →  build_genetics_executive_insight()
  build_reviewer_radar()     →  build_genetics_reviewer_radar()
  build_decision_rules()     →  build_genetics_decision_rules()

All functions return plain-Python objects (no NumPy types).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .compat import fmt_p_display


# ── Executive insight ─────────────────────────────────────────────────────────

def build_genetics_executive_insight(
    p_map: Dict[str, Optional[float]],
    trait: str,
    alpha: float = 0.05,
    h2: Optional[float] = None,
    ga_percent: Optional[float] = None,
    n_locations: int = 1,
    n_genotypes: int = 0,
    cv: Optional[float] = None,
    best_genotype: Optional[str] = None,
) -> str:
    """
    One-paragraph narrative for an executive audience.
    Parallels V2.2 build_executive_insight() with genetics extensions.
    """
    parts: List[str] = []

    # Scope statement
    loc_str = f"{n_locations} location{'s' if n_locations != 1 else ''}"
    parts.append(
        f"Analysis of {trait} across {loc_str} involving "
        f"{n_genotypes} genotype{'s' if n_genotypes != 1 else ''}."
    )

    # Main effects / G×L
    g_p  = p_map.get("Genotype")
    l_p  = p_map.get("Location")
    gl_p = p_map.get("G\u00d7L") or p_map.get("GxL")

    if g_p is not None:
        if g_p <= alpha:
            parts.append(
                f"Genotype differences are significant (p\u202f=\u202f{fmt_p_display(g_p)}), "
                f"confirming substantial genetic variation for {trait}."
            )
        else:
            parts.append(
                f"Genotype effect is not significant (p\u202f=\u202f{fmt_p_display(g_p)}), "
                "suggesting limited genetic variation in this trial."
            )

    if l_p is not None and n_locations > 1:
        if l_p <= alpha:
            parts.append(
                f"Location (environment) effect is significant (p\u202f=\u202f{fmt_p_display(l_p)}), "
                f"indicating that growing conditions substantially influence {trait}."
            )

    if gl_p is not None and n_locations > 1:
        if gl_p <= alpha:
            parts.append(
                f"The G\u00d7L interaction is significant (p\u202f=\u202f{fmt_p_display(gl_p)}), "
                "meaning genotype rankings are inconsistent across locations — "
                "site-specific variety recommendations are warranted."
            )
        else:
            parts.append(
                "The non-significant G\u00d7L interaction supports broadly adapted "
                "variety recommendations across all tested locations."
            )

    # Heritability
    if h2 is not None:
        cat = "high" if h2 >= 0.6 else ("moderate" if h2 >= 0.3 else "low")
        parts.append(
            f"Broad-sense heritability is {cat} (H\u00b2\u202f=\u202f{h2:.2%}), "
            + (
                "suggesting effective direct selection in early generations."
                if h2 >= 0.6
                else "suggesting multi-environment evaluation is needed before final selections."
            )
        )

    # Genetic advance
    if ga_percent is not None:
        parts.append(
            f"Predicted genetic advance under 5\u202f% selection intensity is "
            f"{ga_percent:.1f}\u202f% of the population mean."
        )

    # Experimental precision
    if cv is not None:
        quality = (
            "excellent" if cv < 10
            else ("acceptable" if cv < 20 else "high (consider reviewing field layout)")
        )
        parts.append(f"Experimental CV is {quality} ({cv:.1f}\u202f%).")

    # Top performer
    if best_genotype:
        parts.append(f"Top-performing genotype: {best_genotype}.")

    return "  ".join(parts)


# ── Reviewer radar ────────────────────────────────────────────────────────────

def build_genetics_reviewer_radar(
    shapiro_rec: Dict[str, Any],
    levene_rec: Dict[str, Any],
    p_map: Dict[str, Optional[float]],
    cv: Optional[float],
    n_locations: int = 1,
    n_reps: int = 1,
    h2: Optional[float] = None,
    alpha: float = 0.05,
) -> List[str]:
    """
    Anticipated peer-reviewer questions / red flags.
    Parallels V2.2 build_reviewer_radar() with plant-breeding extensions.
    """
    radar: List[str] = []

    # Normality
    if shapiro_rec.get("passed") is False:
        radar.append(
            "Normality assumption violated \u2014 reviewers may question parametric ANOVA "
            "validity.  Report transformation rationale or Kruskal-Wallis results as supplementary."
        )

    # Homogeneity
    if levene_rec.get("passed") is False:
        radar.append(
            "Heterogeneous variances detected \u2014 consider reporting Welch F or stating "
            "robustness of ANOVA to mild heteroscedasticity."
        )

    # CV
    if cv is not None and cv > 20:
        radar.append(
            f"High coefficient of variation (CV\u202f=\u202f{cv:.1f}\u202f%) may indicate field "
            "variability or data quality issues.  Reviewers will expect an explanation of "
            "experimental control measures."
        )

    # Multi-location coverage
    if n_locations < 3:
        radar.append(
            f"Only {n_locations} location(s) tested \u2014 reviewers may question generalisability. "
            "Multi-environment testing across \u22653 locations is recommended for variety release."
        )

    # Replication
    if n_reps < 3:
        radar.append(
            f"Only {n_reps} replication(s) per location \u2014 statistical power may be limited. "
            "At least 3 reps are the standard minimum for RCBD trials."
        )

    # G×L
    gl_p = p_map.get("G\u00d7L") or p_map.get("GxL")
    if gl_p is not None and gl_p <= alpha:
        radar.append(
            "Significant G\u00d7L interaction \u2014 reviewers will expect AMMI or GGE biplot "
            "interpretation and explicit stability analysis "
            "(e.g. Eberhart-Russell regression coefficients and ASV ranks)."
        )

    # Heritability
    if h2 is not None and h2 < 0.3:
        radar.append(
            f"Low heritability (H\u00b2\u202f=\u202f{h2:.2%}) \u2014 reviewers may question the utility "
            "of direct phenotypic selection.  Discuss environmental confounding factors."
        )

    if not radar:
        radar.append(
            "No major statistical concerns identified.  Ensure the methods section clearly "
            "documents trial design, number of locations, seasons, and replication structure."
        )

    return radar


# ── Decision rules ────────────────────────────────────────────────────────────

def build_genetics_decision_rules(
    means_table: List[Dict[str, Any]],
    genotype_col: str,
    trait: str,
    alpha: float = 0.05,
    h2: Optional[float] = None,
    stable_genotypes: Optional[List[str]] = None,
) -> List[str]:
    """
    Actionable breeding-programme recommendations.
    Parallels V2.2 build_decision_rules() with plant-breeding language.
    """
    rules: List[str] = []

    if not means_table:
        return ["Insufficient data to generate decision rules."]

    # Rank by mean descending
    try:
        ranked = sorted(
            means_table,
            key=lambda r: float(r.get("mean", r.get("Mean", 0))),
            reverse=True,
        )
    except (TypeError, ValueError):
        ranked = means_table

    top    = ranked[0]    if ranked    else None
    bottom = ranked[-1]   if len(ranked) > 1 else None

    if top:
        top_id   = top.get(genotype_col) or top.get("genotype") or top.get("Genotype") or "\u2014"
        top_mean = top.get("mean") or top.get("Mean")
        if top_mean is not None:
            rules.append(
                f"Priority selection: {top_id} ranks highest for {trait} "
                f"(mean\u202f=\u202f{top_mean:.3g}).  "
                "Advance to replicated multi-location trials or F\u2083 bulk selection."
            )

    if bottom:
        bot_id   = bottom.get(genotype_col) or bottom.get("genotype") or bottom.get("Genotype") or "\u2014"
        bot_mean = bottom.get("mean") or bottom.get("Mean")
        if bot_mean is not None:
            rules.append(
                f"Consider discarding {bot_id} (mean\u202f=\u202f{bot_mean:.3g}), "
                "the lowest-performing genotype, to reduce resource expenditure."
            )

    # Heritability-based guidance
    if h2 is not None:
        if h2 >= 0.6:
            rules.append(
                f"High heritability (H\u00b2\u202f=\u202f{h2:.2%}) justifies early-generation "
                "selection (F\u2082 or F\u2083 bulk).  Direct phenotypic selection is effective."
            )
        elif h2 >= 0.3:
            rules.append(
                f"Moderate heritability (H\u00b2\u202f=\u202f{h2:.2%}) \u2014 progeny testing across "
                "multiple environments is recommended before variety release."
            )
        else:
            rules.append(
                f"Low heritability (H\u00b2\u202f=\u202f{h2:.2%}) \u2014 recurrent selection or "
                "marker-assisted selection (MAS) may improve selection efficiency."
            )

    # Stable genotypes
    if stable_genotypes:
        stab_str = ", ".join(str(g) for g in stable_genotypes[:3])
        rules.append(
            f"Stable, broadly-adapted genotype(s): {stab_str}.  "
            "These are preferred candidates for wide-adaptation release."
        )

    return rules


# ── Strict template for genetics ──────────────────────────────────────────────

def attach_genetics_template(
    result: Dict[str, Any],
    trait_name: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Populate the `strict_template` key of a genetics result dict, mirroring
    V2.2's attach_strict_template().  The template provides a structured
    interpretation report with fixed section headings.
    """
    meta     = result.get("meta", {})
    tables   = result.get("tables", {})
    intel    = result.get("intelligence", {})

    anova_recs = tables.get("combined_anova", [])
    vc_recs    = tables.get("variance_components", [])
    stab_recs  = tables.get("stability", [])

    # Build fixed-section report
    sections: List[Dict[str, Any]] = [
        {
            "section": "Overview",
            "content": intel.get("executive_insight", ""),
        },
        {
            "section": "ANOVA Summary",
            "content": (
                f"Combined ANOVA for {trait_name}: "
                + _anova_narrative(anova_recs, alpha)
            ),
        },
        {
            "section": "Variance Components",
            "content": _vc_narrative(vc_recs),
        },
        {
            "section": "Stability",
            "content": _stab_narrative(stab_recs),
        },
        {
            "section": "Statistical Assumptions",
            "content": intel.get("assumptions_verdict", ""),
        },
        {
            "section": "Recommendations",
            "content": "  \u2022  ".join(intel.get("decision_rules", [])),
        },
        {
            "section": "Anticipated Reviewer Concerns",
            "content": "  \u2022  ".join(intel.get("reviewer_radar", [])),
        },
    ]

    result["strict_template"] = {"trait": trait_name, "sections": sections}
    return result


# ── Internal narrative helpers ────────────────────────────────────────────────

def _anova_narrative(records: List[Dict[str, Any]], alpha: float) -> str:
    if not records:
        return "No ANOVA table available."
    lines = []
    for r in records:
        src  = r.get("source", "?")
        p    = r.get("PR(>F)")
        disp = r.get("PR(>F)_display", "")
        if p is None:
            continue
        sig = "significant" if float(p) <= alpha else "not significant"
        lines.append(f"{src} is {sig} (p\u202f=\u202f{disp})")
    return "; ".join(lines) + "." if lines else "ANOVA could not be summarised."


def _vc_narrative(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "Variance components were not estimated."
    row = records[0] if records else {}
    h2  = row.get("H2_broad") or row.get("heritability")
    ga  = row.get("GA_percent") or row.get("ga_percent")
    parts = []
    if h2 is not None:
        parts.append(f"H\u00b2 (broad-sense) = {float(h2):.2%}")
    if ga is not None:
        parts.append(f"Genetic advance = {float(ga):.1f}\u202f% of mean")
    return "; ".join(parts) + "." if parts else "Variance components estimated — see tables."


def _stab_narrative(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "Stability analysis was not performed (requires \u22652 locations)."
    stable = [
        str(r.get("Genotype") or r.get("genotype") or "?")
        for r in records
        if str(r.get("classification", "")).startswith("stable")
    ]
    if stable:
        return f"Stable genotype(s): {', '.join(stable[:4])}."
    return "No genotypes classified as broadly stable in this trial set."
