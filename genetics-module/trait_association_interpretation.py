"""
Trait Association Interpretation Engine (No circular dependencies)

Generates academic-grade trait association interpretation.
No imports from routes modules to avoid circular dependencies.
"""

from typing import Any, Dict, List, Optional


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
