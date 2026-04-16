"""
Trait Association Interpretation Engine (No circular dependencies)

Generates academic-grade trait association interpretation.
No imports from routes modules to avoid circular dependencies.
"""

from typing import Any, Dict, List, Optional


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
    - If n_observations < 10 -> include "preliminary evidence"
    - If "pairwise_n_not_tracked" in risk_flags -> explicitly state confidence is limited
    - If gxe_significant -> explicitly warn that relationships may vary across environments
    - Do not recommend indirect selection based on correlations alone
    """
    sections = []
    
    # 1. Overview
    overview = []
    overview.append(f"This analysis examined trait associations among {n_traits} traits")
    if environment_context == "multi_environment":
        overview.append(f"across multiple environments")
    overview.append(f"using {n_observations} genotype mean(s).")
    
    if n_observations < 10:
        overview.append("The sample size is limited, indicating preliminary evidence.")
    
    sections.append(" ".join(overview))
    
    # 2. Significant Associations
    if n_significant_pairs > 0:
        assoc = []
        assoc.append(f"Significant trait associations (p < 0.05) were detected for {n_significant_pairs} pair(s).")
        
        if strongest_positive:
            assoc.append(
                f"The strongest positive association was between "
                f"{strongest_positive.get('trait_1')} and {strongest_positive.get('trait_2')} "
                f"(r = {strongest_positive.get('r', 0):.2f})."
            )
        
        if strongest_negative:
            assoc.append(
                f"The strongest negative association was between "
                f"{strongest_negative.get('trait_1')} and {strongest_negative.get('trait_2')} "
                f"(r = {strongest_negative.get('r', 0):.2f})."
            )
        
        assoc.append(
            "Positive correlations suggest traits improving together, while negative correlations "
            "may reflect trade-offs in genetic or physiological control."
        )
        
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
