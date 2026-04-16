"""
VivaSense – Cross-Module Intelligence Layer

Orchestrates outputs from ANOVA, Trait Association Intelligence, and Genetic Parameters
modules to produce integrated machine-readable decision objects and interpretation-ready payloads.

This layer consumes existing module outputs and produces deterministic, rule-first decisions
without generating prose content.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class IntegratedSummary(BaseModel):
    """Integrated summary of cross-module analysis."""
    selection_feasibility: str = Field(..., description="Selection feasibility status")
    adaptation_type: str = Field(..., description="Broad vs specific adaptation type")
    stability_requirement: str = Field(..., description="Stability analysis requirement")
    indirect_selection_support: str = Field(..., description="Indirect selection support level")
    experimental_reliability: str = Field(..., description="Experimental reliability assessment")
    overall_confidence: str = Field(..., description="Overall confidence in results")


class DecisionSignals(BaseModel):
    """Machine-readable decision signals for downstream processing."""
    advance_top_genotypes: bool = Field(..., description="Whether to advance top genotypes")
    recommend_stability_analysis: bool = Field(..., description="Whether to recommend stability analysis")
    recommend_indirect_selection: bool = Field(..., description="Whether to recommend indirect selection")
    recommend_environment_specific_selection: bool = Field(..., description="Whether to recommend environment-specific selection")


class CrossModuleSignals(BaseModel):
    """Complete cross-module intelligence output."""
    trait: str
    integrated_summary: IntegratedSummary
    integrated_risk_flags: List[str] = Field(default_factory=list)
    decision_signals: DecisionSignals
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)


def determine_selection_feasibility(
    anova_result: Dict[str, Any],
    genetic_result: Optional[Dict[str, Any]]
) -> str:
    """
    Determine selection feasibility based on ANOVA and genetic parameters.

    Core logic:
    - If genotype_significant is false -> "not_supported"
    - If genotype_significant is true and gxe_significant is true -> "feasible_with_caution"
    - If genotype_significant is true and heritability is high -> "feasible"
    """
    genotype_significant = anova_result.get("genotype_significant")
    gxe_significant = anova_result.get("gxe_significant")

    if genotype_significant is False:
        return "not_supported"

    if genotype_significant is True:
        # Prioritize GxE over heritability
        if gxe_significant is True:
            return "feasible_with_caution"

        # Check heritability if genetic result is available
        if genetic_result:
            heritability = genetic_result.get("heritability", {})
            h2_broad_sense = heritability.get("h2_broad_sense")
            if h2_broad_sense is not None and h2_broad_sense >= 0.5:  # High heritability threshold
                return "feasible"

        return "feasible"

    return "unknown"


def determine_adaptation_type(anova_result: Dict[str, Any]) -> str:
    """
    Determine adaptation type based on GxE significance.

    Core logic:
    - If gxe_significant is true -> "specific_adaptation_likely"
    - If gxe_significant is false -> "broad_adaptation_more_plausible"
    """
    gxe_significant = anova_result.get("gxe_significant")

    if gxe_significant is True:
        return "specific_adaptation_likely"
    elif gxe_significant is False:
        return "broad_adaptation_more_plausible"

    return "unknown"


def determine_stability_requirement(anova_result: Dict[str, Any]) -> str:
    """
    Determine stability analysis requirement based on GxE significance.

    Core logic:
    - If gxe_significant is true -> "required"
    - If gxe_significant is false -> "optional"
    """
    gxe_significant = anova_result.get("gxe_significant")

    if gxe_significant is True:
        return "required"
    elif gxe_significant is False:
        return "optional"

    return "unknown"


def determine_indirect_selection_support(
    trait_association_result: Optional[Dict[str, Any]],
    genetic_result: Optional[Dict[str, Any]],
    anova_result: Dict[str, Any]
) -> str:
    """
    Determine indirect selection support based on trait associations and genetics.

    Core logic:
    - If correlation is strong but pairwise_n_not_tracked exists -> "preliminary_only"
    - If strong correlation exists and genetic evidence supports -> "supported"
    - Otherwise -> "not_supported"
    """
    if not trait_association_result:
        return "not_supported"

    risk_flags = trait_association_result.get("risk_flags", [])
    significant_pairs = trait_association_result.get("significant_pairs", [])

    # Check for strong correlations
    has_strong_correlation = any(
        pair.get("strength") in ["strong", "very strong"]
        for pair in significant_pairs
    )

    if not has_strong_correlation:
        return "not_supported"

    # Check pairwise N tracking
    if "pairwise_n_not_tracked" in risk_flags:
        return "preliminary_only"

    # Check genetic evidence
    if genetic_result:
        heritability = genetic_result.get("heritability", {})
        h2_broad_sense = heritability.get("h2_broad_sense")
        if h2_broad_sense is not None and h2_broad_sense >= 0.3:  # Moderate heritability threshold
            return "supported"

    return "preliminary_only"


def determine_experimental_reliability(anova_result: Dict[str, Any]) -> str:
    """
    Determine experimental reliability based on precision level.

    Core logic:
    - If precision_level = "low" -> "limited"
    - If precision_level = "moderate" -> "acceptable"
    - If precision_level = "good" -> "high"
    """
    precision_level = anova_result.get("precision_level")

    if precision_level == "low":
        return "limited"
    elif precision_level == "moderate":
        return "acceptable"
    elif precision_level == "good":
        return "high"

    return "unknown"


def determine_overall_confidence(
    anova_result: Dict[str, Any],
    trait_association_result: Optional[Dict[str, Any]],
    genetic_result: Optional[Dict[str, Any]]
) -> str:
    """
    Determine overall confidence based on all module results.

    Core logic:
    - High: All modules successful, good precision, high heritability
    - Medium: ANOVA successful, other modules partial
    - Low: ANOVA failed or low precision
    """
    anova_status = anova_result.get("status")
    precision_level = anova_result.get("precision_level")

    if anova_status != "success":
        return "low"

    if precision_level == "low":
        return "low"

    # Check genetic parameters
    confidence_score = 1  # Base confidence

    if genetic_result and genetic_result.get("status") == "success":
        heritability = genetic_result.get("heritability", {})
        h2_broad_sense = heritability.get("h2_broad_sense")
        if h2_broad_sense is not None and h2_broad_sense >= 0.5:
            confidence_score += 1

    # Check trait association
    if trait_association_result and trait_association_result.get("significant_pairs"):
        confidence_score += 1

    if confidence_score >= 3:
        return "high"
    elif confidence_score >= 2:
        return "medium"

    return "low"


def collect_integrated_risk_flags(
    anova_result: Dict[str, Any],
    trait_association_result: Optional[Dict[str, Any]],
    genetic_result: Optional[Dict[str, Any]]
) -> List[str]:
    """
    Collect integrated risk flags from all modules.

    Combines risk flags from ANOVA, trait association, and genetic parameters
    with cross-module flags.
    """
    flags = []

    # ANOVA flags
    anova_warnings = anova_result.get("data_warnings", [])
    flags.extend(anova_warnings)
    if anova_result.get("gxe_significant") is True:
        flags.append("gxe_interaction_detected")
    if anova_result.get("precision_level") == "low":
        flags.append("low_experimental_precision")

    # Trait association flags
    if trait_association_result:
        ta_flags = trait_association_result.get("risk_flags", [])
        flags.extend(ta_flags)

    # Genetic parameters flags
    if genetic_result:
        gp_warnings = genetic_result.get("data_warnings", [])
        flags.extend(gp_warnings)

    # Cross-module flags
    if not trait_association_result:
        flags.append("missing_trait_association_data")
    if not genetic_result:
        flags.append("missing_genetic_parameters_data")

    return list(set(flags))  # Remove duplicates


def build_cross_module_signals(
    anova_result: Dict[str, Any],
    trait_association_result: Optional[Dict[str, Any]] = None,
    genetic_result: Optional[Dict[str, Any]] = None
) -> CrossModuleSignals:
    """
    Master function to build integrated cross-module intelligence signals.

    Consumes existing module outputs and produces deterministic decisions.
    """
    trait = anova_result.get("trait", "unknown")

    # Build integrated summary
    integrated_summary = IntegratedSummary(
        selection_feasibility=determine_selection_feasibility(anova_result, genetic_result),
        adaptation_type=determine_adaptation_type(anova_result),
        stability_requirement=determine_stability_requirement(anova_result),
        indirect_selection_support=determine_indirect_selection_support(
            trait_association_result, genetic_result, anova_result
        ),
        experimental_reliability=determine_experimental_reliability(anova_result),
        overall_confidence=determine_overall_confidence(
            anova_result, trait_association_result, genetic_result
        )
    )

    # Build decision signals
    decision_signals = DecisionSignals(
        advance_top_genotypes=integrated_summary.selection_feasibility in ["feasible", "feasible_with_caution"],
        recommend_stability_analysis=integrated_summary.stability_requirement == "required",
        recommend_indirect_selection=(
            integrated_summary.indirect_selection_support == "supported" and
            trait_association_result is not None and
            genetic_result is not None
        ),
        recommend_environment_specific_selection=anova_result.get("gxe_significant") is True
    )

    # Collect integrated risk flags
    integrated_risk_flags = collect_integrated_risk_flags(
        anova_result, trait_association_result, genetic_result
    )

    # Build supporting evidence
    supporting_evidence = {
        "anova": anova_result,
        "trait_association": trait_association_result or {},
        "genetics": genetic_result or {}
    }

    return CrossModuleSignals(
        trait=trait,
        integrated_summary=integrated_summary,
        integrated_risk_flags=integrated_risk_flags,
        decision_signals=decision_signals,
        supporting_evidence=supporting_evidence
    )


def generate_integrated_interpretation(signals: CrossModuleSignals) -> str:
    """
    Generate a unified, academically rigorous interpretation from cross-module signals.

    This function transforms machine-readable decision signals into a coherent,
    scientifically defensible interpretation that guides research and breeding decisions.
    """
    sections = []

    # 1. Integrated Overview
    overview_parts = []

    # Selection feasibility
    if signals.integrated_summary.selection_feasibility == "not_supported":
        overview_parts.append("the absence of significant genetic variation indicates that selection for this trait is not currently supported")
    elif signals.integrated_summary.selection_feasibility == "feasible":
        overview_parts.append("significant genetic variation suggests that selection for this trait is feasible")
    elif signals.integrated_summary.selection_feasibility == "feasible_with_caution":
        overview_parts.append("significant genetic variation exists, but selection should proceed with caution due to environmental interactions")

    # Adaptation type
    if signals.integrated_summary.adaptation_type == "broad_adaptation_more_plausible":
        overview_parts.append("with evidence suggesting relatively broad adaptation across environments")
    elif signals.integrated_summary.adaptation_type == "specific_adaptation_likely":
        overview_parts.append("with indications of environment-specific adaptation patterns")

    # Experimental reliability
    if signals.integrated_summary.experimental_reliability == "high":
        overview_parts.append("supported by reliable experimental conditions")
    elif signals.integrated_summary.experimental_reliability == "acceptable":
        overview_parts.append("with acceptable experimental precision")
    elif signals.integrated_summary.experimental_reliability == "limited":
        overview_parts.append("though experimental reliability is limited")

    # Overall confidence
    if signals.integrated_summary.overall_confidence == "high":
        overview_parts.append("providing strong evidence for informed decision-making")
    elif signals.integrated_summary.overall_confidence == "medium":
        overview_parts.append("offering moderate confidence in the results")
    elif signals.integrated_summary.overall_confidence == "low":
        overview_parts.append("suggesting cautious interpretation of the findings")

    sections.append("Integrated Overview\n" + "This analysis indicates " + ", ".join(overview_parts) + ".")

    # 2. Selection Implication
    selection_parts = []

    if signals.integrated_summary.selection_feasibility == "not_supported":
        selection_parts.append("The lack of significant genetic variation for this trait suggests that direct selection efforts may not be effective with the current germplasm.")
    else:
        selection_parts.append("The presence of significant genetic variation provides a foundation for selection-based improvement of this trait.")

        # Add heritability context if available
        genetics = signals.supporting_evidence.get("genetics", {})
        if genetics and genetics.get("heritability"):
            h2 = genetics["heritability"].get("h2_broad_sense")
            if h2 is not None:
                if h2 >= 0.5:
                    selection_parts.append("The high heritability suggests that genetic gains through selection are likely to be substantial and reliable.")
                elif h2 >= 0.3:
                    selection_parts.append("The moderate heritability indicates that selection can be effective, though environmental factors will influence expression.")
                else:
                    selection_parts.append("The relatively low heritability suggests that selection progress may be slower and more variable.")

        if signals.integrated_summary.selection_feasibility == "feasible_with_caution":
            selection_parts.append("However, the presence of genotype × environment interaction requires that selection decisions be validated across target environments.")

    sections.append("Selection Implication\n" + " ".join(selection_parts))

    # 3. Adaptation and Stability
    adaptation_parts = []

    if signals.integrated_summary.adaptation_type == "broad_adaptation_more_plausible":
        adaptation_parts.append("The results suggest that genotypes exhibit relatively consistent performance across the tested environments, indicating potential for broad adaptation.")
    elif signals.integrated_summary.adaptation_type == "specific_adaptation_likely":
        adaptation_parts.append("The evidence points toward environment-specific adaptation patterns, where genotype performance varies significantly across different environmental conditions.")

    if signals.integrated_summary.stability_requirement == "required":
        adaptation_parts.append("The significant genotype × environment interaction necessitates stability analysis to identify genotypes with consistent performance across environments.")
        adaptation_parts.append("Ranking based on overall means should be interpreted cautiously, as no single genotype demonstrates universal superiority.")
    elif signals.integrated_summary.stability_requirement == "optional":
        adaptation_parts.append("The absence of significant genotype × environment interaction suggests that genotype rankings are relatively stable across the tested conditions.")

    sections.append("Adaptation and Stability\n" + " ".join(adaptation_parts))

    # 4. Trait Relationship Implication
    relationship_parts = []

    trait_assoc = signals.supporting_evidence.get("trait_association", {})
    if trait_assoc and trait_assoc.get("significant_pairs"):
        relationship_parts.append("Significant trait associations were detected, suggesting potential relationships between this trait and others in the study.")
    else:
        relationship_parts.append("No significant trait associations were identified for this trait.")

    if signals.integrated_summary.indirect_selection_support == "supported":
        relationship_parts.append("The strength of these associations, combined with supporting genetic parameters, suggests that indirect selection strategies may be viable for improving this trait through correlated responses.")
    elif signals.integrated_summary.indirect_selection_support == "preliminary_only":
        relationship_parts.append("While some trait associations exist, the current evidence is preliminary and should not yet be used as the sole basis for indirect selection decisions.")
    elif signals.integrated_summary.indirect_selection_support == "not_supported":
        relationship_parts.append("The available evidence does not support indirect selection approaches for this trait at this time.")

    if "pairwise_n_not_tracked" in signals.integrated_risk_flags:
        relationship_parts.append("However, the correlation analyses are limited by the lack of pairwise sample size tracking, which constrains the reliability of association-based conclusions.")

    sections.append("Trait Relationship Implication\n" + " ".join(relationship_parts))

    # 5. Experimental Reliability
    reliability_parts = []

    if signals.integrated_summary.experimental_reliability == "high":
        reliability_parts.append("The experimental design and execution demonstrate high reliability, with good precision and control over environmental variation.")
        reliability_parts.append("This provides confidence that the observed differences reflect true genetic effects rather than experimental artifacts.")
    elif signals.integrated_summary.experimental_reliability == "acceptable":
        reliability_parts.append("The experimental precision is acceptable, though not optimal, suggesting that the results are generally trustworthy but should be validated in subsequent experiments.")
    elif signals.integrated_summary.experimental_reliability == "limited":
        reliability_parts.append("The experimental reliability is limited by high variability and potentially inadequate precision.")
        reliability_parts.append("This introduces uncertainty into the results and suggests that experimental conditions may have substantially influenced the observed outcomes.")

    sections.append("Experimental Reliability\n" + " ".join(reliability_parts))

    # 6. Risk and Caution
    risk_parts = []

    risks_mentioned = []

    if "gxe_interaction_detected" in signals.integrated_risk_flags:
        risk_parts.append("The significant genotype × environment interaction represents a major consideration, as it complicates genotype evaluation and may limit the transferability of results across environments.")
        risks_mentioned.append("GxE")

    if "low_experimental_precision" in signals.integrated_risk_flags:
        risk_parts.append("The low experimental precision introduces variability that could mask true genetic differences or exaggerate environmental effects.")
        risks_mentioned.append("precision")

    if "pairwise_n_not_tracked" in signals.integrated_risk_flags:
        risk_parts.append("The absence of pairwise sample size tracking limits the confidence that can be placed in correlation-based inferences and trait relationship conclusions.")
        risks_mentioned.append("correlation tracking")

    if "small_sample_size" in signals.integrated_risk_flags:
        risk_parts.append("The relatively small sample sizes may limit the generalizability of these findings and increase the risk of spurious results.")
        risks_mentioned.append("sample size")

    if "missing_trait_association_data" in signals.integrated_risk_flags:
        risk_parts.append("The absence of trait association data prevents comprehensive evaluation of indirect selection opportunities.")
        risks_mentioned.append("missing association data")

    if "missing_genetic_parameters_data" in signals.integrated_risk_flags:
        risk_parts.append("The lack of genetic parameters data constrains the ability to assess heritability and predict selection response.")
        risks_mentioned.append("missing genetic data")

    if not risk_parts:
        risk_parts.append("No major experimental or analytical limitations were identified in this integrated analysis.")

    sections.append("Risk and Caution\n" + " ".join(risk_parts))

    # 7. Final Recommendation
    recommendation_parts = []

    if signals.decision_signals.advance_top_genotypes:
        if signals.integrated_summary.selection_feasibility == "feasible_with_caution":
            recommendation_parts.append("The top-performing genotypes should be advanced for further evaluation, but with validation across multiple environments to confirm their stability.")
        else:
            recommendation_parts.append("The top-performing genotypes should be advanced for further evaluation and potential use in breeding programs.")
    else:
        recommendation_parts.append("Genotype advancement is not recommended at this time due to insufficient evidence of genetic variation.")

    if signals.decision_signals.recommend_stability_analysis:
        recommendation_parts.append("Stability analysis, such as AMMI or GGE biplot analysis, should be conducted to identify genotypes with consistent performance across environments.")

    if signals.decision_signals.recommend_indirect_selection:
        recommendation_parts.append("Indirect selection strategies targeting correlated traits may be considered, provided that the genetic basis of these relationships is confirmed.")
    elif signals.integrated_summary.indirect_selection_support == "preliminary_only":
        recommendation_parts.append("While preliminary trait associations exist, indirect selection should not be implemented without further validation of the genetic relationships.")

    if signals.decision_signals.recommend_environment_specific_selection:
        recommendation_parts.append("Selection strategies should account for environment-specific adaptation, with separate evaluation and advancement decisions for different target environments.")

    if signals.integrated_summary.overall_confidence in ["low", "medium"]:
        recommendation_parts.append("Given the current confidence level, these results should be validated through additional experimentation before making major breeding decisions.")

    sections.append("Final Recommendation\n" + " ".join(recommendation_parts))

    return "\n\n".join(sections)