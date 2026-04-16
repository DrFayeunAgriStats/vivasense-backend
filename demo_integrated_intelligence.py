#!/usr/bin/env python3
"""
VivaSense Integrated Intelligence Engine - Demonstration

This script demonstrates how the Cross-Module Intelligence Layer
transforms individual module outputs into unified, academic-grade interpretations.
"""

import sys
sys.path.append('genetics-module')

from cross_module_intelligence import build_cross_module_signals, generate_integrated_interpretation

def demo_scenario_1():
    """Demo: High-confidence scenario with feasible selection and GxE"""
    print("=" * 80)
    print("DEMO SCENARIO 1: High-confidence feasible selection with GxE interaction")
    print("=" * 80)

    # Simulate module outputs
    anova_result = {
        "trait": "grain_yield",
        "status": "success",
        "genotype_significant": True,
        "gxe_significant": True,
        "precision_level": "good",
        "data_warnings": []
    }

    trait_association_result = {
        "significant_pairs": [
            {"trait_1": "grain_yield", "trait_2": "plant_height", "strength": "strong", "r": 0.75}
        ],
        "risk_flags": ["pairwise_n_not_tracked"]
    }

    genetic_result = {
        "status": "success",
        "heritability": {"h2_broad_sense": 0.65},
        "data_warnings": []
    }

    # Build cross-module signals
    signals = build_cross_module_signals(anova_result, trait_association_result, genetic_result)

    # Generate integrated interpretation
    interpretation = generate_integrated_interpretation(signals)

    print(interpretation)
    print("\n" + "=" * 80 + "\n")

def demo_scenario_2():
    """Demo: Low-confidence scenario with limited selection support"""
    print("=" * 80)
    print("DEMO SCENARIO 2: Low-confidence with limited selection support")
    print("=" * 80)

    # Simulate module outputs
    anova_result = {
        "trait": "yield",
        "status": "success",
        "genotype_significant": False,
        "gxe_significant": False,
        "precision_level": "low",
        "data_warnings": ["unbalanced_design"]
    }

    # No trait association or genetic data
    trait_association_result = None
    genetic_result = None

    # Build cross-module signals
    signals = build_cross_module_signals(anova_result, trait_association_result, genetic_result)

    # Generate integrated interpretation
    interpretation = generate_integrated_interpretation(signals)

    print(interpretation)
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    demo_scenario_1()
    demo_scenario_2()