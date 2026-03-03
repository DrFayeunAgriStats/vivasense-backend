"""
Stability and adaptation analysis.

Methods
-------
Eberhart & Russell (1966):
  Environmental index  Ij = ȳ_.j − ȳ..
  Regression           ȳij = ȳi. + bi·Ij + δij
  bi = Σj(ȳij·Ij) / Σj(Ij²)          ← regression coefficient
  S²di = [SSdev_i / (l-2)] − (MSE/r)  ← deviation mean square

Classification:
  Stable high-yielding   → mean > grand_mean, bi ≈ 1, S²di ns
  Above-average resp.    → mean > grand_mean, bi > 1
  Below-average stable   → mean < grand_mean, bi < 1, S²di ns
  Widely adapted         → mean ≈ grand_mean, bi ≈ 1, S²di ns
  Unstable               → S²di significant (p < α)

AMMI Stability Value (Zobel et al. 1988):
  ASV_i = √[(SS_IPCA1/SS_IPCA2 × score_i_IPCA1)² + (score_i_IPCA2)²]
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StabilityEngine:
    """Eberhart-Russell regression stability + AMMI stability value."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # EBERHART-RUSSELL
    # ------------------------------------------------------------------

    def eberhart_russell(
        self,
        anova_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute bi, S²di, and stability classification for each genotype.

        Requires the combined ANOVA result dict from MultilocationEngine.
        """
        alpha = self.config.alpha
        grand_mean = anova_result["grand_mean"]
        geno_names = anova_result["genotype_names"]
        loc_names = anova_result["location_names"]
        cell_matrix = anova_result["cell_matrix"]   # pd.DataFrame g×l
        MS_e = anova_result["MS"]["error"]
        r = anova_result["n_reps"]
        l = len(loc_names)
        df_S2di = l - 2

        # Environmental index: Ij = ȳ_.j − ȳ..
        loc_means = cell_matrix.mean(axis=0)
        env_index = (loc_means - grand_mean).to_dict()  # {loc: Ij}
        Ij = np.array([env_index[loc] for loc in loc_names])
        sum_Ij2 = float((Ij ** 2).sum())

        results = []
        for geno in geno_names:
            Yi_j = np.array([float(cell_matrix.loc[geno, loc]) for loc in loc_names])
            Yi_mean = float(Yi_j.mean())

            # Regression slope bi
            bi = float((Yi_j * Ij).sum() / sum_Ij2) if sum_Ij2 > 0 else 1.0
            ai = Yi_mean  # intercept (Ij sums to 0)

            # Deviation sum of squares
            predicted = ai + bi * Ij
            SS_dev = float(((Yi_j - predicted) ** 2).sum())

            # S²di (Eberhart & Russell 1966, eq. 6)
            S2di_raw = (SS_dev / df_S2di) - (MS_e / r) if df_S2di > 0 else 0.0
            S2di = S2di_raw   # may be negative (report as-is, cf. Eberhart & Russell)

            # F-test for S²di: MS_dev vs MSE
            MS_dev = SS_dev / df_S2di if df_S2di > 0 else 0.0
            F_s2di = MS_dev / MS_e if MS_e > 0 else 0.0
            df_error = anova_result["df"]["error"]
            p_s2di = float(1 - stats.f.cdf(F_s2di, df_S2di, df_error)) if F_s2di > 0 else 1.0
            s2di_sig = p_s2di < alpha

            # Standard error of bi
            se_bi = float(np.sqrt(MS_e / (r * sum_Ij2))) if (r > 0 and sum_Ij2 > 0) else 0.0

            classification, recommendation = _classify(
                Yi_mean, grand_mean, bi, s2di_sig, alpha
            )

            results.append({
                "genotype": geno,
                "grand_mean": round(Yi_mean, 4),
                "bi": {
                    "value": round(bi, 4),
                    "se": round(se_bi, 4),
                    "formula": "bi = Σ(ȳij × Ij) / Σ(Ij²)",
                    "formula_description": "Regression coefficient of genotype performance on environmental index",
                    "interpretation": _interp_bi(bi),
                },
                "S2di": {
                    "value": round(S2di, 6),
                    "MS_dev": round(MS_dev, 6),
                    "F": round(F_s2di, 3),
                    "p_value": round(p_s2di, 4),
                    "significant": s2di_sig,
                    "formula": "S²di = [Σ(ȳij − ȳi. − bi·Ij)² / (l−2)] − (MSE/r)",
                    "formula_description": "Deviation mean square — lower indicates more stable (predictable) genotype",
                    "interpretation": "Significant — unstable (unpredictable)" if s2di_sig else "Non-significant — stable",
                },
                "classification": classification,
                "recommendation": recommendation,
            })

        # Sort by descending mean yield
        results.sort(key=lambda x: x["grand_mean"], reverse=True)

        environmental_indices = {
            loc: {
                "index": round(float(env_index[loc]), 4),
                "formula": "Ij = ȳ_.j − ȳ..",
                "formula_description": "Deviation of location mean from grand mean",
                "location_mean": round(float(loc_means[loc]), 4),
                "environment_type": "Favourable" if env_index[loc] > 0 else "Unfavourable",
            }
            for loc in loc_names
        }

        return {
            "method": "Eberhart & Russell (1966) Regression Stability Analysis",
            "grand_mean": round(grand_mean, 4),
            "n_environments": l,
            "environmental_indices": environmental_indices,
            "genotype_stability": results,
            "stability_classification_rules": {
                "stable_high_yielding":     "mean > grand_mean AND bi ≈ 1 AND S²di not significant",
                "above_average_responsive": "mean > grand_mean AND bi > 1",
                "below_average_stable":     "mean < grand_mean AND bi < 1 AND S²di not significant",
                "widely_adapted":           "bi ≈ 1 AND S²di not significant (regardless of mean)",
                "unstable":                 "S²di significant (p < α)",
            },
        }

    # ------------------------------------------------------------------
    # AMMI STABILITY VALUE
    # ------------------------------------------------------------------

    def compute_asv(
        self,
        er_result: Dict[str, Any],
        ammi_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Attach AMMI Stability Value (ASV) to each genotype in er_result.

        ASV_i = √[(SS_IPCA1/SS_IPCA2 × score_i_IPCA1)² + score_i_IPCA2²]
        """
        ev = ammi_result.get("explained_variance", [])
        if len(ev) < 2:
            return er_result  # not enough axes for ASV

        ss1 = ev[0].get("ss", 1.0)
        ss2 = ev[1].get("ss", 1.0)
        ratio = ss1 / ss2 if ss2 > 0 else 1.0

        ipca_geno = ammi_result.get("ipca_scores", {}).get("genotypes", {})

        asv_values = {}
        for entry in er_result["genotype_stability"]:
            geno = entry["genotype"]
            scores = ipca_geno.get(geno, {})
            ipca1 = scores.get("IPCA1", 0.0)
            ipca2 = scores.get("IPCA2", 0.0)
            asv = float(np.sqrt((ratio * ipca1) ** 2 + ipca2 ** 2))
            asv_values[geno] = asv

        # Rank by ASV (lower = more stable)
        ranked = sorted(asv_values, key=asv_values.get)
        asv_ranks = {g: rank + 1 for rank, g in enumerate(ranked)}

        for entry in er_result["genotype_stability"]:
            geno = entry["genotype"]
            entry["ASV"] = {
                "value": round(asv_values.get(geno, 0.0), 4),
                "rank": asv_ranks.get(geno),
                "formula": "ASV = √[(SS_IPCA1/SS_IPCA2 × IPCA1_score)² + IPCA2_score²]",
                "formula_description": "AMMI Stability Value — lower = more stable across environments",
                "interpretation": "Stable" if asv_ranks.get(geno, 99) <= len(ranked) // 3 else "Moderately stable" if asv_ranks.get(geno, 99) <= 2 * len(ranked) // 3 else "Unstable",
            }

        return er_result


# ── Classification helpers ──────────────────────────────────────────────────

def _classify(mean: float, grand: float, bi: float, s2_sig: bool, alpha: float):
    bi_tol = 0.25  # ±0.25 around 1.0 = "approximately 1"
    above = mean > grand
    bi_avg = abs(bi - 1.0) <= bi_tol

    if s2_sig:
        return "unstable", "Not recommended for release — unpredictable performance across environments"
    if above and bi_avg:
        return "stable_high_yielding", "Excellent candidate for broad release — high yield AND stable"
    if above and bi > 1:
        return "above_average_responsive", "Best in high-input/favourable environments (Kano, irrigated)"
    if above and bi < 1:
        return "high_yield_low_responsive", "High yield in poor environments — suited for subsistence farming"
    if not above and bi_avg:
        return "widely_adapted_average", "Broadly adapted but below average yield — useful for marginal environments"
    if not above and bi < 1:
        return "below_average_stable", "Low yield, stable — suited for very harsh environments only"
    return "below_average_responsive", "Below average, responsive — avoid in target environments"


def _interp_bi(bi: float) -> str:
    if bi > 1.25:
        return "Above-average response — thrives in high-input / favourable environments"
    if bi < 0.75:
        return "Below-average response — suited to low-input / marginal environments"
    return "Average stability — broadly adapted"
