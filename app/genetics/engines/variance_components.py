"""
Variance component estimation from combined ANOVA mean squares.

Formulas (balanced RCBD, mixed model: G fixed, L random)
---------------------------------------------------------
σ²e  = MSE
σ²gl = max(0,  (MSGL − MSE) / r)
σ²g  = max(0,  (MSG  − MSGL) / (r × l))
σ²p  = σ²g + σ²gl/l + σ²e/(r×l)          ← mean basis over locations & reps

H²   = σ²g / σ²p                           (broad-sense, mean basis)
H²_loc = σ²g / (σ²g + σ²e/r)              (per-location)

GA   = k × √σ²p × H²                      (k=2.063 for 5% selection)
GA%  = (GA / grand_mean) × 100
GCV  = (√σ²g / grand_mean) × 100
PCV  = (√σ²p / grand_mean) × 100
ECV  = (√σ²e / grand_mean) × 100
"""
from __future__ import annotations
import logging
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VarianceComponentEngine:
    """Estimates genetic variance components from combined-ANOVA MS values."""

    def __init__(self, config):
        self.config = config

    def estimate(
        self,
        anova_result: Dict[str, Any],
        df: pd.DataFrame,
        trait: str,
        geno_col: str,
        loc_col: Optional[str],
        rep_col: str,
    ) -> Dict[str, Any]:
        """
        Compute σ²g, σ²e, σ²gl, σ²p, H², GA, GCV, PCV and per-location variants.
        Returns a fully annotated dict with formula, components, and interpretation
        fields on every statistic.
        """
        k_sel = self.config.selection_intensity
        grand_mean = anova_result["grand_mean"]
        g = anova_result["n_genotypes"]
        r = anova_result["n_reps"]
        l = anova_result["n_locations"]

        MS = anova_result["MS"]
        MS_G = MS["G"]
        MS_GL = MS["GL"]
        MS_e = MS["error"]

        # ── Variance components ──────────────────────────────────────────
        sigma2_e = MS_e
        sigma2_gl_raw = (MS_GL - MS_e) / r
        sigma2_gl = max(0.0, sigma2_gl_raw)
        neg_gl = sigma2_gl_raw < 0

        sigma2_g_raw = (MS_G - MS_GL) / (r * l)
        sigma2_g = max(0.0, sigma2_g_raw)
        neg_g = sigma2_g_raw < 0

        # Phenotypic variance (mean basis)
        sigma2_p = sigma2_g + (sigma2_gl / l) + (sigma2_e / (r * l))

        # ── Heritability ─────────────────────────────────────────────────
        H2 = sigma2_g / sigma2_p if sigma2_p > 0 else 0.0

        # ── Genetic advance ──────────────────────────────────────────────
        GA = k_sel * math.sqrt(sigma2_p) * H2 if sigma2_p > 0 else 0.0
        GA_pct = (GA / grand_mean * 100) if grand_mean != 0 else 0.0

        # ── Coefficients of variation ────────────────────────────────────
        GCV = (math.sqrt(sigma2_g) / grand_mean * 100) if grand_mean != 0 else 0.0
        PCV = (math.sqrt(sigma2_p) / grand_mean * 100) if grand_mean != 0 else 0.0
        ECV = (math.sqrt(sigma2_e) / grand_mean * 100) if grand_mean != 0 else 0.0

        # ── Per-location heritability ────────────────────────────────────
        H2_per_loc = {}
        GA_per_loc = {}
        GCV_per_loc = {}
        if loc_col and loc_col in df.columns:
            for loc in anova_result["location_names"]:
                loc_df = df[df[loc_col] == loc]
                loc_grand_mean = float(loc_df[trait].mean())
                # Within-location error: use pooled MSE from combined model
                sigma2_p_loc = sigma2_g + sigma2_e / r
                H2_loc = sigma2_g / sigma2_p_loc if sigma2_p_loc > 0 else 0.0
                GA_loc = k_sel * math.sqrt(sigma2_p_loc) * H2_loc if sigma2_p_loc > 0 else 0.0
                GA_pct_loc = (GA_loc / loc_grand_mean * 100) if loc_grand_mean != 0 else 0.0
                GCV_loc = (math.sqrt(sigma2_g) / loc_grand_mean * 100) if loc_grand_mean != 0 else 0.0

                H2_per_loc[loc] = {
                    "value": round(H2_loc, 4),
                    "percent": round(H2_loc * 100, 2),
                    "formula": "H²_loc = σ²g / (σ²g + σ²e/r)",
                }
                GA_per_loc[loc] = {
                    "GA": round(GA_loc, 4),
                    "GA_percent": round(GA_pct_loc, 2),
                    "location_mean": round(loc_grand_mean, 4),
                }
                GCV_per_loc[loc] = round(GCV_loc, 2)

        warnings = []
        if neg_g:
            warnings.append(
                f"Negative σ²g estimate ({round(sigma2_g_raw, 4)}) corrected to 0. "
                "Genotype variation is confounded with G×L interaction."
            )
        if neg_gl:
            warnings.append(
                f"Negative σ²gl estimate ({round(sigma2_gl_raw, 4)}) corrected to 0. "
                "G×L interaction is negligible relative to error."
            )

        return {
            "grand_mean": round(grand_mean, 4),
            "n_genotypes": g,
            "n_locations": l,
            "n_reps": r,
            "mode": "multilocational" if l > 1 else "single",
            "warnings": warnings,

            "components": {
                "sigma2_g": {
                    "value": round(sigma2_g, 6),
                    "formula": "σ²g = (MSG − MSGL) / (r × l)",
                    "formula_description": "Genotypic variance estimated from genotype vs. G×L mean squares",
                    "components": {"MSG": round(MS_G, 4), "MSGL": round(MS_GL, 4), "r": r, "l": l},
                    "negative_corrected": neg_g,
                },
                "sigma2_e": {
                    "value": round(sigma2_e, 6),
                    "formula": "σ²e = MSE",
                    "formula_description": "Environmental (error) variance — within-cell variance",
                },
                "sigma2_gl": {
                    "value": round(sigma2_gl, 6),
                    "formula": "σ²gl = (MSGL − MSE) / r",
                    "formula_description": "Genotype × Location interaction variance",
                    "components": {"MSGL": round(MS_GL, 4), "MSE": round(MS_e, 4), "r": r},
                    "negative_corrected": neg_gl,
                },
                "sigma2_p": {
                    "value": round(sigma2_p, 6),
                    "formula": "σ²p = σ²g + (σ²gl / l) + (σ²e / (r × l))",
                    "formula_description": "Phenotypic variance of a genotype mean on a mean-basis across locations and reps",
                    "components": {
                        "sigma2_g": round(sigma2_g, 6),
                        "sigma2_gl": round(sigma2_gl, 6),
                        "sigma2_e": round(sigma2_e, 6),
                        "l": l, "r": r,
                    },
                },
            },

            "heritability": {
                "H2_broad": {
                    "value": round(H2, 4),
                    "percentage": round(H2 * 100, 2),
                    "formula": "H² = σ²g / σ²p",
                    "formula_description": "Broad-sense heritability on a mean basis across all locations",
                    "interpretation": _interp_H2(H2),
                },
                "H2_per_location": H2_per_loc,
            },

            "genetic_advance": {
                "GA": {
                    "value": round(GA, 4),
                    "formula": "GA = k × √σ²p × H²",
                    "formula_description": "Expected genetic advance under directional selection",
                    "components": {
                        "k": k_sel,
                        "sqrt_sigma2p": round(math.sqrt(sigma2_p), 4),
                        "H2": round(H2, 4),
                    },
                },
                "GA_percent": {
                    "value": round(GA_pct, 2),
                    "formula": "GA% = (GA / Grand Mean) × 100",
                    "interpretation": _interp_GA(GA_pct),
                },
                "GA_per_location": GA_per_loc,
            },

            "coefficients_of_variation": {
                "GCV": {
                    "value": round(GCV, 2),
                    "formula": "GCV = (√σ²g / Grand Mean) × 100",
                    "formula_description": "Genotypic coefficient of variation — relative genetic variability",
                    "interpretation": _interp_cv(GCV),
                },
                "PCV": {
                    "value": round(PCV, 2),
                    "formula": "PCV = (√σ²p / Grand Mean) × 100",
                    "formula_description": "Phenotypic coefficient of variation",
                },
                "ECV": {
                    "value": round(ECV, 2),
                    "formula": "ECV = (√σ²e / Grand Mean) × 100",
                    "formula_description": "Environmental coefficient of variation",
                },
                "GCV_per_location": GCV_per_loc,
                "GCV_PCV_ratio": round(GCV / PCV, 3) if PCV > 0 else None,
            },
        }


# ── Interpretation helpers ──────────────────────────────────────────────────

def _interp_H2(H2: float) -> str:
    if H2 >= 0.60:
        return "High heritability — genetic selection will be reliable and effective"
    if H2 >= 0.30:
        return "Moderate heritability — selection will be moderately effective"
    return "Low heritability — environmental effects dominant; consider progeny testing"


def _interp_GA(GA_pct: float) -> str:
    if GA_pct >= 20:
        return "High genetic advance — mass selection expected to be effective"
    if GA_pct >= 10:
        return "Moderate genetic advance"
    return "Low genetic advance — limited response to selection expected"


def _interp_cv(cv: float) -> str:
    if cv >= 20:
        return "High — substantial genotypic variation present for selection"
    if cv >= 10:
        return "Moderate genotypic variation"
    return "Low genotypic variation — limited scope for selection"
