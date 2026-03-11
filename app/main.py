# Version 2.0.0
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import hashlib
import json
import warnings
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import os
import math

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Suppress warnings
warnings.filterwarnings("ignore")

# =========================
# LOGGING CONFIGURATION
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURATION
# =========================

@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis"""
    alpha: float = 0.05
    figure_dpi: int = 300
    figure_format: str = 'png'
    include_interactions: bool = True
    max_interaction_level: int = 2
    p_value_correction: str = 'bonferroni'  # 'bonferroni', 'fdr_bh', or None
    effect_size_threshold_small: float = 0.01
    effect_size_threshold_medium: float = 0.06
    effect_size_threshold_large: float = 0.14
    
    def to_dict(self):
        return asdict(self)

# =========================
# DATA MODELS
# =========================

@dataclass
class AssumptionTest:
    """Assumption test results"""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    message: str = ""

@dataclass
class EffectSize:
    """Effect size calculations"""
    eta_squared: float
    omega_squared: float
    cohens_f: float
    interpretation: str

@dataclass
class AnalysisResult:
    """Complete analysis result for a single trait"""
    trait_name: str
    formula: str
    anova_table: Dict
    descriptive_stats: Dict
    means: Dict
    letters: Dict
    effect_sizes: Dict[str, EffectSize]
    assumptions: Dict[str, AssumptionTest]
    plots: Dict[str, str]
    interpretation: str
    timestamp: str
    analysis_id: str

# =========================
# PUBLICATION INTERPRETATION HELPERS
# =========================

def _pub_sig(p) -> str:
    """Return significance stars for a p-value."""
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def _pub_fmt(v, decimals: int = 4) -> str:
    """Format a value to `decimals` decimal places; return '—' for None/NaN/Inf."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return "—"
        return f"{fv:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _pub_fmt_large(v, decimals: int = 2) -> str:
    """Format with thousands separator (e.g. 4,360.00)."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return "—"
        return f"{fv:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _generate_backend_interpretation(result: Dict, response: str, pred: str,
                                     f_notation: str, alpha: float) -> str:
    """
    Generate a concise 2-paragraph backend interpretation of ANOVA results.
    Covers the main finding, top/bottom performers, and a practical implication.
    """
    anova = result.get("anova", {})
    means_dict = result.get("means", {})
    letters_dict = result.get("letters", {})
    r2 = result.get("r_squared")
    f_val = result.get("f_value")
    f_p = result.get("f_pvalue")

    # Significance of main effect
    pred_p = None
    if pred and pred in anova and isinstance(anova[pred], dict):
        pred_p = anova[pred].get("PR(>F)")

    # Top/bottom performers
    sorted_means = []
    if pred and pred in means_dict:
        sorted_means = sorted(means_dict[pred].items(), key=lambda x: float(x[1]), reverse=True)

    sig_str = "highly significant" if (pred_p is not None and float(pred_p) < 0.001) else \
              "significant" if (pred_p is not None and float(pred_p) < 0.05) else "not significant"
    p_str = "< 0.001" if (pred_p is not None and float(pred_p) < 0.001) else \
            (_pub_fmt(pred_p, 4) if pred_p is not None else "—")

    f_str = f"{f_notation} = {_pub_fmt(f_val, 2)}" if f_val else ""
    r2_str = f"{float(r2) * 100:.1f}%" if r2 else "—"

    para1 = (
        f"An analysis of variance revealed {sig_str} differences in {response} "
        f"among the tested treatments ({f_str}, p {('= ' + p_str) if not p_str.startswith('<') else p_str}). "
    )
    if r2:
        para1 += (
            f"The model explained {r2_str} of the total variation in {response}, "
            "indicating that the experimental factors accounted for the majority of observed differences. "
        )

    para2 = ""
    if sorted_means:
        best = sorted_means[0]
        worst = sorted_means[-1]
        diff = float(best[1]) - float(worst[1])
        pct = diff / float(worst[1]) * 100 if float(worst[1]) != 0 else 0
        best_letter = letters_dict.get(pred, {}).get(str(best[0]), "")
        para2 = (
            f"{best[0]} produced the highest mean {response} at {_pub_fmt_large(best[1], 2)}"
            f"{(' (' + best_letter + ')') if best_letter else ''}, "
            f"which was {_pub_fmt_large(diff, 2)} ({pct:.1f}%) higher than the lowest-performing "
            f"treatment {worst[0]} ({_pub_fmt_large(worst[1], 2)}). "
        )
        if len(sorted_means) > 2:
            para2 += (
                f"These results suggest that {best[0]} is the most suitable treatment under the "
                "trial conditions, and should be prioritized for further evaluation or adoption. "
                "Breeders and agronomists are advised to validate these findings across multiple "
                "environments before making broad recommendations."
            )

    return para1 + para2


def _generate_academic_interpretation(result: Dict, response: str, pred: str,
                                      f_notation: str, alpha: float,
                                      analysis_type: str, n_total: int) -> Dict[str, str]:
    """
    Generate Dr. Fayeun's structured 8-section academic interpretation.
    Returns a dict keyed by section name → paragraph text.
    """
    anova = result.get("anova", {})
    means_dict = result.get("means", {})
    letters_dict = result.get("letters", {})
    effect_sizes = result.get("effect_sizes", {})
    assumptions = result.get("assumptions", {})
    r2 = result.get("r_squared")
    adj_r2 = result.get("adj_r_squared")
    f_val = result.get("f_value")
    f_p = result.get("f_pvalue")
    overall_stats = result.get("descriptive_stats", {}).get("overall", {})

    pred_p = None
    if pred and pred in anova and isinstance(anova[pred], dict):
        pred_p = anova[pred].get("PR(>F)")

    sorted_means = []
    if pred and pred in means_dict:
        sorted_means = sorted(means_dict[pred].items(), key=lambda x: float(x[1]), reverse=True)

    main_es = (effect_sizes.get(pred) or
               next((v for k, v in effect_sizes.items() if k != "Residual"), None))

    sig_str = "highly significant" if (pred_p is not None and float(pred_p) < 0.001) else \
              "significant" if (pred_p is not None and float(pred_p) < 0.05) else "not significant"
    p_str = "< 0.001" if (pred_p is not None and float(pred_p) < 0.001) else _pub_fmt(pred_p, 4)

    # -- Section 1: Experimental Overview
    n_treatments = len(sorted_means)
    sec1 = (
        f"This study employed a {analysis_type} to evaluate the performance of "
        f"{n_treatments} treatment level{'s' if n_treatments != 1 else ''} "
        f"for the trait {response}, with a total of {n_total} observations. "
        f"The primary objective was to determine whether statistically significant differences "
        f"existed among the treatments and to identify the best-performing treatment(s) for "
        f"practical application. The experimental design provides an appropriate framework "
        f"for controlling extraneous variation and ensuring valid statistical inferences."
    )

    # -- Section 2: Statistical Results & Model Fit
    r2_pct = float(r2) * 100 if r2 else 0
    adj_r2_pct = float(adj_r2) * 100 if adj_r2 else 0
    sec2 = (
        f"The overall model was statistically {sig_str} "
        f"({f_notation} = {_pub_fmt(f_val, 2)}, p {('= ' + p_str) if not p_str.startswith('<') else p_str}), "
        f"confirming that treatment level differences in {response} are unlikely to be due to chance. "
        f"The coefficient of determination (R² = {_pub_fmt(r2, 4)}) indicates that {r2_pct:.1f}% "
        f"of the total variation in {response} was explained by the fitted model, "
        f"with an adjusted R² of {_pub_fmt(adj_r2, 4)} ({adj_r2_pct:.1f}%) after penalising "
        f"for model complexity. "
    )
    if main_es and isinstance(main_es, dict):
        eta = float(main_es.get("eta_squared", 0) or 0)
        ome = float(main_es.get("omega_squared", 0) or 0)
        interp = main_es.get("interpretation", "large")
        sec2 += (
            f"The eta-squared effect size (η² = {_pub_fmt(eta, 4)}) and the unbiased "
            f"omega-squared estimate (ω² = {_pub_fmt(ome, 4)}) both indicate a {interp} "
            f"practical effect, underscoring the agronomic importance of the observed differences."
        )

    # -- Section 3: Post-hoc Analysis
    if sorted_means:
        top3 = sorted_means[:3]
        top3_str = "; ".join(
            f"{t} ({_pub_fmt_large(m, 2)}{(' ' + letters_dict.get(pred, {}).get(str(t), '')) if letters_dict.get(pred, {}).get(str(t)) else ''})"
            for t, m in top3
        )
        bot = sorted_means[-1]
        sec3 = (
            f"Tukey's Honest Significant Difference (HSD) post-hoc test was applied to identify "
            f"pairwise differences among treatments. The three highest-performing treatments were: "
            f"{top3_str}. The lowest performer was {bot[0]} ({_pub_fmt_large(bot[1], 2)}). "
            f"Treatments sharing the same letter grouping were not significantly different "
            f"from each other at the α = {alpha} significance level. "
            f"The compact letter display facilitates rapid identification of statistically "
            f"homogeneous groups, which is essential for practical selection decisions."
        )
    else:
        sec3 = "Post-hoc comparisons were not applicable as no significant treatment effects were detected."

    # -- Section 4: Assumption Testing
    norm_rec = assumptions.get("normality", {})
    hom_rec = assumptions.get("homogeneity", {})
    norm_pass = norm_rec.get("passed", True)
    hom_pass = hom_rec.get("passed", True)
    norm_test = norm_rec.get("test_name", "Shapiro-Wilk")
    norm_stat = norm_rec.get("statistic")
    norm_p = norm_rec.get("p_value")
    hom_stat = hom_rec.get("statistic")
    hom_p = hom_rec.get("p_value")

    sec4 = (
        f"Prior to interpreting the ANOVA results, key statistical assumptions were evaluated. "
        f"The {norm_test} test for normality of residuals yielded "
        f"W = {_pub_fmt(norm_stat, 4)}, p = {_pub_fmt(norm_p, 4)}, "
        f"{'confirming that residuals were normally distributed' if norm_pass else 'suggesting a departure from normality'}. "
        f"Levene's test for homogeneity of variance produced "
        f"F = {_pub_fmt(hom_stat, 4)}, p = {_pub_fmt(hom_p, 4)}, "
        f"{'indicating that group variances were homogeneous' if hom_pass else 'suggesting heterogeneous variances'}. "
    )
    if norm_pass and hom_pass:
        sec4 += (
            "Both assumptions were satisfied, confirming that the parametric ANOVA "
            "is appropriate for this dataset and that the inferences drawn are statistically valid."
        )
    else:
        sec4 += (
            "Where assumptions were not fully met, ANOVA is generally robust to mild "
            "violations, particularly with balanced designs. Results should nonetheless "
            "be interpreted with appropriate caution, and non-parametric alternatives "
            "(Kruskal-Wallis) may be considered for confirmation."
        )

    # -- Section 5: Biological/Agronomic Interpretation
    sec5 = (
        f"The {sig_str} variation observed in {response} among the tested treatments "
        "reflects the influence of genetic and/or management factors on crop performance. "
        "Significant treatment differences are attributable to intrinsic genetic potential, "
        "differential responses to the growing environment, and/or treatment-specific "
        "management interactions. "
    )
    if sorted_means:
        best = sorted_means[0]
        sec5 += (
            f"The superior performance of {best[0]} suggests inherent genetic advantages "
            f"in traits that directly contribute to {response} under the prevailing "
            "environmental conditions of this trial. This finding aligns with the "
            "well-established principle that genotype × environment interactions shape "
            "phenotypic expression, and that superior genotypes maintain competitive "
            "performance across a range of conditions."
        )

    # -- Section 6: Practical Recommendations
    sec6 = ""
    if sorted_means:
        best = sorted_means[0]
        best_letter = letters_dict.get(pred, {}).get(str(best[0]), "")
        same_group = [t for t, _ in sorted_means
                      if letters_dict.get(pred, {}).get(str(t), "") == best_letter
                      and str(t) != str(best[0])]
        sec6 = (
            f"Based on the statistical analysis, {best[0]} is recommended as the primary "
            f"treatment of choice for {response} under conditions similar to those of this trial. "
        )
        if same_group:
            sec6 += (
                f"The following treatments were statistically comparable to {best[0]} "
                f"and represent viable alternatives: {', '.join(str(t) for t in same_group[:3])}. "
                "These alternatives may offer cost or logistical advantages without a "
                "statistically significant yield penalty. "
            )
        sec6 += (
            "It is strongly recommended that these findings be validated across multiple "
            "locations and seasons before large-scale adoption, to account for "
            "genotype × environment interaction effects."
        )

    # -- Section 7: Limitations & Future Work
    sec7 = (
        f"This study was conducted at a single location and/or season, which limits the "
        f"generalisability of the findings. The observed differences in {response} may be "
        "specific to the agro-ecological conditions of this trial site. Future studies "
        "should include multi-location, multi-season designs to assess the stability "
        "and adaptability of the top-performing treatments. Additionally, "
        "economic analyses incorporating input costs and market prices would strengthen "
        "the practical recommendations derived from these results."
    )

    # -- Section 8: Conclusion
    sec8 = (
        f"In conclusion, this {analysis_type} demonstrated {sig_str} variation in {response} "
        f"among the tested treatments (p {('= ' + p_str) if not p_str.startswith('<') else p_str}). "
    )
    if sorted_means:
        best = sorted_means[0]
        sec8 += (
            f"The treatment {best[0]} consistently outperformed the others, "
            f"achieving a mean {response} of {_pub_fmt_large(best[1], 2)}. "
        )
    sec8 += (
        "These findings contribute to the evidence base for evidence-based agricultural "
        "decision-making and provide a foundation for further research into the mechanisms "
        "underlying the observed performance differences. "
        "Multi-environment validation and economic assessment are recommended as "
        "essential next steps before widespread adoption."
    )

    return {
        "Section 1 — Experimental Overview": sec1,
        "Section 2 — Statistical Results & Model Fit": sec2,
        "Section 3 — Post-hoc & Detailed Analysis": sec3,
        "Section 4 — Assumption Testing": sec4,
        "Section 5 — Biological / Agronomic Interpretation": sec5,
        "Section 6 — Practical Recommendations": sec6,
        "Section 7 — Limitations & Future Work": sec7,
        "Section 8 — Conclusion": sec8,
    }


# =========================
# ANOVA HTML TABLE BUILDER
# =========================

_ANOVA_CSS = """<style>
.vv-pub-table{border-collapse:collapse;width:100%;font-family:'Times New Roman',Times,serif;font-size:11pt;margin-bottom:8px}
.vv-pub-table caption{font-weight:bold;font-size:12pt;text-align:left;padding-bottom:4px;caption-side:top}
.vv-pub-table th{background-color:#1a3a5c;color:#fff;padding:6px 10px;text-align:center;border:1px solid #bbb;font-size:10pt}
.vv-pub-table td{padding:5px 10px;border:1px solid #ccc;text-align:center;vertical-align:middle}
.vv-pub-table tr:nth-child(even) td{background-color:#f0f4f8}
.vv-pub-table .td-left{text-align:left}
.vv-pub-table .total-row td{font-weight:bold;border-top:2px solid #1a3a5c}
.vv-pub-table .footnote td{font-size:9pt;color:#555;text-align:left;border:none;padding:2px 4px}
.sig-mark{color:#b00;font-weight:bold}.sig-ns{color:#888}
</style>"""

def _anova_html_sig(p) -> str:
    if p is None: return ""
    try:
        pf = float(p)
        if pf < 0.001: return '<span class="sig-mark">***</span>'
        if pf < 0.01:  return '<span class="sig-mark">**</span>'
        if pf < 0.05:  return '<span class="sig-mark">*</span>'
        return '<span class="sig-ns">ns</span>'
    except Exception: return ""

def _anova_html_table(table_number: str, title: str, headers: List[str],
                      rows: List[List[str]], footnotes: Optional[List[str]] = None,
                      total_row: Optional[List[str]] = None,
                      left_cols: Optional[List[int]] = None) -> str:
    left_cols = left_cols or [0]
    parts = [_ANOVA_CSS, f'<table class="vv-pub-table">',
             f'  <caption>{table_number}: {title}</caption>', '  <thead><tr>']
    for h in headers:
        parts.append(f'    <th>{h}</th>')
    parts += ['  </tr></thead>', '  <tbody>']
    for row in rows:
        parts.append('    <tr>')
        for ci, cell in enumerate(row):
            cls = ' class="td-left"' if ci in left_cols else ''
            parts.append(f'      <td{cls}>{cell}</td>')
        parts.append('    </tr>')
    if total_row:
        parts.append('    <tr class="total-row">')
        for ci, cell in enumerate(total_row):
            cls = ' class="td-left"' if ci in left_cols else ''
            parts.append(f'      <td{cls}>{cell}</td>')
        parts.append('    </tr>')
    parts.append('  </tbody>')
    if footnotes:
        parts.append('  <tfoot>')
        colspan = len(headers)
        for fn in footnotes:
            parts.append(f'  <tr class="footnote"><td colspan="{colspan}">{fn}</td></tr>')
        parts.append('  </tfoot>')
    parts.append('</table>')
    return '\n'.join(parts)


def build_anova_html_tables(pub: Dict) -> List[Dict[str, str]]:
    """
    Convert the structured publication tables dict (from build_publication_tables)
    into a list of {"name": str, "html": str} for direct frontend rendering.
    """
    result: List[Dict[str, str]] = []

    # ── Table 1: ANOVA ────────────────────────────────────────────────────────
    anova_tbl = pub.get("anova_table", {})
    if anova_tbl:
        rows = []
        for row in anova_tbl.get("rows", []):
            rows.append([str(c) for c in row])
        ft = anova_tbl.get("footer", {})
        footnotes = [
            f"R² = {_pub_fmt(ft.get('r_squared'), 4)};  "
            f"Adj. R² = {_pub_fmt(ft.get('adj_r_squared'), 4)};  "
            f"CV = {_pub_fmt(ft.get('cv'), 2)}%;  "
            f"F-test: {ft.get('f_test', '—')}",
            "Type II ANOVA.  Significance: *** p &lt; 0.001; ** p &lt; 0.01; * p &lt; 0.05; ns = not significant.",
        ] + (anova_tbl.get("footnotes") or [])
        html = _anova_html_table(
            "Table 1", anova_tbl.get("title", "ANOVA"),
            anova_tbl.get("headers", []), rows,
            footnotes=footnotes, left_cols=[0])
        result.append({"name": "ANOVA", "html": html})

    # ── Table 2: Descriptive Statistics ──────────────────────────────────────
    desc_tbl = pub.get("descriptive_table", {})
    if desc_tbl:
        rows = [[str(c) for c in row] for row in desc_tbl.get("rows", [])]
        overall = desc_tbl.get("overall_row")
        total_row = [str(c) for c in overall] if overall else None
        html = _anova_html_table(
            "Table 2", desc_tbl.get("title", "Descriptive Statistics"),
            desc_tbl.get("headers", []), rows,
            footnotes=desc_tbl.get("footnotes"),
            total_row=total_row, left_cols=[0])
        result.append({"name": "Descriptive Statistics", "html": html})

    # ── Table 3: Post-hoc Tukey HSD ───────────────────────────────────────────
    ph_tbl = pub.get("posthoc_table", {})
    if ph_tbl:
        rows = [[str(c) for c in row] for row in ph_tbl.get("rows", [])]
        html = _anova_html_table(
            "Table 3", ph_tbl.get("title", "Post-hoc Tukey HSD"),
            ph_tbl.get("headers", []), rows,
            footnotes=ph_tbl.get("footnotes"), left_cols=[0, 4])
        result.append({"name": "Post-hoc Tukey HSD", "html": html})

    # ── Table 4: Assumption Tests ─────────────────────────────────────────────
    ass_tbl = pub.get("assumptions_table", {})
    if ass_tbl:
        rows = [[str(c) for c in row] for row in ass_tbl.get("rows", [])]
        html = _anova_html_table(
            "Table 4", ass_tbl.get("title", "Assumption Tests"),
            ass_tbl.get("headers", []), rows,
            footnotes=ass_tbl.get("footnotes"), left_cols=[0, 4])
        result.append({"name": "Assumption Tests", "html": html})

    # ── Table 5: Model Fit ────────────────────────────────────────────────────
    fit_tbl = pub.get("model_fit_table", {})
    if fit_tbl:
        rows = [[str(c) for c in row] for row in fit_tbl.get("rows", [])]
        html = _anova_html_table(
            "Table 5", fit_tbl.get("title", "Model Fit Statistics"),
            fit_tbl.get("headers", []), rows,
            footnotes=fit_tbl.get("footnotes"), left_cols=[0, 3])
        result.append({"name": "Model Fit Statistics", "html": html})

    return result


# =========================
# STATISTICAL ANALYZER
# =========================

class StatisticalAnalyzer:
    """Core statistical analysis engine"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        logger.info("StatisticalAnalyzer initialized with config: %s", self.config)
    
    def detect_variable_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Auto-detect categorical and continuous variables"""
        categorical = []
        continuous = []
        
        # Common blocking factor names
        block_keywords = ['block', 'rep', 'replicate', 'batch', 'plot', 'field']
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Check if column should be categorical
            if (df[col].dtype in ['object', 'category'] or
                col_lower in block_keywords or
                any(keyword in col_lower for keyword in block_keywords) or
                (df[col].dtype in ['int64', 'float64'] and df[col].nunique() < 10)):
                categorical.append(col)
            else:
                continuous.append(col)
        
        logger.info(f"Detected {len(categorical)} categorical and {len(continuous)} continuous variables")
        return categorical, continuous
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate input data quality"""
        errors = []
        warnings = []
        
        # Check minimum sample size
        if len(df) < 10:
            errors.append(f"Sample size too small ({len(df)} rows). Minimum 10 rows required.")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            cols_with_missing = missing[missing > 0].index.tolist()
            missing_pct = (missing[missing > 0] / len(df) * 100).round(1)
            for col, pct in zip(cols_with_missing, missing_pct):
                warnings.append(f"Column '{col}' has {pct}% missing values")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                warnings.append(f"Column '{col}' has constant values - may not be informative")
        
        # Check for extreme outliers (beyond 6 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 6).sum()
                if outliers > 0:
                    warnings.append(f"Column '{col}' has {outliers} extreme outliers (>6 SD)")
        
        return {"errors": errors, "warnings": warnings}
    
    def build_formula(self, response: str, predictors: List[str], 
                     blocks: List[str] = None) -> str:
        """Build model formula with interactions"""
        all_predictors = predictors.copy()
        
        # Add blocks if provided
        if blocks:
            all_predictors.extend(blocks)
        
        # Start with main effects
        formula = f"{response} ~ " + " + ".join(all_predictors)
        
        # Add interactions if requested and appropriate
        if self.config.include_interactions and len(predictors) >= 2:
            interactions = []
            
            # Add two-way interactions
            for i in range(len(predictors)):
                for j in range(i+1, len(predictors)):
                    interactions.append(f"{predictors[i]}:{predictors[j]}")
            
            # Add three-way interactions if specified and enough predictors
            if self.config.max_interaction_level >= 3 and len(predictors) >= 3:
                for i in range(len(predictors)):
                    for j in range(i+1, len(predictors)):
                        for k in range(j+1, len(predictors)):
                            interactions.append(f"{predictors[i]}:{predictors[j]}:{predictors[k]}")
            
            if interactions:
                formula += " + " + " + ".join(interactions)
        
        return formula
    
    def calculate_effect_sizes(self, anova_table: pd.DataFrame, 
                              ss_total: float) -> Dict[str, EffectSize]:
        """Calculate various effect sizes"""
        effect_sizes = {}
        
        for effect in anova_table.index:
            if effect != 'Residual':
                ss_effect = anova_table.loc[effect, 'sum_sq']
                df_effect = anova_table.loc[effect, 'df']
                ms_effect = ss_effect / df_effect if df_effect > 0 else 0
                if 'Residual' in anova_table.index:
                    _ss_res = anova_table.loc['Residual', 'sum_sq']
                    _df_res = anova_table.loc['Residual', 'df']
                    ms_error = _ss_res / _df_res if _df_res > 0 else 0
                else:
                    ms_error = 0
                
                # Eta-squared
                eta_sq = ss_effect / ss_total
                
                # Omega-squared
                if ms_error > 0:
                    omega_sq = (ss_effect - (df_effect * ms_error)) / (ss_total + ms_error)
                else:
                    omega_sq = eta_sq
                
                # Cohen's f
                if ms_error > 0:
                    cohens_f = np.sqrt((ss_effect / df_effect) / ms_error) if ms_error > 0 else 0
                else:
                    cohens_f = np.sqrt(eta_sq / (1 - eta_sq)) if eta_sq < 1 else float('inf')
                
                # Interpret effect size
                if eta_sq < self.config.effect_size_threshold_small:
                    interpretation = "negligible"
                elif eta_sq < self.config.effect_size_threshold_medium:
                    interpretation = "small"
                elif eta_sq < self.config.effect_size_threshold_large:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                effect_sizes[effect] = EffectSize(
                    eta_squared=eta_sq,
                    omega_squared=max(0, omega_sq),  # Can't be negative
                    cohens_f=cohens_f,
                    interpretation=interpretation
                )
        
        return effect_sizes
    
    def check_assumptions(self, data: pd.DataFrame, formula: str, 
                         group_var: str) -> Dict[str, AssumptionTest]:
        """Comprehensive assumption testing"""
        assumptions = {}
        
        try:
            # Fit model
            model = ols(formula, data=data).fit()
            residuals = model.resid
            fitted = model.fittedvalues
            
            # 1. Normality test (Shapiro-Wilk)
            if len(residuals) <= 5000:  # Shapiro has sample size limit
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                assumptions['normality'] = AssumptionTest(
                    test_name="Shapiro-Wilk",
                    statistic=shapiro_stat,
                    p_value=shapiro_p,
                    passed=shapiro_p > self.config.alpha,
                    message=f"Data {'appears' if shapiro_p > self.config.alpha else 'does not appear'} normally distributed"
                )
            else:
                # Use Kolmogorov-Smirnov for large samples
                ks_stat, ks_p = stats.kstest(residuals, 'norm')
                assumptions['normality'] = AssumptionTest(
                    test_name="Kolmogorov-Smirnov",
                    statistic=ks_stat,
                    p_value=ks_p,
                    passed=ks_p > self.config.alpha,
                    message=f"Data {'appears' if ks_p > self.config.alpha else 'does not appear'} normally distributed"
                )
            
            # 2. Homogeneity of variance (Levene's test)
            groups = []
            unique_groups = data[group_var].unique()
            
            if len(unique_groups) > 1:
                for level in unique_groups:
                    group_data = data[data[group_var] == level][model.endog_names]
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) > 1:
                    levene_stat, levene_p = stats.levene(*groups)
                    assumptions['homogeneity'] = AssumptionTest(
                        test_name="Levene's Test",
                        statistic=levene_stat,
                        p_value=levene_p,
                        passed=levene_p > self.config.alpha,
                        message=f"Variances {'are' if levene_p > self.config.alpha else 'are not'} homogeneous"
                    )
            
            # 3. Independence test (Durbin-Watson)
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(residuals)
            # DW ≈ 2 indicates independence
            assumptions['independence'] = AssumptionTest(
                test_name="Durbin-Watson",
                statistic=dw_stat,
                p_value=None,  # Not applicable
                passed=1.5 < dw_stat < 2.5,
                message=f"Residuals {'appear' if 1.5 < dw_stat < 2.5 else 'may not be'} independent (DW={dw_stat:.3f})"
            )
            
            # 4. Linearity test (Rainbow test)
            from statsmodels.stats.diagnostic import linear_rainbow
            rainbow_stat, rainbow_p = linear_rainbow(model)
            assumptions['linearity'] = AssumptionTest(
                test_name="Rainbow Test",
                statistic=rainbow_stat,
                p_value=rainbow_p,
                passed=rainbow_p > self.config.alpha,
                message=f"Relationship {'appears' if rainbow_p > self.config.alpha else 'may not be'} linear"
            )
            
        except Exception as e:
            logger.warning(f"Assumption checks failed: {str(e)}")
            assumptions['error'] = AssumptionTest(
                test_name="Error",
                statistic=0,
                p_value=1,
                passed=False,
                message=f"Assumption testing failed: {str(e)}"
            )
        
        return assumptions
    
    def compact_letter_display(self, tukey_result) -> Dict[str, str]:
        """Generate compact letter display from Tukey HSD results"""
        try:
            groups = list(tukey_result.groups_unique)
            n_groups = len(groups)
            
            if n_groups == 0:
                return {}
            
            # Create p-value matrix
            p_matrix = np.ones((n_groups, n_groups))
            
            # Fill matrix with p-values
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    if i < j:
                        # Find the comparison in tukey result
                        mask1 = (tukey_result.groups == g1) & (tukey_result.groups_other == g2)
                        mask2 = (tukey_result.groups == g2) & (tukey_result.groups_other == g1)
                        
                        if mask1.any():
                            p_matrix[i, j] = tukey_result.pvalues[mask1][0]
                            p_matrix[j, i] = p_matrix[i, j]
                        elif mask2.any():
                            p_matrix[i, j] = tukey_result.pvalues[mask2][0]
                            p_matrix[j, i] = p_matrix[i, j]
            
            # Generate letters using algorithm
            letters = {group: '' for group in groups}
            
            # Start with first group
            current_letter = 65  # ASCII 'A'
            
            for i in range(n_groups):
                if not letters[groups[i]]:
                    # Start new letter group
                    group_members = [i]
                    letters[groups[i]] = chr(current_letter)
                    
                    # Find all groups not significantly different from current group
                    for j in range(i + 1, n_groups):
                        if not letters[groups[j]]:
                            # Check if non-significant with all current group members
                            non_sig_with_all = True
                            for member in group_members:
                                if p_matrix[member, j] <= self.config.alpha:
                                    non_sig_with_all = False
                                    break
                            
                            if non_sig_with_all:
                                group_members.append(j)
                                letters[groups[j]] = chr(current_letter)
                    
                    current_letter += 1
            
            return letters
            
        except Exception as e:
            logger.error(f"Letter display generation failed: {str(e)}")
            return {group: chr(65 + i) for i, group in enumerate(groups)}
    
    def build_publication_tables(self, result: Dict, response: str,
                                primary_predictor: str = None,
                                analysis_type: str = "ANOVA") -> Dict:
        """
        Build publication-ready formatted tables from an ANOVA result dict.

        Returns a dict with:
          report_header           — Metadata block
          anova_table             — Source / df / SS / MS / F / p / Sig  (Table 1)
          descriptive_table       — Treatment / n / Mean / SD / SE / CV / Min / Max  (Table 2)
          posthoc_table           — Rank / Treatment / Mean / Letter / Interpretation  (Table 3)
          assumptions_table       — Test / Statistic / p / Result / Interpretation  (Table 4)
          model_fit_table         — R² / adj-R² / F / p / η² / ω² / Cohen's f  (Table 5)
          backend_interpretation  — Concise 2-paragraph plain-language summary
          academic_interpretation — Dr. Fayeun's 8-section structured report
        """
        alpha = self.config.alpha
        pub: Dict[str, Any] = {}

        anova_raw    = result.get("anova", {})
        means_dict   = result.get("means", {})
        letters_dict = result.get("letters", {})
        desc_raw     = result.get("descriptive_stats", {})
        overall_st   = desc_raw.get("overall", {})

        # Resolve primary predictor
        desc_keys = [k for k in desc_raw if k != "overall"]
        pred = (primary_predictor if (primary_predictor and primary_predictor in desc_raw)
                else (desc_keys[0] if desc_keys else None))
        pred_ph = (primary_predictor if (primary_predictor and primary_predictor in means_dict)
                   else (list(means_dict.keys())[0] if means_dict else None))

        # Residual MS (for RSE and F notation)
        ms_residual = None
        df_residual = None
        if "Residual" in anova_raw and isinstance(anova_raw["Residual"], dict):
            _ss_r = anova_raw["Residual"].get("sum_sq")
            _df_r = anova_raw["Residual"].get("df")
            if _ss_r is not None and _df_r and float(_df_r) > 0:
                ms_residual = float(_ss_r) / float(_df_r)
                df_residual = float(_df_r)

        # Treatment df for F(df1,df2) notation
        df1 = None
        if pred and pred in anova_raw and isinstance(anova_raw[pred], dict):
            _d = anova_raw[pred].get("df")
            if _d is not None:
                df1 = int(float(_d))
        df2 = int(df_residual) if df_residual else None
        f_notation = f"F({df1},{df2})" if (df1 and df2) else "F"

        # Overall CV
        cv_overall = None
        if overall_st.get("mean") and overall_st.get("std") and float(overall_st["mean"]) != 0:
            cv_overall = float(overall_st["std"]) / float(overall_st["mean"]) * 100

        n_total = int(overall_st.get("n", 0))

        # ── REPORT HEADER ──────────────────────────────────────────────────
        level_names = list((means_dict.get(pred_ph) or {}).keys())
        n_levels = len(level_names)
        n_per    = n_total // n_levels if n_levels else 0
        pub["report_header"] = {
            "title": "VivaSense™ — Statistical Analysis Report",
            "generated": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
            "software": "VivaSense V2.2",
            "sections": [
                {"label": "Analysis Type",       "value": analysis_type},
                {"label": "Trait / Variable",    "value": response},
                {"label": "Treatment Factor",    "value": pred or "—"},
                {"label": "Number of Levels",    "value": (
                    f"{n_levels} ({', '.join(str(x) for x in level_names[:5])}"
                    f"{'...' if len(level_names) > 5 else ')'}"
                ) if level_names else "—"},
                {"label": "Total Observations",  "value": str(n_total)},
                {"label": "Replications (est.)", "value": str(n_per) if n_per else "—"},
                {"label": "Significance Level",  "value": f"α = {alpha}"},
                {"label": "Date Analyzed",       "value": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")},
            ],
        }

        # ── TABLE 1: ANOVA ─────────────────────────────────────────────────
        anova_rows = []
        ss_total = sum(
            (float(v.get("sum_sq") or 0))
            for v in anova_raw.values() if isinstance(v, dict)
        )
        df_total = sum(
            (float(v.get("df") or 0))
            for v in anova_raw.values() if isinstance(v, dict)
        )
        for src, vals in anova_raw.items():
            if not isinstance(vals, dict):
                continue
            ss  = vals.get("sum_sq")
            df_ = vals.get("df")
            ms  = float(ss) / float(df_) if (ss is not None and df_ and float(df_) > 0) else None
            f   = vals.get("F")
            p   = vals.get("PR(>F)")
            p_disp = ("< 0.001" if (p is not None and float(p) < 0.001)
                      else _pub_fmt(p, 4))
            anova_rows.append([
                src,
                str(int(float(df_))) if df_ is not None else "—",
                _pub_fmt_large(ss, 2),
                _pub_fmt_large(ms, 2),
                _pub_fmt(f, 3),
                p_disp,
                _pub_sig(p),
            ])
        anova_rows.append([
            "Total",
            str(int(df_total)) if df_total else "—",
            _pub_fmt_large(ss_total, 2),
            "—", "—", "—", "",
        ])
        r2    = result.get("r_squared")
        adj_r2= result.get("adj_r_squared")
        f_val = result.get("f_value")
        f_p   = result.get("f_pvalue")
        pub["anova_table"] = {
            "title": f"Analysis of Variance for {response}",
            "table_number": "Table 1",
            "headers": ["Source of Variation", "df", "Sum of Squares (SS)",
                        "Mean Square (MS)", "F-value", "p-value", "Sig."],
            "rows": anova_rows,
            "footer": {
                "r_squared":     f"R² = {_pub_fmt(r2, 4)} ({_pub_fmt(float(r2)*100 if r2 else None, 2)}% of variance explained)",
                "adj_r_squared": f"Adjusted R² = {_pub_fmt(adj_r2, 4)}",
                "cv":            f"Coefficient of Variation (CV) = {_pub_fmt(cv_overall, 2)}%",
                "f_test":        (f"Overall F-test: {f_notation} = {_pub_fmt(f_val, 2)}, "
                                  f"p {'< 0.001' if f_p and float(f_p) < 0.001 else '= ' + _pub_fmt(f_p, 4)} "
                                  f"{_pub_sig(f_p)}"),
            },
            "footnotes": [
                "Significance codes:  *** p < 0.001  ** p < 0.01  * p < 0.05  ns p ≥ 0.05",
                f"Type II Sum of Squares.  α = {alpha}.",
                "SS = Sum of Squares, MS = Mean Square, df = degrees of freedom.",
            ],
        }

        # ── TABLE 2: DESCRIPTIVE STATISTICS ────────────────────────────────
        desc_rows = []
        if pred and pred in desc_raw:
            sorted_groups = sorted(
                desc_raw[pred].items(),
                key=lambda kv: float(kv[1].get("mean") or 0),
                reverse=True,
            )
            for grp, sd in sorted_groups:
                n = sd.get("count", sd.get("n"))
                desc_rows.append([
                    str(grp),
                    str(int(float(n))) if n is not None else "—",
                    _pub_fmt_large(sd.get("mean"), 2),
                    _pub_fmt(sd.get("std"),  2),
                    _pub_fmt(sd.get("sem"),  2),
                    _pub_fmt_large(sd.get("min"),  2),
                    _pub_fmt_large(sd.get("max"),  2),
                    _pub_fmt(sd.get("cv"),   2),
                ])
        # Overall row
        if overall_st:
            desc_rows.append([
                "Overall",
                str(n_total) if n_total else "—",
                _pub_fmt_large(overall_st.get("mean"), 2),
                _pub_fmt(overall_st.get("std"),  2),
                _pub_fmt(overall_st.get("sem"),  2),
                _pub_fmt_large(overall_st.get("min"),  2),
                _pub_fmt_large(overall_st.get("max"),  2),
                _pub_fmt(overall_st.get("cv"),   2),
            ])
        pub["descriptive_table"] = {
            "title": f"Descriptive Statistics for {response}",
            "table_number": "Table 2",
            "headers": ["Treatment", "n", f"Mean ({response})", "Std Dev (SD)",
                        "Std Error (SE)", "Min", "Max", "CV (%)"],
            "rows": desc_rows,
            "footnotes": [
                "Values shown to 2 decimal places.",
                "SD = Standard Deviation.  SE = Standard Error = SD / √n.",
                "CV = Coefficient of Variation (%) = (SD / Mean) × 100.",
                "Rows sorted by mean (highest to lowest).  Last row = overall statistics.",
            ],
        }

        # ── TABLE 3: POST-HOC (Tukey HSD) ─────────────────────────────────
        posthoc_rows = []
        effect_sig = False
        if pred_ph and pred_ph in anova_raw and isinstance(anova_raw[pred_ph], dict):
            _pp = anova_raw[pred_ph].get("PR(>F)")
            effect_sig = _pp is not None and float(_pp) < alpha

        if pred_ph and pred_ph in means_dict:
            letters  = letters_dict.get(pred_ph, {})
            s_means  = sorted(means_dict[pred_ph].items(), key=lambda x: float(x[1]), reverse=True)
            n_s      = len(s_means)
            top_ltr  = letters.get(str(s_means[0][0]), "") if s_means else ""
            bot_ltr  = letters.get(str(s_means[-1][0]), "") if s_means else ""
            for rank, (trt, mean) in enumerate(s_means, 1):
                ltr = letters.get(str(trt), "—")
                if rank == 1:
                    interp = "Highest performing"
                elif rank == n_s:
                    interp = "Lowest performing"
                elif ltr and ltr == top_ltr:
                    interp = f"Comparable to {s_means[0][0]}"
                elif ltr and ltr == bot_ltr:
                    interp = f"Comparable to {s_means[-1][0]}"
                else:
                    interp = "Intermediate performance"
                posthoc_rows.append([str(rank), str(trt), _pub_fmt_large(mean, 2), ltr, interp])

        pub["posthoc_table"] = {
            "title": f"Post-hoc Mean Comparisons (Tukey HSD) for {response}",
            "table_number": "Table 3",
            "headers": ["Rank", "Treatment", "Mean", "Letter Grouping", "Interpretation"],
            "rows": posthoc_rows,
            "significant_effect": effect_sig,
            "footnotes": [
                "Tukey's Honest Significant Difference (HSD) test.",
                "Treatments sharing the same letter are NOT significantly different (p > 0.05).",
                "Treatments with different letters ARE significantly different (p ≤ 0.05).",
                "Letter grouping shown only when the main effect is significant (p < α).",
                "Ranked from highest to lowest mean.",
            ],
        }

        # ── TABLE 4: ASSUMPTION TESTS ──────────────────────────────────────
        assumptions_raw = result.get("assumptions", {})
        assumption_rows = []
        all_passed = True
        for test_key, test_data in assumptions_raw.items():
            if not isinstance(test_data, dict) or test_key == "error":
                continue
            name   = test_data.get("test_name", test_key)
            stat   = test_data.get("statistic")
            p      = test_data.get("p_value")
            passed = test_data.get("passed")
            msg    = test_data.get("message", "")
            if passed is False:
                all_passed = False
            result_str = "PASS ✓" if passed is True else ("FAIL ✗" if passed is False else "—")
            assumption_rows.append([name, _pub_fmt(stat, 4), _pub_fmt(p, 4), result_str, msg])
        overall_assess = (
            "✓ PASS — All ANOVA assumptions satisfied. Parametric results are valid."
            if all_passed else
            "⚠ WARNING — One or more assumptions violated. Interpret with caution."
        )
        pub["assumptions_table"] = {
            "title": "Statistical Assumption Tests",
            "table_number": "Table 4",
            "headers": ["Test", "Statistic", "p-value", "Result", "Interpretation"],
            "rows": assumption_rows,
            "overall_assessment": overall_assess,
            "footnotes": [
                "Normality: Shapiro-Wilk (n ≤ 5000) or Kolmogorov-Smirnov (n > 5000).",
                "Homogeneity of variance: Levene's test for equality of group variances.",
                "Independence: Durbin-Watson statistic (acceptable range 1.5–2.5; no p-value).",
                f"PASS criterion: p > {alpha} for parametric tests.",
            ],
        }

        # ── TABLE 5: MODEL FIT ─────────────────────────────────────────────
        es_dict = result.get("effect_sizes", {})
        main_es = (es_dict.get(pred_ph) or es_dict.get(pred)
                   or next((v for k, v in es_dict.items() if k != "Residual"), None))
        rse = math.sqrt(ms_residual) if ms_residual and ms_residual > 0 else None

        r2_interp = "—"
        if r2:
            r2f = float(r2)
            if r2f >= 0.9:  r2_interp = f"Excellent fit ({r2f*100:.1f}% variance explained)"
            elif r2f >= 0.7: r2_interp = f"Good fit ({r2f*100:.1f}% variance explained)"
            elif r2f >= 0.5: r2_interp = f"Moderate fit ({r2f*100:.1f}% variance explained)"
            else:            r2_interp = f"Poor fit ({r2f*100:.1f}% variance explained)"

        model_rows: List[List[str]] = [
            ["R² (Coefficient of Determination)", _pub_fmt(r2,    4), r2_interp],
            ["Adjusted R² (accounts for df)",     _pub_fmt(adj_r2, 4), "R² penalized for model complexity"],
            [f"{f_notation} (overall F-statistic)", _pub_fmt(f_val, 3),
             ("Highly significant" if f_p and float(f_p) < 0.001 else
              "Significant" if f_p and float(f_p) < 0.05 else "Not significant") + f" ({_pub_sig(f_p)})"],
            ["Model p-value",
             "< 0.001" if f_p and float(f_p) < 0.001 else _pub_fmt(f_p, 4),
             _pub_sig(f_p)],
            ["Residual Standard Error (RSE)", _pub_fmt(rse, 4),
             "√MS_error — average prediction error"],
            ["Coefficient of Variation (%)", _pub_fmt(cv_overall, 2),
             "CV < 10% excellent, 10–20% acceptable, > 20% high"],
        ]
        if isinstance(main_es, dict):
            eta  = float(main_es.get("eta_squared", 0) or 0)
            ome  = float(main_es.get("omega_squared", 0) or 0)
            cf   = float(main_es.get("cohens_f", 0) or 0)
            interp = main_es.get("interpretation", "")
            ome_lbl = ("Large" if ome >= 0.14 else "Medium" if ome >= 0.06
                       else "Small" if ome >= 0.01 else "Negligible")
            cf_lbl  = ("Very large" if cf >= 0.80 else "Large" if cf >= 0.35
                       else "Moderate" if cf >= 0.10 else "Small")
            model_rows += [
                ["η² (Eta-squared)",   _pub_fmt(eta, 4), f"Effect size: {interp} (η² of treatment)"],
                ["ω² (Omega-squared)", _pub_fmt(ome, 4), f"{ome_lbl} effect (unbiased estimate)"],
                ["Cohen's f",          _pub_fmt(cf,  4), f"{cf_lbl} standardized effect"],
            ]
        pub["model_fit_table"] = {
            "title": f"Model Fit and Effect Size Statistics for {response}",
            "table_number": "Table 5",
            "headers": ["Statistic", "Value", "Interpretation"],
            "rows": model_rows,
            "footnotes": [
                "η² benchmarks: < 0.01 negligible, 0.01–0.06 small, 0.06–0.14 medium, ≥ 0.14 large.",
                "ω² is the preferred unbiased population effect size estimator.",
                "Cohen's f: 0.10 small, 0.25 medium, 0.40 large, ≥ 0.80 very large.",
            ],
        }

        # ── BACKEND INTERPRETATION ─────────────────────────────────────────
        pub["backend_interpretation"] = _generate_backend_interpretation(
            result, response, pred, f_notation, alpha
        )

        # ── DR. FAYEUN'S ACADEMIC INTERPRETATION ──────────────────────────
        pub["academic_interpretation"] = _generate_academic_interpretation(
            result, response, pred, f_notation, alpha, analysis_type, n_total
        )

        # ── HTML TABLES (copy-paste ready for Word / LaTeX) ────────────────
        try:
            pub["html_tables"] = build_anova_html_tables(pub)
        except Exception as _ht_exc:
            logger.warning("Could not build ANOVA HTML tables: %s", _ht_exc)
            pub["html_tables"] = []

        return pub

    def generate_publication_bar_chart(
        self,
        df: pd.DataFrame,
        response: str,
        predictor: str,
        letters: Dict[str, str],
        means: Dict[str, float],
    ) -> str:
        """
        Generate a publication-quality bar chart (300 DPI) with:
        - Bars sorted by mean (highest → lowest)
        - ±1 SE error bars
        - Tukey HSD letter annotations above each bar
        - Professional colour scheme (#4A90E2)
        Returns base64-encoded PNG string.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np

            sorted_items = sorted(means.items(), key=lambda x: float(x[1]), reverse=True)
            labels = [str(k) for k, _ in sorted_items]
            mean_vals = [float(v) for _, v in sorted_items]

            # Compute SEM per group from df
            grouped = df.groupby(predictor)[response]
            sems = [float(grouped.get_group(k).sem()) if k in grouped.groups else 0.0
                    for k in [ki for ki, _ in sorted_items]]

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 6))
            x_pos = np.arange(len(labels))
            bar_colour = "#4A90E2"
            bars = ax.bar(
                x_pos, mean_vals, yerr=sems,
                capsize=5, color=bar_colour, edgecolor="#2c5f9e",
                alpha=0.88, ecolor="#333333", linewidth=1.2,
                error_kw={"linewidth": 1.5},
            )

            # Annotate bars with value and Tukey letter
            y_max = max(mean_vals) + max(sems) if sems else max(mean_vals)
            y_pad = y_max * 0.04
            for i, (bar, ltr_key) in enumerate(zip(bars, [str(k) for k, _ in sorted_items])):
                letter = letters.get(ltr_key, "")
                bar_top = bar.get_height() + sems[i] + y_pad
                # Mean value inside/near top of bar
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() / 2,
                    _pub_fmt_large(mean_vals[i], 1),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )
                # Tukey letter above error bar
                if letter:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar_top,
                        letter,
                        ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="#333333",
                    )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0,
                               ha="right" if len(labels) > 4 else "center",
                               fontsize=10)
            ax.set_xlabel(predictor, fontsize=12, fontweight="bold", labelpad=8)
            ax.set_ylabel(response, fontsize=12, fontweight="bold", labelpad=8)
            ax.set_title(f"{response} by {predictor}", fontsize=14,
                         fontweight="bold", pad=16)
            ax.set_ylim(0, y_max * 1.18)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Legend / footnote
            note = ("Error bars = ±1 SE.  "
                    "Bars with the same letter are not significantly different "
                    "(Tukey HSD, p > 0.05).")
            fig.text(0.5, -0.04, note, ha="center", fontsize=8,
                     color="#555555", style="italic", wrap=True)

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode()
            plt.close()
            return encoded
        except Exception as exc:
            logger.warning("Publication bar chart generation failed: %s", exc)
            return ""

    def descriptive_statistics(self, df: pd.DataFrame, response: str,
                              predictors: List[str]) -> Dict:
        """Generate comprehensive descriptive statistics"""
        stats_dict = {}
        
        # Overall statistics
        stats_dict['overall'] = {
            'n': int(len(df)),
            'mean': float(df[response].mean()),
            'std': float(df[response].std()),
            'sem': float(df[response].sem()),
            'cv': float((df[response].std() / df[response].mean()) * 100 if df[response].mean() != 0 else 0),
            'min': float(df[response].min()),
            'max': float(df[response].max()),
            'range': float(df[response].max() - df[response].min()),
            'q1': float(df[response].quantile(0.25)),
            'median': float(df[response].median()),
            'q3': float(df[response].quantile(0.75)),
            'iqr': float(df[response].quantile(0.75) - df[response].quantile(0.25))
        }
        
        # Statistics by each predictor
        for predictor in predictors:
            grouped = df.groupby(predictor)[response].agg([
                'count', 'mean', 'std', 'sem', 'min', 'max'
            ]).round(3)
            
            # Add CV
            grouped['cv'] = (grouped['std'] / grouped['mean'] * 100).round(1)
            
            # Convert to nested dict
            stats_dict[predictor] = {}
            for idx in grouped.index:
                stats_dict[predictor][str(idx)] = {
                    col: float(val) if isinstance(val, (int, float)) else val
                    for col, val in grouped.loc[idx].items()
                }
        
        return stats_dict
    
    def run_anova(self, df: pd.DataFrame, response: str, 
                  predictors: List[str], blocks: List[str] = None) -> Dict:
        """Run complete ANOVA analysis"""
        logger.info(f"Running ANOVA for response: {response}")
        
        try:
            # Build formula
            formula = self.build_formula(response, predictors, blocks)
            logger.debug(f"Model formula: {formula}")
            
            # Fit model
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate total sum of squares
            ss_total = anova_table['sum_sq'].sum()
            
            # Calculate effect sizes
            effect_sizes = self.calculate_effect_sizes(anova_table, ss_total)
            
            # Check assumptions (use first predictor as grouping variable)
            assumptions = self.check_assumptions(df, formula, predictors[0] if predictors else None)
            
            # Calculate means for all predictors
            means_dict = {}
            for predictor in predictors:
                means = df.groupby(predictor)[response].mean().to_dict()
                means_dict[predictor] = {str(k): float(v) for k, v in means.items()}
            
            # Tukey HSD for significant main effects
            letters_dict = {}
            for predictor in predictors:
                if predictor in anova_table.index:
                    p_value = anova_table.loc[predictor, 'PR(>F)']
                    if pd.notna(p_value) and p_value < self.config.alpha:
                        try:
                            tukey = pairwise_tukeyhsd(df[response], df[predictor], 
                                                     alpha=self.config.alpha)
                            letters = self.compact_letter_display(tukey)
                            letters_dict[predictor] = {str(k): str(v) for k, v in letters.items()}
                        except Exception as e:
                            logger.warning(f"Tukey HSD failed for {predictor}: {str(e)}")
                            letters_dict[predictor] = {}
            
            # Get descriptive statistics
            desc_stats = self.descriptive_statistics(df, response, predictors)
            
            # Prepare result
            result = {
                'formula': formula,
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'f_value': float(model.fvalue) if hasattr(model, 'fvalue') else None,
                'f_pvalue': float(model.f_pvalue) if hasattr(model, 'f_pvalue') else None,
                'anova': json.loads(anova_table.round(4).to_json()),
                'means': means_dict,
                'letters': letters_dict,
                'effect_sizes': {k: asdict(v) for k, v in effect_sizes.items()},
                'assumptions': {k: asdict(v) for k, v in assumptions.items()},
                'descriptive_stats': desc_stats
            }

            # Add publication-ready formatted tables
            result['publication_tables'] = self.build_publication_tables(
                result, response,
                primary_predictor=predictors[0] if predictors else None,
            )

            return result
            
        except Exception as e:
            logger.error(f"ANOVA failed for {response}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'formula': '',
                'anova': {},
                'means': {},
                'letters': {},
                'effect_sizes': {},
                'assumptions': {},
                'descriptive_stats': {}
            }
    
    def generate_plots(self, df: pd.DataFrame, response: str, 
                      predictor: str) -> Dict[str, str]:
        """Generate publication-quality plots"""
        plots = {}
        
        try:
            # Set style for publication
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            # Calculate statistics for plotting
            means = df.groupby(predictor)[response].agg(['mean', 'sem']).reset_index()
            
            # 1. Bar plot with error bars
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = np.arange(len(means))
            bars = ax.bar(x_pos, means['mean'], yerr=means['sem'], 
                         capsize=5, color='steelblue', edgecolor='black', 
                         alpha=0.8, ecolor='black', linewidth=1)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, means['mean'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + means['sem'].iloc[i],
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Customize plot
            ax.set_xlabel(predictor, fontsize=12, fontweight='bold')
            ax.set_ylabel(response, fontsize=12, fontweight='bold')
            ax.set_title(f'{response} by {predictor}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(means[predictor], rotation=45, ha='right')
            
            # Add grid for readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format=self.config.figure_format, 
                       dpi=self.config.figure_dpi, bbox_inches='tight')
            buf.seek(0)
            plots['bar'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 2. Box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create boxplot with custom colors
            bp = df.boxplot(column=response, by=predictor, ax=ax, grid=False,
                           patch_artist=True, return_type='dict')
            
            # Color boxes
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(plt.cm.Set3(i / len(bp['boxes'])))
                box.set_alpha(0.7)
            
            # Customize
            ax.set_xlabel(predictor, fontsize=12, fontweight='bold')
            ax.set_ylabel(response, fontsize=12, fontweight='bold')
            ax.set_title(f'{response} Distribution by {predictor}', fontsize=14, fontweight='bold', pad=20)
            plt.suptitle('')  # Remove automatic suptitle
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format=self.config.figure_format, 
                       dpi=self.config.figure_dpi, bbox_inches='tight')
            buf.seek(0)
            plots['box'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 3. Interaction plot (if multiple predictors)
            if len(df.select_dtypes(include=['category']).columns) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get second predictor
                other_predictors = [c for c in df.select_dtypes(include=['category']).columns 
                                  if c != predictor]
                
                if other_predictors:
                    second_pred = other_predictors[0]
                    
                    # Calculate means for interaction
                    interaction_means = df.groupby([predictor, second_pred])[response].mean().unstack()
                    
                    # Plot
                    interaction_means.plot(marker='o', linewidth=2, markersize=8, ax=ax)
                    
                    ax.set_xlabel(predictor, fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'Mean {response}', fontsize=12, fontweight='bold')
                    ax.set_title(f'Interaction Plot: {predictor} × {second_pred}', 
                               fontsize=14, fontweight='bold', pad=20)
                    ax.legend(title=second_pred, bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    
                    buf = BytesIO()
                    plt.savefig(buf, format=self.config.figure_format, 
                               dpi=self.config.figure_dpi, bbox_inches='tight')
                    buf.seek(0)
                    plots['interaction'] = base64.b64encode(buf.read()).decode()
                    plt.close()
            
            # 4. Residuals plot (diagnostic)
            try:
                formula = f"{response} ~ {predictor}"
                model = ols(formula, data=df).fit()
                residuals = model.resid
                fitted = model.fittedvalues
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(fitted, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
                ax.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
                ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
                ax.set_title('Residuals vs Fitted', fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                
                buf = BytesIO()
                plt.savefig(buf, format=self.config.figure_format, 
                           dpi=self.config.figure_dpi, bbox_inches='tight')
                buf.seek(0)
                plots['residuals'] = base64.b64encode(buf.read()).decode()
                plt.close()
                
            except Exception as e:
                logger.warning(f"Residual plot failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Plot generation failed: {str(e)}")
            plots['error'] = base64.b64encode(b"Plot generation failed").decode()
        
        return plots

# =========================
# AI INTERPRETER
# =========================

class AIInterpreter:
    """Generate plain English interpretations of statistical results"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.use_ai = api_key is not None
        logger.info(f"AIInterpreter initialized. AI mode: {'ON' if self.use_ai else 'OFF'}")
    
    def interpret(self, result: Dict, response: str, context: Dict = None) -> str:
        """Generate comprehensive interpretation"""
        
        if self.use_ai:
            return self._ai_interpretation(result, response, context)
        else:
            return self._template_interpretation(result, response)
    
    def _template_interpretation(self, result: Dict, response: str) -> str:
        """Enhanced template-based interpretation"""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"{response.upper()} ANALYSIS INTERPRETATION")
        lines.append(f"{'='*60}\n")
        
        # Check for errors
        if 'error' in result:
            lines.append(f"⚠️ ANALYSIS FAILED: {result['error']}")
            lines.append("\nPlease check your data format and try again.")
            return '\n'.join(lines)
        
        # 1. MODEL SUMMARY
        lines.append("📊 MODEL SUMMARY")
        lines.append("-" * 40)
        
        if result.get('r_squared'):
            lines.append(f"• R² = {result['r_squared']:.3f} (adjusted R² = {result['adj_r_squared']:.3f})")
            lines.append(f"  This means {result['r_squared']*100:.1f}% of the variation in {response} is explained by the model.")
        
        if result.get('f_value') and result.get('f_pvalue'):
            f_p = result['f_pvalue']
            sig_text = "significant" if f_p < 0.05 else "not significant"
            lines.append(f"• Overall model F-test: F = {result['f_value']:.2f}, p = {f_p:.4f} ({sig_text})")
        
        lines.append("")
        
        # 2. ANOVA RESULTS
        lines.append("📈 SIGNIFICANT EFFECTS")
        lines.append("-" * 40)
        
        anova = result.get('anova', {})
        sig_effects = []
        
        for effect, stats in anova.items():
            if effect != 'Residual' and isinstance(stats, dict):
                p_val = stats.get('PR(>F)', 1.0)
                f_val = stats.get('F', 0)
                
                if p_val < 0.05:
                    sig_effects.append({
                        'name': effect,
                        'p': p_val,
                        'f': f_val,
                        'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    })
        
        if sig_effects:
            for effect in sig_effects:
                lines.append(f"• {effect['name']}: F = {effect['f']:.2f}, p = {effect['p']:.4f} {effect['sig_level']}")
        else:
            lines.append("• No significant effects detected (p > 0.05)")
        
        lines.append("")
        
        # 3. TREATMENT COMPARISONS
        lines.append("🏆 TREATMENT RANKINGS")
        lines.append("-" * 40)
        
        means_dict = result.get('means', {})
        letters_dict = result.get('letters', {})
        
        for predictor, means in means_dict.items():
            if means:
                lines.append(f"\n{predictor}:")
                
                # Sort means in descending order
                sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
                
                for i, (trt, mean) in enumerate(sorted_means[:5]):  # Show top 5
                    letter = letters_dict.get(predictor, {}).get(trt, '')
                    lines.append(f"  {i+1}. {trt}: {mean:.2f} {letter}")
                
                if len(sorted_means) > 5:
                    lines.append(f"  ... and {len(sorted_means)-5} more treatments")
        
        lines.append("")
        
        # 4. BEST TREATMENT ANALYSIS
        lines.append("🎯 OPTIMAL TREATMENT IDENTIFICATION")
        lines.append("-" * 40)
        
        for predictor, means in means_dict.items():
            if means:
                best_trt = max(means.items(), key=lambda x: x[1])
                worst_trt = min(means.items(), key=lambda x: x[1])
                
                # Calculate differences
                abs_diff = best_trt[1] - worst_trt[1]
                rel_diff = (abs_diff / worst_trt[1]) * 100 if worst_trt[1] != 0 else float('inf')
                
                lines.append(f"\n{predictor}:")
                lines.append(f"  • Best: {best_trt[0]} ({best_trt[1]:.2f})")
                lines.append(f"  • Improvement: {rel_diff:.1f}% higher than worst treatment ({worst_trt[0]}: {worst_trt[1]:.2f})")
                
                # Check for statistical grouping
                if predictor in letters_dict and best_trt[0] in letters_dict[predictor]:
                    best_letter = letters_dict[predictor][best_trt[0]]
                    
                    # Find other treatments in same group
            same_group = [trt for trt, ltr in letters_dict[predictor].items() 
                         if ltr == best_letter and trt != best_trt[0]]
            
            if same_group:
                lines.append(f"  • Statistically similar to: {', '.join(same_group[:3])}")
                if len(same_group) > 3:
                    lines.append(f"    and {len(same_group)-3} others")
        
        lines.append("")
        
        # 5. EFFECT SIZES
        lines.append("📐 EFFECT SIZE INTERPRETATION")
        lines.append("-" * 40)
        
        effect_sizes = result.get('effect_sizes', {})
        for effect, es in effect_sizes.items():
            if effect != 'Residual':
                lines.append(f"• {effect}: η² = {es.get('eta_squared', 0):.3f} ({es.get('interpretation', 'unknown')} effect)")
        
        lines.append("")
        
        # 6. ASSUMPTION CHECKS
        lines.append("🔍 ASSUMPTION VALIDATION")
        lines.append("-" * 40)
        
        assumptions = result.get('assumptions', {})
        for test_name, test_result in assumptions.items():
            if test_name != 'error' and isinstance(test_result, dict):
                status = "✓" if test_result.get('passed', False) else "⚠️"
                p_val = test_result.get('p_value', 'N/A')
                if p_val != 'N/A':
                    p_text = f"(p={p_val:.3f})" if isinstance(p_val, (int, float)) else ""
                else:
                    p_text = ""
                lines.append(f"{status} {test_name}: {test_result.get('message', '')} {p_text}")
        
        if 'error' in assumptions:
            lines.append(f"⚠️ Assumption testing encountered issues: {assumptions['error'].get('message', '')}")
        
        lines.append("")
        
        # 7. RECOMMENDATIONS
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if sig_effects:
            for predictor, means in means_dict.items():
                if means:
                    best_trt = max(means.items(), key=lambda x: x[1])
                    lines.append(f"• Use {best_trt[0]} to maximize {response} (based on {predictor})")
                    
                    # Economic consideration
                    sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_means) > 1:
                        second_best = sorted_means[1]
                        # Check if statistically similar
                        if (predictor in letters_dict and 
                            letters_dict[predictor].get(best_trt[0]) == 
                            letters_dict[predictor].get(second_best[0])):
                            lines.append(f"  Consider {second_best[0]} for cost-effectiveness (statistically similar)")
        else:
            lines.append(f"• No significant treatment differences detected")
            lines.append(f"• Consider other factors or experimental variables for {response} optimization")
            lines.append(f"• Review experimental design and data quality")
        
        lines.append(f"\n{'='*60}")
        
        return '\n'.join(lines)
    
    def _ai_interpretation(self, result: Dict, response: str, context: Dict = None) -> str:
        """Use Claude (Anthropic) to generate interpretation, falling back to template."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            # Build a concise prompt from the result
            anova = result.get("anova", {})
            means = result.get("means", {})
            letters = result.get("letters", {})
            r2 = result.get("r_squared")
            f_p = result.get("f_pvalue")

            prompt_lines = [
                f"You are a statistical consultant. Interpret the following ANOVA results for the trait '{response}' concisely (3-5 sentences, plain English, suitable for a journal methods section).",
                f"R² = {r2:.3f}" if r2 is not None else "",
                f"Overall F p-value = {f_p:.4f}" if f_p is not None else "",
            ]
            for effect, vals in anova.items():
                if isinstance(vals, dict) and effect != "Residual":
                    p = vals.get("PR(>F)")
                    f = vals.get("F")
                    if p is not None:
                        sig = "significant" if float(p) < 0.05 else "not significant"
                        prompt_lines.append(f"  {effect}: F={f:.2f}, p={p:.4f} ({sig})")
            for pred, m in means.items():
                top = sorted(m.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join(f"{k}={v:.2f}{(' '+letters.get(pred,{}).get(k,''))}" for k, v in top)
                prompt_lines.append(f"  Top {pred} means: {top_str}")

            prompt = "\n".join(l for l in prompt_lines if l)

            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.warning("AI interpretation failed (%s), using template fallback", e)
            return self._template_interpretation(result, response)

# =========================
# CACHE MANAGER
# =========================

class CacheManager:
    """Manage result caching"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        
    def get_cache_key(self, data_hash: str, config: AnalysisConfig) -> str:
        """Generate cache key"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(f"{data_hash}_{config_str}".encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve from cache"""
        # Check memory cache
        if key in self.memory_cache:
            logger.info(f"Cache hit (memory): {key}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Cache hit (disk): {key}")
                return data
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        return None
    
    def set(self, key: str, data: Dict, ttl: int = 3600):
        """Store in cache"""
        # Memory cache
        self.memory_cache[key] = data
        
        # Disk cache
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Cached: {key}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self):
        """Clear all cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")

# =========================
# BACKEND PIPELINE
# =========================

class VivaSenseBackend:
    """Main backend pipeline"""
    
    def __init__(self, config: AnalysisConfig = None, ai_api_key: str = None):
        self.config = config or AnalysisConfig()
        self.analyzer = StatisticalAnalyzer(self.config)
        self.interpreter = AIInterpreter(ai_api_key)
        self.cache = CacheManager()
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("VivaSenseBackend initialized")
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for caching"""
        df_str = df.to_json().encode()
        return hashlib.md5(df_str).hexdigest()
    
    def process_dataframe(self, df: pd.DataFrame, filename: str = None) -> Dict:
        """Process pandas DataFrame and return results"""
        logger.info(f"Processing dataframe with {len(df)} rows, {len(df.columns)} columns")
        
        # Validate data
        validation = self.analyzer.validate_data(df)
        if validation["errors"]:
            return {
                "status": "error",
                "errors": validation["errors"],
                "warnings": validation["warnings"]
            }
        
        # Check cache
        data_hash = self.compute_data_hash(df)
        cache_key = self.cache.get_cache_key(data_hash, self.config)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached result")
            return cached_result
        
        # Detect variable types
        categorical, continuous = self.analyzer.detect_variable_types(df)
        
        if not categorical:
            return {
                "status": "error",
                "errors": ["No categorical predictors found. Please ensure your data includes treatment/group columns."],
                "warnings": validation["warnings"]
            }
        
        # Detect blocks
        blocks = [col for col in categorical if any(term in col.lower() 
                  for term in ['block', 'rep', 'replicate', 'batch'])]
        
        logger.info(f"Detected blocks: {blocks}")
        
        # Process each continuous variable
        results = {
            "status": "success",
            "metadata": {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "categorical_vars": categorical,
                "continuous_vars": continuous,
                "blocks": blocks,
                "config": self.config.to_dict(),
                "analysis_id": str(uuid.uuid4())
            },
            "warnings": validation["warnings"],
            "traits": {}
        }
        
        for trait in continuous:
            logger.info(f"Analyzing trait: {trait}")
            
            try:
                # Run ANOVA
                anova_result = self.analyzer.run_anova(df, trait, categorical, blocks)
                
                if 'error' in anova_result:
                    results["traits"][trait] = {
                        "error": anova_result['error'],
                        "status": "failed"
                    }
                    continue
                
                # Generate plots (using first categorical predictor)
                plots = self.analyzer.generate_plots(df, trait, categorical[0])
                # Publication bar chart with Tukey letter annotations
                try:
                    _letters = anova_result.get("letters", {}).get(categorical[0], {})
                    _means   = anova_result.get("means",   {}).get(categorical[0], {})
                    if _means:
                        plots["publication_bar"] = self.analyzer.generate_publication_bar_chart(
                            df, trait, categorical[0], _letters, _means
                        )
                except Exception as _pe:
                    logger.warning("Publication bar chart failed for %s: %s", trait, _pe)

                # Generate interpretation
                context = {
                    "trait": trait,
                    "predictors": categorical,
                    "n_obs": len(df)
                }
                interpretation = self.interpreter.interpret(anova_result, trait, context)
                
                # Store results
                results["traits"][trait] = {
                    "status": "success",
                    "statistical_results": anova_result,
                    "plots": plots,
                    "interpretation": interpretation
                }
                
            except Exception as e:
                logger.error(f"Analysis failed for {trait}: {str(e)}")
                logger.error(traceback.format_exc())
                results["traits"][trait] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Cache results
        self.cache.set(cache_key, results)
        
        # Save to file
        self.save_results(results, filename)
        
        return results
    
    def process_file(self, file_path: Union[str, Path]) -> Dict:
        """Process file from path"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}
        
        # Read file based on extension
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return {"status": "error", "error": f"Unsupported file format: {file_path.suffix}"}
            
            return self.process_dataframe(df, file_path.name)
            
        except Exception as e:
            logger.error(f"File read failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem if filename else "analysis"
        result_file = self.results_dir / f"{base_name}_{timestamp}.json"
        
        # Convert non-serializable objects
        def json_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(result_file, 'w') as f:
            json.dump(results, f, default=json_serializer, indent=2)
        
        logger.info(f"Results saved to {result_file}")
        return result_file

# =========================
# FASTAPI APPLICATION
# =========================

# Create FastAPI app
app = FastAPI(
    title="VivaSense Statistical Engine",
    description="Journal-grade ANOVA analysis platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backend — pick up AI key from environment if present
_ai_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AI_KEY")
backend = VivaSenseBackend(ai_api_key=_ai_api_key)

# Genetics module
try:
    from genetics import genetics_router
    GENETICS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Genetics module import failed: {e}")
    GENETICS_AVAILABLE = False
    genetics_router = None

if GENETICS_AVAILABLE and genetics_router is not None:
    app.include_router(genetics_router)

# Store background tasks
background_tasks_store = {}

# =========================
# API ENDPOINTS
# =========================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VivaSense Statistical Engine",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/analyze/")
async def analyze_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and analyze experimental data file
    
    - Supports CSV and Excel files
    - Automatically detects variable types
    - Performs comprehensive ANOVA analysis
    - Generates publication-ready plots
    - Provides plain English interpretation
    """
    logger.info(f"Received file: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files (.csv, .xlsx, .xls) are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse based on file extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))
        
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Process data
        results = backend.process_dataframe(df, file.filename)
        
        # Schedule cleanup if background tasks available
        if background_tasks:
            analysis_id = results.get("metadata", {}).get("analysis_id")
            if analysis_id:
                background_tasks_store[analysis_id] = results
                # Schedule cleanup after 1 hour
                background_tasks.add_task(cleanup_analysis, analysis_id, delay=3600)
        
        return results
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/json/")
async def analyze_json(data: Dict):
    """
    Analyze JSON data
    
    Expects: {
        "data": [{"col1": val1, "col2": val2, ...}],
        "config": {...} (optional)
    }
    """
    try:
        df = pd.DataFrame(data.get("data", []))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Update config if provided
        if "config" in data:
            config_dict = data["config"]
            config = AnalysisConfig(**config_dict)
            temp_backend = VivaSenseBackend(config)
            results = temp_backend.process_dataframe(df, "json_upload")
        else:
            results = backend.process_dataframe(df, "json_upload")
        
        return results
        
    except Exception as e:
        logger.error(f"JSON analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Retrieve analysis results by ID"""
    if analysis_id in background_tasks_store:
        return background_tasks_store[analysis_id]
    
    # Check if results file exists
    result_files = list(backend.results_dir.glob(f"*{analysis_id}*.json"))
    if result_files:
        with open(result_files[0], 'r') as f:
            return json.load(f)
    
    raise HTTPException(status_code=404, detail="Analysis results not found")

@app.delete("/cache/")
async def clear_cache():
    """Clear analysis cache"""
    backend.cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.get("/config/")
async def get_config():
    """Get current configuration"""
    return backend.config.to_dict()

@app.post("/config/")
async def update_config(config: Dict):
    """Update analysis configuration"""
    try:
        backend.config = AnalysisConfig(**config)
        backend.analyzer.config = backend.config
        return {"status": "success", "config": backend.config.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================
# ANOVA-SPECIFIC ENDPOINTS
# =========================

def _parse_anova_request(data: Dict) -> Tuple[pd.DataFrame, str, float]:
    """Validate and parse common fields from an ANOVA request body."""
    rows = data.get("data")
    if not rows:
        raise HTTPException(status_code=400, detail="'data' field is required and must not be empty")
    df = pd.DataFrame(rows)
    response = data.get("response")
    if not response:
        raise HTTPException(status_code=400, detail="'response' field is required")
    if isinstance(response, list):
        response = response[0]
    if response not in df.columns:
        raise HTTPException(status_code=400, detail=f"Response column '{response}' not found in data")
    df[response] = pd.to_numeric(df[response], errors='coerce')
    alpha = float(data.get("alpha", 0.05))
    return df, response, alpha


def _anova_with_plots(df: pd.DataFrame, response: str, predictors: List[str],
                      blocks: List[str] = None, alpha: float = 0.05) -> Dict:
    """Run ANOVA via StatisticalAnalyzer and attach plots + interpretation."""
    cfg = AnalysisConfig(alpha=alpha)
    analyzer = StatisticalAnalyzer(cfg)
    result = analyzer.run_anova(df, response, predictors, blocks)
    try:
        result["plots"] = analyzer.generate_plots(df, response, predictors[0])
    except Exception:
        result["plots"] = {}
    # Publication-quality bar chart with Tukey letter annotations
    try:
        letters = result.get("letters", {}).get(predictors[0], {})
        means   = result.get("means",   {}).get(predictors[0], {})
        if means:
            pub_bar = analyzer.generate_publication_bar_chart(
                df, response, predictors[0], letters, means
            )
            result["plots"]["publication_bar"] = pub_bar
    except Exception as _e:
        logger.warning("Publication bar chart failed: %s", _e)
    result["interpretation"] = backend.interpreter.interpret(result, response)
    return result


@app.post("/analyze/descriptive")
async def analyze_descriptive(data: Dict):
    """Descriptive statistics for one or two grouping factors."""
    df, response, alpha = _parse_anova_request(data)
    predictors = [
        col for key in ("treatment", "treatment_a", "treatment_b", "block")
        if (col := data.get(key)) and col in df.columns
    ]
    if not predictors:
        raise HTTPException(status_code=400,
                            detail="At least one of 'treatment', 'treatment_a', 'treatment_b', 'block' is required")
    analyzer = StatisticalAnalyzer(AnalysisConfig(alpha=alpha))
    desc = analyzer.descriptive_statistics(df, response, predictors)
    return {"status": "success", "design": "descriptive", "response": response,
            "n": len(df), "descriptive_stats": desc}


@app.post("/analyze/anova/oneway")
async def analyze_oneway(data: Dict):
    """One-way ANOVA — completely randomized design (CRD)."""
    df, response, alpha = _parse_anova_request(data)
    treatment = data.get("treatment")
    if not treatment or treatment not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment' column is required")
    result = _anova_with_plots(df, response, [treatment], alpha=alpha)
    return {"status": "success", "design": "oneway", "response": response,
            "treatment": treatment, **result}


@app.post("/analyze/anova/oneway_rcbd")
async def analyze_oneway_rcbd(data: Dict):
    """One-way ANOVA in a Randomized Complete Block Design (RCBD)."""
    df, response, alpha = _parse_anova_request(data)
    treatment = data.get("treatment")
    block = data.get("block")
    if not treatment or treatment not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment' column is required")
    if not block or block not in df.columns:
        raise HTTPException(status_code=400, detail="'block' column is required")
    result = _anova_with_plots(df, response, [treatment], blocks=[block], alpha=alpha)
    return {"status": "success", "design": "oneway_rcbd", "response": response,
            "treatment": treatment, "block": block, **result}


@app.post("/analyze/anova/twoway")
async def analyze_twoway(data: Dict):
    """Two-way factorial ANOVA with interaction (CRD)."""
    df, response, alpha = _parse_anova_request(data)
    treatment_a = data.get("treatment_a")
    treatment_b = data.get("treatment_b")
    if not treatment_a or treatment_a not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment_a' column is required")
    if not treatment_b or treatment_b not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment_b' column is required")
    result = _anova_with_plots(df, response, [treatment_a, treatment_b], alpha=alpha)
    return {"status": "success", "design": "twoway", "response": response,
            "treatment_a": treatment_a, "treatment_b": treatment_b, **result}


@app.post("/analyze/anova/rcbd_factorial")
async def analyze_rcbd_factorial(data: Dict):
    """Factorial ANOVA in an RCBD (two treatment factors + blocking)."""
    df, response, alpha = _parse_anova_request(data)
    treatment_a = data.get("treatment_a")
    treatment_b = data.get("treatment_b")
    block = data.get("block")
    if not treatment_a or treatment_a not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment_a' column is required")
    if not treatment_b or treatment_b not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment_b' column is required")
    if not block or block not in df.columns:
        raise HTTPException(status_code=400, detail="'block' column is required")
    result = _anova_with_plots(df, response, [treatment_a, treatment_b], blocks=[block], alpha=alpha)
    return {"status": "success", "design": "rcbd_factorial", "response": response,
            "treatment_a": treatment_a, "treatment_b": treatment_b, "block": block, **result}


@app.post("/analyze/anova/splitplot")
async def analyze_splitplot(data: Dict):
    """Split-plot ANOVA — whole-plot and sub-plot factors with blocking."""
    df, response, alpha = _parse_anova_request(data)
    whole_plot = data.get("whole_plot")
    sub_plot   = data.get("sub_plot")
    block      = data.get("block")
    if not whole_plot or whole_plot not in df.columns:
        raise HTTPException(status_code=400, detail="'whole_plot' column is required")
    if not sub_plot or sub_plot not in df.columns:
        raise HTTPException(status_code=400, detail="'sub_plot' column is required")
    if not block or block not in df.columns:
        raise HTTPException(status_code=400, detail="'block' column is required")
    try:
        cfg = AnalysisConfig(alpha=alpha)
        analyzer = StatisticalAnalyzer(cfg)
        # Include block:whole_plot as the whole-plot error stratum
        formula = (f"{response} ~ {block} + {whole_plot} + {block}:{whole_plot}"
                   f" + {sub_plot} + {whole_plot}:{sub_plot}")
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        ss_total = anova_table["sum_sq"].sum()
        effect_sizes = analyzer.calculate_effect_sizes(anova_table, ss_total)
        assumptions = analyzer.check_assumptions(df, formula, whole_plot)
        desc_stats = analyzer.descriptive_statistics(df, response, [whole_plot, sub_plot])
        means = {
            whole_plot: {str(k): float(v) for k, v in df.groupby(whole_plot)[response].mean().items()},
            sub_plot:   {str(k): float(v) for k, v in df.groupby(sub_plot)[response].mean().items()},
        }
        try:
            plots = analyzer.generate_plots(df, response, whole_plot)
        except Exception:
            plots = {}
        sp_result = {
            "formula": formula,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_value": float(model.fvalue) if hasattr(model, "fvalue") else None,
            "f_pvalue": float(model.f_pvalue) if hasattr(model, "f_pvalue") else None,
            "anova": json.loads(anova_table.round(4).to_json()),
            "means": means,
            "letters": {},
            "effect_sizes": {k: asdict(v) for k, v in effect_sizes.items()},
            "assumptions": {k: asdict(v) for k, v in assumptions.items()},
            "descriptive_stats": desc_stats,
        }
        sp_result["publication_tables"] = analyzer.build_publication_tables(
            sp_result, response, primary_predictor=whole_plot
        )
        return {
            "status": "success", "design": "splitplot", "response": response,
            "whole_plot": whole_plot, "sub_plot": sub_plot, "block": block,
            "plots": plots,
            **sp_result,
        }
    except Exception as e:
        logger.error("Split-plot ANOVA failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/nonparametric/kruskal")
async def analyze_kruskal(data: Dict):
    """Kruskal-Wallis H test with pairwise Mann-Whitney post-hoc (Bonferroni)."""
    df, response, alpha = _parse_anova_request(data)
    treatment = data.get("treatment")
    if not treatment or treatment not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment' column is required")
    try:
        groups = [g[response].dropna().values for _, g in df.groupby(treatment)]
        h_stat, p_value = stats.kruskal(*groups)
        group_means   = {str(k): float(v) for k, v in df.groupby(treatment)[response].mean().items()}
        group_medians = {str(k): float(v) for k, v in df.groupby(treatment)[response].median().items()}
        group_counts  = {str(k): int(v)   for k, v in df.groupby(treatment)[response].count().items()}
        # Pairwise Mann-Whitney U with Bonferroni correction
        group_names = sorted(df[treatment].unique().tolist(), key=str)
        n_pairs = max(len(group_names) * (len(group_names) - 1) // 2, 1)
        posthoc: Dict[str, Any] = {}
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1 = df[df[treatment] == group_names[i]][response].dropna().values
                g2 = df[df[treatment] == group_names[j]][response].dropna().values
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pair = f"{group_names[i]} vs {group_names[j]}"
                posthoc[pair] = {
                    "U": float(u_stat),
                    "p_value": float(p_val),
                    "p_adjusted": float(min(p_val * n_pairs, 1.0)),
                    "significant": bool(p_val * n_pairs < alpha),
                }
        analyzer = StatisticalAnalyzer(AnalysisConfig(alpha=alpha))
        try:
            plots = analyzer.generate_plots(df, response, treatment)
        except Exception:
            plots = {}
        return {
            "status": "success", "design": "kruskal", "response": response,
            "treatment": treatment, "n": len(df),
            "H_statistic": float(h_stat), "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "group_means": group_means, "group_medians": group_medians,
            "group_counts": group_counts, "posthoc": posthoc, "plots": plots,
        }
    except Exception as e:
        logger.error("Kruskal-Wallis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/nonparametric/friedman")
async def analyze_friedman(data: Dict):
    """Friedman test with pairwise Wilcoxon signed-rank post-hoc (Bonferroni)."""
    df, response, alpha = _parse_anova_request(data)
    treatment = data.get("treatment")
    block = data.get("block")
    if not treatment or treatment not in df.columns:
        raise HTTPException(status_code=400, detail="'treatment' column is required")
    if not block or block not in df.columns:
        raise HTTPException(status_code=400, detail="'block' column is required")
    try:
        pivot = df.pivot_table(index=block, columns=treatment, values=response, aggfunc="mean")
        col_data = [pivot[col].dropna().values for col in pivot.columns]
        chi2_stat, p_value = stats.friedmanchisquare(*col_data)
        group_means   = {str(k): float(v) for k, v in df.groupby(treatment)[response].mean().items()}
        group_medians = {str(k): float(v) for k, v in df.groupby(treatment)[response].median().items()}
        # Pairwise Wilcoxon signed-rank with Bonferroni correction
        treatment_names = list(pivot.columns)
        n_pairs = max(len(treatment_names) * (len(treatment_names) - 1) // 2, 1)
        posthoc: Dict[str, Any] = {}
        for i in range(len(treatment_names)):
            for j in range(i + 1, len(treatment_names)):
                g1 = pivot[treatment_names[i]].dropna().values
                g2 = pivot[treatment_names[j]].dropna().values
                n = min(len(g1), len(g2))
                w_stat, p_val = stats.wilcoxon(g1[:n], g2[:n])
                pair = f"{treatment_names[i]} vs {treatment_names[j]}"
                posthoc[pair] = {
                    "W": float(w_stat),
                    "p_value": float(p_val),
                    "p_adjusted": float(min(p_val * n_pairs, 1.0)),
                    "significant": bool(p_val * n_pairs < alpha),
                }
        analyzer = StatisticalAnalyzer(AnalysisConfig(alpha=alpha))
        try:
            plots = analyzer.generate_plots(df, response, treatment)
        except Exception:
            plots = {}
        return {
            "status": "success", "design": "friedman", "response": response,
            "treatment": treatment, "block": block, "n": len(df),
            "chi2_statistic": float(chi2_stat), "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "group_means": group_means, "group_medians": group_medians,
            "posthoc": posthoc, "plots": plots,
        }
    except Exception as e:
        logger.error("Friedman test failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# BACKGROUND TASKS
# =========================

async def cleanup_analysis(analysis_id: str, delay: int = 3600):
    """Clean up analysis results after delay"""
    import asyncio
    await asyncio.sleep(delay)
    if analysis_id in background_tasks_store:
        del background_tasks_store[analysis_id]
        logger.info(f"Cleaned up analysis: {analysis_id}")

# =========================
# MAIN ENTRY POINT
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VivaSense Statistical Backend')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--input-file', help='Process a file and exit (non-server mode)')
    parser.add_argument('--output-dir', default='./output', help='Output directory for file mode')
    parser.add_argument('--ai-key', help='API key for AI interpretation')
    
    args = parser.parse_args()
    
    if args.input_file:
        # File processing mode
        print(f"Processing file: {args.input_file}")
        backend = VivaSenseBackend(ai_api_key=args.ai_key)
        results = backend.process_file(args.input_file)
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        result_file = backend.save_results(results, Path(args.input_file).name)
        print(f"\n✅ Analysis complete!")
        print(f"📊 Results saved to: {result_file}")
        
        # Print summary
        print("\n📋 Summary:")
        if results.get("status") == "success":
            for trait, trait_result in results.get("traits", {}).items():
                if trait_result.get("status") == "success":
                    print(f"  • {trait}: ✓ Analyzed")
                else:
                    print(f"  • {trait}: ✗ Failed - {trait_result.get('error', 'Unknown error')}")
        
        if results.get("warnings"):
            print("\n⚠️ Warnings:")
            for warning in results["warnings"]:
                print(f"  • {warning}")
    
    else:
        # Server mode
        print(f"🚀 Starting VivaSense Statistical Engine v2.0.0")
        print(f"📊 Server running at http://{args.host}:{args.port}")
        print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔍 Health check: http://{args.host}:{args.port}/health")
        print("\nPress Ctrl+C to stop")
        
        uvicorn.run(
            app,
            host=args.host,
            port=int(os.environ.get("PORT", args.port)),
            reload=args.reload
        )
