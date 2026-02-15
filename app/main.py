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
                ms_error = anova_table.loc['Residual', 'mean_sq'] if 'Residual' in anova_table.index else 0
                
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
                    if p_value < self.config.alpha:
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
                'anova': anova_table.round(4).to_dict(),
                'means': means_dict,
                'letters': letters_dict,
                'effect_sizes': {k: asdict(v) for k, v in effect_sizes.items()},
                'assumptions': {k: asdict(v) for k, v in assumptions.items()},
                'descriptive_stats': desc_stats
            }
            
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
        """Use AI to generate interpretation (placeholder for API integration)"""
        # This would integrate with OpenAI, Claude, etc.
        # For now, return template version with note
        template = self._template_interpretation(result, response)
        return template + "\n\n[Note: AI-powered interpretation would provide enhanced insights here]"

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

# Initialize backend
backend = VivaSenseBackend()

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
            "main:app",
            host=args.host,
            port=int(os.environ.get("PORT", 8000)),
            reload=args.reload
        )
