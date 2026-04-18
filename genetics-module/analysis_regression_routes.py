"""
VivaSense – General Statistical Regression Analysis Module

POST /analysis/regression

Provides general-purpose linear regression capabilities, independent of 
agronomic or genetics-specific terminology. Suitable for Relationship Analysis.
"""

import base64
import io
import logging
import math
from typing import List, Optional

import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import dataset_cache


def classify_strength(r_squared: float, p_value: float) -> str:
    """
    REGRESSION STRENGTH CLASSIFICATION THRESHOLDS (LOCKED)
    These thresholds are intentionally explicit to prevent drift.
    Change ONLY if a documented change request exists.

    R² Classification:
      R² < 0.25        -> "weak"
      R² 0.25 – 0.49   -> "moderate"
      R² >= 0.50       -> "strong"

    Significance Override:
      p >= 0.05        -> "negligible_or_unreliable"  (regardless of R²)

    Decision Logic:
      if p >= 0.05:        return "negligible_or_unreliable"
      elif R² < 0.25:      return "weak"
      elif R² < 0.50:      return "moderate"
      else:                return "strong"

    Last reviewed: 2026-04-18
    Author: VivaSense Regression Module
    """
    if p_value >= 0.05:
        return "negligible_or_unreliable"
    if r_squared < 0.25:
        return "weak"
    if r_squared < 0.50:
        return "moderate"
    return "strong"


def _read_file_for_regression(content: bytes, file_type: str) -> pd.DataFrame:
    """
    Minimal file reader for regression — no genetics-specific row-floor.
    The regression endpoint enforces its own n >= 3 check after loading.
    """
    try:
        if file_type == "csv":
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}") from exc
    if df.empty:
        raise ValueError("File is empty or has no data rows")
    df.columns = [str(c).strip() for c in df.columns]
    return df

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


class RegressionRequest(BaseModel):
    """Request schema for regression analysis."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    x_variable: str = Field(..., description="Independent variable column name")
    y_variable: str = Field(..., description="Dependent variable column name")
    model_type: str = Field(default="linear", description="Type of regression model")

    model_config = {'protected_namespaces': ()}


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class PlotData(BaseModel):
    x: List[float]
    y: List[float]
    fitted_y: List[float]


class RegressionResponse(BaseModel):
    """Response schema for regression analysis."""
    status: str
    analysis_type: str = "regression"
    model_type: str = "linear"
    x_variable: str
    y_variable: str
    n: int
    equation: str
    intercept: float
    slope: float
    r_squared: float
    adjusted_r_squared: float
    correlation_coefficient: float
    p_value_slope: float
    standard_error_slope: float
    confidence_interval_slope: ConfidenceInterval
    direction: str
    strength_class: str
    significance_class: str
    plain_language_effect: str
    summary_interpretation: str
    reliability_flags: List[str]
    warnings: List[str]
    plot_data: PlotData

    model_config = {'protected_namespaces': ()}


@router.post(
    "/analysis/regression",
    response_model=RegressionResponse,
    summary="Run linear regression between two variables",
)
async def analysis_regression(request: RegressionRequest):
    """
    Executes simple linear regression predicting Y from X.
    Returns statistical metrics, structured interpretation, reliability flags,
    and data points required for frontend scatter/fitted line plotting.
    """
    if request.model_type != "linear":
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model_type: '{request.model_type}'. Currently only 'linear' is supported."
        )

    if request.x_variable == request.y_variable:
        raise HTTPException(
            status_code=400, 
            detail="x_variable and y_variable must be different columns."
        )

    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset token '{request.dataset_token}' not found."
        )

    try:
        if "dataframe" in ctx and isinstance(ctx["dataframe"], pd.DataFrame):
            df = ctx["dataframe"].copy()
        else:
            file_bytes = base64.b64decode(ctx["base64_content"])
            df = _read_file_for_regression(file_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc

    # Variable existence validation
    missing = [c for c in [request.x_variable, request.y_variable] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Variables not found in dataset: {missing}")

    # Numeric validation
    if not pd.api.types.is_numeric_dtype(df[request.x_variable]):
        raise HTTPException(status_code=400, detail=f"x_variable '{request.x_variable}' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df[request.y_variable]):
        raise HTTPException(status_code=400, detail=f"y_variable '{request.y_variable}' must be numeric.")

    # Pairwise complete observations
    df_clean = df[[request.x_variable, request.y_variable]].dropna()
    n = len(df_clean)

    if n < 3:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient complete paired observations (n={n}). Minimum 3 required."
        )

    df_clean = df_clean.sort_values(by=request.x_variable)

    x = df_clean[request.x_variable].astype(float)
    y = df_clean[request.y_variable].astype(float)

    if x.var() == 0:
        raise HTTPException(status_code=400, detail=f"x_variable '{request.x_variable}' has zero variance.")
    if y.var() == 0:
        raise HTTPException(status_code=400, detail=f"y_variable '{request.y_variable}' has zero variance.")

    # Fit OLS model using statsmodels
    X_with_const = sm.add_constant(x)
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    # Extract parameters by NAME, not position, so intercept/slope are never swapped.
    # sm.add_constant(x) names the slope column after x's Series name (request.x_variable)
    # and the intercept "const".  Named access is the single source of truth for all
    # coefficient-level statistics — this is the only p-value source (Bug 3 fix).
    x_name = request.x_variable
    intercept     = float(results.params.get("const", 0.0))
    slope         = float(results.params[x_name])
    p_value_slope = float(results.pvalues[x_name])   # slope p-value ONLY — no F-test, no intercept p-value
    se_slope      = float(results.bse[x_name])

    conf_int = results.conf_int(alpha=0.05)
    ci_lower = float(conf_int.loc[x_name, 0])
    ci_upper = float(conf_int.loc[x_name, 1])

    r_squared = float(results.rsquared)
    # Bug 2 fix: derive r from OLS R² so both metrics share the same model.
    # r = sign(slope) × sqrt(R²)  — algebraically exact for simple OLS.
    r_coef = math.copysign(math.sqrt(max(0.0, r_squared)), slope)

    # Build interpretation fields — thresholds enforced via classify_strength()
    strength  = classify_strength(r_squared, p_value_slope)
    is_sig    = p_value_slope < 0.05
    sig_class = "significant" if is_sig else "not_significant"

    if not is_sig:
        direction = "no_clear_relationship"
        plain_effect = (
            "Because the fitted linear relationship is not statistically reliable, "
            "the estimated change per unit increase should not be interpreted as meaningful."
        )
        summary_interpretation = (
            f"{request.x_variable} does not appear to be a useful linear predictor of "
            f"{request.y_variable} in this dataset. Other factors may be more important "
            "in explaining variation in the response."
        )
    else:
        direction = "positive" if slope > 0 else "negative" if slope < 0 else "none"
        if slope > 0:
            plain_effect = (
                f"For every 1-unit increase in {request.x_variable}, "
                f"{request.y_variable} increases by {slope:.4f} units on average."
            )
        elif slope < 0:
            plain_effect = (
                f"For every 1-unit increase in {request.x_variable}, "
                f"{request.y_variable} decreases by {abs(slope):.4f} units on average."
            )
        else:
            plain_effect = (
                f"Changes in {request.x_variable} are not associated with "
                f"changes in {request.y_variable}."
            )
        # Summary text mirrors the R² threshold boundary in classify_strength()
        if r_squared < 0.25:
            summary_interpretation = (
                "A statistically detectable relationship is present, but the model "
                "explains only a limited proportion of the variation in the response."
            )
        else:
            summary_interpretation = (
                "A statistically reliable linear relationship is present, and the model "
                "captures a meaningful proportion of the variation in the response."
            )

    # Publication-ready equation: "{outcome} = {intercept} + {slope} × {predictor}"
    # U+00D7 (×) is safe in UTF-8 JSON responses and python-docx (UTF-8 XML).
    # The earlier removal was only needed for the cp1252 Windows console demo.
    # Spaces around × prevent hyphenated variable names from being read as subtraction.
    equation = f"{request.y_variable} = {intercept:.6f} {slope:+.6f} \u00d7 {request.x_variable}"

    # Reliability checks
    warnings = []
    flags = []

    if n < 10:
        warnings.append("Small sample size — results may be unstable.")
        flags.append("small_sample")
    else:
        flags.append("sample_size_ok")

    if r_squared > 0.9 and n < 15:
        warnings.append("Very high model fit with small sample size — may be misleading.")
        flags.append("high_fit_small_n")

    if not is_sig:
        warnings.append("No statistically reliable linear relationship detected.")
        flags.append("non_significant_slope")

    X_with_const_sorted = sm.add_constant(x)
    fitted_values = results.predict(X_with_const_sorted)

    return RegressionResponse(
        status="success",
        analysis_type="regression",
        model_type="linear",
        x_variable=request.x_variable,
        y_variable=request.y_variable,
        n=n,
        equation=equation,
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        adjusted_r_squared=float(results.rsquared_adj),
        correlation_coefficient=r_coef,
        p_value_slope=p_value_slope,
        standard_error_slope=se_slope,
        confidence_interval_slope=ConfidenceInterval(lower=ci_lower, upper=ci_upper),
        direction=direction,
        strength_class=strength,
        significance_class=sig_class,
        plain_language_effect=plain_effect,
        summary_interpretation=summary_interpretation,
        reliability_flags=flags,
        warnings=warnings,
        plot_data=PlotData(x=x.tolist(), y=y.tolist(), fitted_y=fitted_values.tolist())
    )