from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import traceback
import json

app = FastAPI(title="VivaSense API")

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Helpers
# -----------------------
def safe_json_value(x):
    """Convert any value to JSON-safe format"""
    # Handle None, NaN, inf
    if x is None or pd.isna(x):
        return None
    if np.isinf(x):
        return None
    
    # Handle boolean (numpy and native)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    
    # Handle numeric types
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    
    # Handle strings
    if isinstance(x, str):
        return x
    
    # Fallback: convert to string
    return str(x)

def dataframe_to_records(df):
    """Convert DataFrame to JSON-safe list of dicts"""
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            record[str(col)] = safe_json_value(val)
        records.append(record)
    return records

# -----------------------
# API Endpoint
# -----------------------
@app.post("/vivasense/run")
async def run_vivasense(
    file: UploadFile,
    outcome: str = Form(...),
    predictors: str = Form(...),
    user_level: str = Form(None)
):
    try:
        # Read the Excel file
        df = pd.read_excel(file.file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Parse predictors
        predictors_list = [p.strip() for p in predictors.split(",")]
        
        # VALIDATION: Check columns exist
        if outcome not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Outcome column '{outcome}' not found. Available: {list(df.columns)}"
            )
        
        missing_preds = [p for p in predictors_list if p not in df.columns]
        if missing_preds:
            raise HTTPException(
                status_code=400,
                detail=f"Predictor(s) {missing_preds} not found. Available: {list(df.columns)}"
            )
        
        # Check outcome is numeric
        if not pd.api.types.is_numeric_dtype(df[outcome]):
            raise HTTPException(
                status_code=400,
                detail=f"Outcome '{outcome}' must be numeric"
            )
        
        # Drop missing values
        required_cols = [outcome] + predictors_list
        df_clean = df[required_cols].dropna()
        
        if len(df_clean) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data. Need at least 3 rows, got {len(df_clean)}"
            )
        
        # Build formula
        formula = f"{outcome} ~ " + " + ".join([f"C({p})" for p in predictors_list])
        
        # Fit model
        model = ols(formula, data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # ---------- ANOVA TABLE ----------
        anova_table = anova_table.reset_index()
        
        # Handle different column structures
        if len(anova_table.columns) == 5:
            anova_table.columns = ["Source", "SS", "DF", "F", "p_value"]
        elif len(anova_table.columns) == 6:
            anova_table.columns = ["Source", "SS", "DF", "MS", "F", "p_value"]
        
        # Clean source names
        if "Source" in anova_table.columns:
            anova_table["Source"] = anova_table["Source"].astype(str).str.replace("C(", "").str.replace(")", "")
        
        anova_json = dataframe_to_records(anova_table)
        
        # ---------- GROUP MEANS ----------
        group_means = (
            df_clean.groupby(predictors_list)[outcome]
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'Mean', 'std': 'SD', 'count': 'N'})
        )
        group_means_json = dataframe_to_records(group_means)
        
        # ---------- TUKEY HSD ----------
        tukey_json = []
        if len(predictors_list) == 1:
            try:
                tukey = pairwise_tukeyhsd(
                    df_clean[outcome],
                    df_clean[predictors_list[0]]
                )
                tukey_df = pd.DataFrame(
                    data=tukey.summary().data[1:],
                    columns=tukey.summary().data[0]
                )
                tukey_json = dataframe_to_records(tukey_df)
            except Exception as e:
                tukey_json = []
        
        # ---------- ASSUMPTIONS ----------
        residuals = model.resid
        
        # Shapiro-Wilk
        if 3 <= len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Levene test
        groups = [df_clean[df_clean[predictors_list[0]] == g][outcome].values 
                  for g in df_clean[predictors_list[0]].unique()]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            levene_stat, levene_p = stats.levene(*groups)
        else:
            levene_stat, levene_p = None, None
        
        shapiro_result = {
            "statistic": safe_json_value(shapiro_stat),
            "p_value": safe_json_value(shapiro_p),
            "pass": safe_json_value(shapiro_p >= 0.05 if shapiro_p is not None else None)
        }
        
        levene_result = {
            "statistic": safe_json_value(levene_stat),
            "p_value": safe_json_value(levene_p),
            "pass": safe_json_value(levene_p >= 0.05 if levene_p is not None else None)
        }
        
        # ---------- INTERPRETATION ----------
        predictor_rows = anova_table[anova_table["Source"].str.contains(predictors_list[0], case=False, na=False)]
        
        if len(predictor_rows) > 0:
            pval = predictor_rows.iloc[0]["p_value"]
            if pd.notna(pval) and pval < 0.05:
                interpretation = f"Significant difference detected (p = {pval:.4f})"
            elif pd.notna(pval):
                interpretation = f"No significant difference detected (p = {pval:.4f})"
            else:
                interpretation = "Could not determine significance"
        else:
            interpretation = "Could not find treatment effect"
        
        # ---------- RESPONSE ----------
        return {
            "anova_table": anova_json,
            "group_means": group_means_json,
            "tukey_hsd": tukey_json,
            "assumptions": {
                "shapiro_wilk": shapiro_result,
                "levene_test": levene_result
            },
            "interpretation": interpretation,
            "metadata": {
                "n_observations": int(len(df_clean)),
                "n_groups": int(len(df_clean[predictors_list[0]].unique())),
                "outcome_variable": str(outcome),
                "predictor_variables": [str(p) for p in predictors_list]
            }
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": error_trace
            }
        )


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "VivaSense API is running",
        "version": "1.0.1"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
