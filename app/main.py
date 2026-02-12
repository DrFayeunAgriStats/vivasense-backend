from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import traceback

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
def safe_number(x):
    """Convert NaN/inf to None so JSON does not crash"""
    if pd.isna(x) or np.isinf(x):
        return None
    return float(x)

def dataframe_to_records(df):
    return [
        {col: safe_number(val) if isinstance(val, (int, float)) else val
         for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]

# -----------------------
# API Endpoint
# -----------------------
@app.post("/vivasense/run")
async def run_vivasense(
    file: UploadFile,
    outcome: str = Form(...),
    predictors: str = Form(...)
):
    try:
        # Read the Excel file
        df = pd.read_excel(file.file)
        
        # Clean column names (remove extra spaces, standardize)
        df.columns = df.columns.str.strip()
        
        # Parse predictors
        predictors_list = [p.strip() for p in predictors.split(",")]
        
        # VALIDATION: Check if outcome column exists
        if outcome not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Outcome column '{outcome}' not found. Available columns: {list(df.columns)}"
            )
        
        # VALIDATION: Check if predictor columns exist
        missing_predictors = [p for p in predictors_list if p not in df.columns]
        if missing_predictors:
            raise HTTPException(
                status_code=400,
                detail=f"Predictor columns {missing_predictors} not found. Available columns: {list(df.columns)}"
            )
        
        # VALIDATION: Check if outcome is numeric
        if not pd.api.types.is_numeric_dtype(df[outcome]):
            raise HTTPException(
                status_code=400,
                detail=f"Outcome column '{outcome}' must be numeric. Current type: {df[outcome].dtype}"
            )
        
        # Drop rows with missing values in required columns
        required_cols = [outcome] + predictors_list
        df_clean = df[required_cols].dropna()
        
        if len(df_clean) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data after removing missing values. Need at least 3 rows, got {len(df_clean)}"
            )
        
        # Build formula
        formula = f"{outcome} ~ " + " + ".join([f"C({p})" for p in predictors_list])
        
        # Fit model
        model = ols(formula, data=df_clean).fit()
        
        # Get ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # ---------- ANOVA TABLE ----------
        anova_table = anova_table.reset_index()
        anova_table.columns = ["Source", "SS", "DF", "MS", "F", "p_value"]
        
        # Clean up source names (remove C() wrapper)
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
                # If Tukey fails, just skip it
                tukey_json = [{"error": f"Tukey test failed: {str(e)}"}]
        
        # ---------- ASSUMPTION TESTS ----------
        residuals = model.resid
        
        # Shapiro-Wilk test
        if len(residuals) >= 3 and len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Levene test
        groups = [df_clean[df_clean[predictors_list[0]] == g][outcome].values 
                  for g in df_clean[predictors_list[0]].unique()]
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            levene_stat, levene_p = stats.levene(*groups)
        else:
            levene_stat, levene_p = None, None
        
        assumptions = {
            "shapiro_wilk": {
                "statistic": safe_number(shapiro_stat),
                "p_value": safe_number(shapiro_p),
                "pass": shapiro_p >= 0.05 if shapiro_p is not None else None
            },
            "levene_test": {
                "statistic": safe_number(levene_stat),
                "p_value": safe_number(levene_p),
                "pass": levene_p >= 0.05 if levene_p is not None else None
            }
        }
        
        # ---------- INTERPRETATION ----------
        # Find the predictor row in ANOVA table
        predictor_rows = anova_table[anova_table["Source"].str.contains(predictors_list[0], case=False)]
        
        if len(predictor_rows) > 0:
            pval = predictor_rows.iloc[0]["p_value"]
            if pd.notna(pval) and pval < 0.05:
                interpretation = f"Significant difference detected (p = {pval:.4f})"
            elif pd.notna(pval):
                interpretation = f"No significant difference detected (p = {pval:.4f})"
            else:
                interpretation = "Could not determine significance"
        else:
            interpretation = "Could not find treatment effect in ANOVA table"
        
        # ---------- FINAL RESPONSE ----------
        return {
            "anova_table": anova_json,
            "group_means": group_means_json,
            "tukey_hsd": tukey_json,
            "assumptions": assumptions,
            "interpretation": interpretation,
            "metadata": {
                "n_observations": len(df_clean),
                "n_groups": len(df_clean[predictors_list[0]].unique()),
                "outcome_variable": outcome,
                "predictor_variables": predictors_list
            }
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Catch any other errors and return detailed error message
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": error_trace
            }
        )


# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "message": "VivaSense API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
