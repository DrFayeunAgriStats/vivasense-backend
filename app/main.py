from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

    df = pd.read_excel(file.file)

    predictors_list = [p.strip() for p in predictors.split(",")]

    # Build formula
    formula = f"{outcome} ~ " + " + ".join(predictors_list)

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # ---------- ANOVA TABLE ----------
    anova_table = anova_table.reset_index()
    anova_table.columns = ["Source", "SS", "DF", "MS", "F", "p_value"]

    anova_json = dataframe_to_records(anova_table)

    # ---------- GROUP MEANS ----------
    group_means = (
        df.groupby(predictors_list)[outcome]
        .mean()
        .reset_index()
        .rename(columns={outcome: "Mean"})
    )

    group_means_json = dataframe_to_records(group_means)

    # ---------- TUKEY HSD ----------
    if len(predictors_list) == 1:
        tukey = pairwise_tukeyhsd(
            df[outcome],
            df[predictors_list[0]]
        )
        tukey_df = pd.DataFrame(
            data=tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )
        tukey_json = dataframe_to_records(tukey_df)
    else:
        tukey_json = []

    # ---------- ASSUMPTION TESTS ----------
    residuals = model.resid

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    levene_stat, levene_p = stats.levene(
        *[df[df[predictors_list[0]] == g][outcome]
          for g in df[predictors_list[0]].unique()]
    )

    assumptions = {
        "shapiro_wilk": {
            "statistic": safe_number(shapiro_stat),
            "p_value": safe_number(shapiro_p),
            "pass": shapiro_p >= 0.05
        },
        "levene_test": {
            "statistic": safe_number(levene_stat),
            "p_value": safe_number(levene_p),
            "pass": levene_p >= 0.05
        }
    }

    # ---------- INTERPRETATION ----------
    pval = anova_table.loc[anova_table["Source"] == predictors_list[0], "p_value"].values[0]

    if pval < 0.05:
        interpretation = "Significant difference detected (p < 0.05)"
    else:
        interpretation = "No significant difference detected (p ≥ 0.05)"

    # ---------- FINAL RESPONSE ----------
    return {
        "anova_table": anova_json,
        "group_means": group_means_json,
        "tukey_hsd": tukey_json,
        "assumptions": assumptions,
        "interpretation": interpretation
    }
