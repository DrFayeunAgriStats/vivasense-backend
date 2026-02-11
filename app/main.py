from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utility: Clean NaN for JSON ----------
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    else:
        return obj

# ---------- API ROUTE ----------
@app.post("/vivasense/run")
async def run_vivasense(
    file: UploadFile = File(...),
    outcome: str = Form(...),
    predictors: str = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        predictors_list = [p.strip() for p in predictors.split(",")]

        # Use first predictor only for now (one-way ANOVA)
        factor = predictors_list[0]

        # Drop missing rows
        df = df[[outcome, factor]].dropna()

        # ---------------- ANOVA ----------------
        formula = f"{outcome} ~ C({factor})"
        model = smf.ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        anova_table_reset = anova_table.reset_index()
        anova_table_dict = anova_table_reset.to_dict(orient="records")

        # ---------------- Group Means ----------------
        group_means = df.groupby(factor)[outcome].mean().reset_index()
        group_means_dict = group_means.to_dict(orient="records")

        # ---------------- Tukey HSD ----------------
        tukey = pairwise_tukeyhsd(
            endog=df[outcome],
            groups=df[factor],
            alpha=0.05
        )

        tukey_df = pd.DataFrame(
            tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )
        tukey_dict = tukey_df.to_dict(orient="records")

        # ---------------- Assumptions ----------------

        # Shapiro-Wilk (normality of residuals)
        shapiro_stat, shapiro_p = stats.shapiro(model.resid)

        shapiro_result = {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p),
            "normal": bool(shapiro_p > 0.05)
        }

        # Levene (homogeneity of variance)
        groups = [group[outcome].values for name, group in df.groupby(factor)]
        levene_stat, levene_p = stats.levene(*groups)

        levene_result = {
            "statistic": float(levene_stat),
            "p_value": float(levene_p),
            "homogeneous": bool(levene_p > 0.05)
        }

        # ---------------- Interpretation ----------------
        p_value = anova_table["PR(>F)"][0]

        if p_value < 0.05:
            interpretation = "Significant difference detected (p < 0.05)"
        else:
            interpretation = "No significant difference detected (p ≥ 0.05)"

        # ---------------- Response ----------------
        response = {
            "interpretation": interpretation,
            "anova_table": anova_table_dict,
            "group_means": group_means_dict,
            "posthoc": tukey_dict,
            "shapiro_wilk": shapiro_result,
            "levene_test": levene_result
        }

        return clean_nan(response)

    except Exception as e:
        return {"error": str(e)}

# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "VivaSense backend running"}
