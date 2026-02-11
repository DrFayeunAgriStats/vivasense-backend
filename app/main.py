from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats

app = FastAPI(title="VivaSense API")

# ---------------------------
# CORS SETTINGS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# CLEAN JSON VALUES
# ---------------------------
def clean_json(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_json(v) for v in obj]

    return obj

# ---------------------------
# HOME
# ---------------------------
@app.get("/")
def home():
    return {"status": "VivaSense backend running"}

# ---------------------------
# RUN ANALYSIS
# ---------------------------
@app.post("/vivasense/run")
async def run_vivasense(
    file: UploadFile = File(...),
    outcome: str = Form(...),
    predictors: str = Form(...)
):
    try:
        # Load dataset
        df = pd.read_excel(file.file)

        predictor_list = [p.strip() for p in predictors.split(",")]

        groups = []
        for level in df[predictor_list[0]].unique():
            group = df[df[predictor_list[0]] == level][outcome]
            groups.append(group)

        # Run ANOVA
        f_stat, p_val = stats.f_oneway(*groups)

        results = {
            "analysis": "ANOVA",
            "outcome_variable": outcome,
            "predictor": predictor_list,
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "interpretation": (
                "Significant difference detected (p < 0.05)"
                if p_val < 0.05
                else "No significant difference detected (p ≥ 0.05)"
            )
        }

        results = clean_json(results)

        return results

    except Exception as e:
        return {"error": str(e)}
