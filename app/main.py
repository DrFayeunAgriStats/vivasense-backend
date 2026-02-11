from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats

app = FastAPI()

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
# Health check
# -----------------------
@app.get("/")
def home():
    return {"status": "VivaSense backend running"}

# -----------------------
# Main Analysis Endpoint
# -----------------------
@app.post("/vivasense/run")
async def run_vivasense(
    file: UploadFile = File(...),
    outcome: str = Form(...),
    predictors: str = Form(...)
):
    try:
        df = pd.read_excel(file.file)

        predictors_list = predictors.split(",")

        results = {}

        # One-way ANOVA
        groups = [
            df[df[predictors_list[0]] == g][outcome]
            for g in df[predictors_list[0]].dropna().unique()
        ]

        f_stat, p_value = stats.f_oneway(*groups)

        results["anova"] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_value)
        }

        results["interpretation"] = (
            "Significant difference detected"
            if p_value < 0.05
            else "No significant difference detected"
        )

        return results

    except Exception as e:
        return {"error": str(e)}
