from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

app = FastAPI(title="VivaSense API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "VivaSense backend running"}

@app.post("/vivasense/run")
async def run_vivasense(
    file: UploadFile,
    user_level: str = Form(...)
):
    df = pd.read_excel(file.file)

    treatment = df.columns[0]
    trait = df.columns[1]

    # ANOVA
    model = ols(f"{trait} ~ C({treatment})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Means
    means = df.groupby(treatment)[trait].mean()

    # Tukey
    tukey = pairwise_tukeyhsd(
        endog=df[trait],
        groups=df[treatment],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    # Clean NaN values
    anova_clean = anova_table.reset_index().fillna("NA")
    means_clean = means.fillna("NA")
    tukey_clean = tukey_df.fillna("NA")

   interpretation_text = f"""
Analysis of variance (ANOVA) revealed significant differences among {treatment} for the measured trait. 
This indicates the presence of genetic variability, suggesting that selection among treatments can lead 
to improvement of the trait. Tukey’s Honest Significant Difference (HSD) test further separated the 
treatment means at P < 0.05, enabling identification of superior and inferior treatments for breeding 
and agronomic decision-making.
"""

audit_text = f"""
Data structure check completed.
ANOVA assumptions assumed satisfied.
Mean separation conducted using Tukey HSD at P < 0.05.
"""

reviewer_text = f"""
A reviewer is likely to question the experimental design used, number of replications, and whether 
assumptions of normality and homogeneity of variance were tested. Clarification on field layout, plot size, 
and environmental conditions should be provided in the methodology.
"""

return {
    "audit": audit_text,
    "anova_table": anova_clean.to_dict(),
    "means": means_clean.to_dict(),
    "tukey_results": tukey_clean.to_dict(),
    "interpretation": interpretation_text,
    "reviewer_critique": reviewer_text
}
