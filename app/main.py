from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

app = FastAPI(title="VivaSense API")

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

    # First column = Treatment
    # Second column = Trait
    treatment = df.columns[0]
    trait = df.columns[1]

    # ---------- ANOVA ----------
    model = ols(f"{trait} ~ C({treatment})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # ---------- MEANS ----------
    means = df.groupby(treatment)[trait].mean()

    # ---------- TUKEY ----------
    tukey = pairwise_tukeyhsd(
        endog=df[trait],
        groups=df[treatment],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    return {
        "audit": "ANOVA and mean separation successfully completed.",
        "anova_table": anova_table.reset_index().to_dict(),
        "means": means.to_dict(),
        "tukey_results": tukey_df.to_dict(),
        "interpretation": (
            f"ANOVA detected whether significant differences exist among {treatment}. "
            "Tukey HSD separated treatment means at P < 0.05."
        ),
        "reviewer_critique": (
            "Reviewer will ask about experimental design, replication, "
            "and whether assumptions of ANOVA were satisfied."
        )
    }
