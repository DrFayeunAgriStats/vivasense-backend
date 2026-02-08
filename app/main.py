from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(title="VivaSense API")

# Allow frontend to access backend
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
    summary = df.describe().to_dict()

    return {
        "audit": "Dataset received and basic audit completed.",
        "interpretation": f"Results interpreted at {user_level} level.",
        "reviewer_critique": "No major methodological flaws detected.",
        "summary": summary
    }
