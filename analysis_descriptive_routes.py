import os
import json
import tempfile
import subprocess
import pandas as pd
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, Form, HTTPException

# Import Word report generator (Task 3)
from report_descriptive_stats import generate_descriptive_stats_word_report

router = APIRouter()

def run_r_descriptive_stats(df: pd.DataFrame, traits: List[str], group_var: Optional[str] = None):
    """Runs the R statistical engine securely via subprocess."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.json")
        # Handle NaNs properly for JSON parsing in R
        df_clean = df.where(pd.notnull(df), None)
        df_clean.to_json(data_path, orient="records")
        
        r_script_content = f"""
        source('c:/Users/ADMIN/vivasense-backend/genetics-module/vivasense_descriptive_stats.R')
        data <- fromJSON('{data_path.replace(os.sep, "/")}')
        traits <- c({", ".join([f"'{t}'" for t in traits])})
        group_var <- {'"' + group_var + '"' if group_var else "NULL"}
        
        results <- list(
            missing_data = assess_missing_data(data),
            summary_stats = list(),
            assumption_tests = list(),
            outliers = list()
        )
        
        for (trait in traits) {
            if (!trait %in% names(data)) next
            results$summary_stats[[trait]] <- compute_descriptive_stats(data, trait, group_var)
            
            norm_res <- test_normality_shapiro(data, trait, group_var)
            homog_res <- NULL
            if (!is.null(group_var)) {
                homog_res <- test_homogeneity_levene(data, trait, group_var)
            }
            
            results$assumption_tests[[trait]] <- list(normality = norm_res, homogeneity = homog_res)
            results$outliers[[trait]] <- detect_outliers_iqr(data, trait)
        }
        
        cat(toJSON(results, auto_unbox = TRUE))
        """
        
        script_path = os.path.join(tmpdir, "run_stats.R")
        with open(script_path, "w") as f:
            f.write(r_script_content)
            
        try:
            result = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"R Script execution failed: {e.stderr}")

def interpret_results(r_results, traits):
    """VivaSense Interpretation Layer: Analyzes assumptions & translates to ANOVA readiness."""
    interpretations = {}
    warnings = []
    anova_readiness = {}

    for trait in traits:
        if trait not in r_results["summary_stats"]:
            continue
            
        stats = r_results["summary_stats"][trait]["overall"]
        normality = r_results["assumption_tests"][trait]["normality"]
        homogeneity = r_results["assumption_tests"][trait].get("homogeneity")
        outliers = r_results["outliers"].get(trait, [])

        # 1. Variability
        cv = stats.get("cv_percent")
        if cv is None:
            var_interp = "Variability could not be calculated."
        elif cv < 10:
            var_interp = f"Low variability (CV = {cv:.2f}%)."
        elif cv <= 30:
            var_interp = f"Moderate variability (CV = {cv:.2f}%)."
        else:
            var_interp = f"High variability (CV = {cv:.2f}%)."

        # 2. Distribution
        skew = stats.get("skewness", 0)
        is_normal = normality.get("normal_05", False)
        dist_interp = "Normally distributed" if is_normal else "Non-normally distributed"
        dist_interp += f" with a skewness of {skew:.2f}."

        # 3. Data Quality
        missing = stats.get("n_missing", 0)
        dq_interp = f"{missing} missing values. "
        if len(outliers) > 0:
            dq_interp += f"{len(outliers)} outliers detected."
            warnings.append(f"{trait}: {len(outliers)} outliers detected.")
        else:
            dq_interp += "No outliers detected."

        # 4. ANOVA Readiness & 5. Decision Guidance
        is_homog = homogeneity.get("homogeneous_05", True) if homogeneity else True

        if not is_normal:
            p_val = normality.get('p_value', 0)
            warnings.append(f"{trait}: Non-normal distribution (p={p_val:.4f})")
            
        if homogeneity and not is_homog:
            p_val = homogeneity.get('p_value', 0)
            warnings.append(f"{trait}: Unequal variances (p={p_val:.4f})")

        if is_normal and is_homog:
            readiness = "Ready for parametric ANOVA"
            decision = "Trait meets all assumptions for standard parametric analysis."
        elif is_normal and not is_homog:
            readiness = "Use Welch's ANOVA"
            decision = "Trait is normal but variances are unequal. Welch's ANOVA is recommended."
        else:
            readiness = "Consider transformation or Kruskal-Wallis"
            decision = "Trait violates normality assumption. Consider data transformation (e.g., log, square root) or non-parametric tests like Kruskal-Wallis."

        anova_readiness[trait] = readiness
        interpretations[trait] = {
            "variability": var_interp,
            "distribution": dist_interp,
            "data_quality": dq_interp,
            "decision_guidance": decision
        }

    return interpretations, warnings, anova_readiness

@router.post("/statistics/descriptive-analysis")
async def descriptive_analysis(
    file: UploadFile,
    traits: str = Form(...),
    group_variable: Optional[str] = Form(None)
):
    try:
        traits_list = json.loads(traits)
    except Exception:
        traits_list = [t.strip() for t in traits.split(",")]

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    missing_traits = [t for t in traits_list if t not in df.columns]
    if missing_traits:
        raise HTTPException(status_code=400, detail=f"Traits not found in dataset: {missing_traits}")

    if group_variable and group_variable not in df.columns:
        raise HTTPException(status_code=400, detail=f"Group variable not found: {group_variable}")

    # 1. R Compute
    r_results = run_r_descriptive_stats(df, traits_list, group_variable)
    
    # 2. Python Interpret
    interpretations, warnings, anova_readiness = interpret_results(r_results, traits_list)
    
    # 3. Word Report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    report_url = f"{out_dir}/descriptive_stats_{timestamp}.docx"
    
    generate_descriptive_stats_word_report(r_results, df, traits_list, group_variable, interpretations, anova_readiness, warnings, report_url)

    return {
        "status": "success",
        "summary_stats": r_results["summary_stats"],
        "assumption_tests": r_results["assumption_tests"],
        "missing_data": r_results["missing_data"],
        "outliers": r_results["outliers"],
        "warnings": warnings,
        "interpretations": interpretations,
        "anova_readiness": anova_readiness,
        "report_url": report_url
    }