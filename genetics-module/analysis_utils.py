import pandas as pd
from typing import Dict, List, Optional, Any

def compute_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute numeric descriptive statistics for a trait series safely."""
    missing_count = int(series.isna().sum())
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    zero_count = int((clean == 0).sum())

    if n == 0:
        return {
            "grand_mean": None,
            "standard_deviation": None,
            "variance": None,
            "standard_error": None,
            "cv_percent": None,
            "min": None,
            "max": None,
            "range": None,
            "median": None,
            "skewness": None,
            "kurtosis": None,
            "missing_count": missing_count,
            "zero_count": zero_count,
        }

    grand_mean = float(clean.mean())
    min_val = float(clean.min())
    max_val = float(clean.max())
    range_val = max_val - min_val
    median_val = float(clean.median())

    if n >= 2:
        variance = float(clean.var(ddof=1))
        standard_deviation = float(variance ** 0.5)
        standard_error = float(standard_deviation / (n ** 0.5))
    else:
        variance = None
        standard_deviation = None
        standard_error = None
        
    skewness = float(clean.skew()) if n >= 3 else None
    kurtosis = float(clean.kurt()) if n >= 4 else None

    cv_percent = None
    if grand_mean != 0 and standard_deviation is not None:
        cv_percent = float((standard_deviation / abs(grand_mean)) * 100)

    return {
        "grand_mean": grand_mean,
        "standard_deviation": standard_deviation,
        "variance": variance,
        "standard_error": standard_error,
        "cv_percent": cv_percent,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "median": median_val,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "missing_count": missing_count,
        "zero_count": zero_count,
    }


def compute_per_genotype_stats(
    df: pd.DataFrame, trait_column: str, genotype_column: str
) -> List[Dict[str, Optional[float]]]:
    """Compute per-genotype descriptive statistics for the requested trait."""
    if genotype_column not in df.columns:
        return []
    grouped = df[[genotype_column, trait_column]].copy()
    grouped[trait_column] = pd.to_numeric(grouped[trait_column], errors="coerce")
    stats: List[Dict[str, Optional[float]]] = []
    for genotype, group in grouped.groupby(genotype_column, sort=True):
        clean = group[trait_column].dropna()
        n = len(clean)
        if n == 0:
            stats.append({"genotype": genotype, "mean": None, "sd": None, "cv_percent": None})
            continue
        mean_val = float(clean.mean())
        sd_val, cv_percent = None, None
        if n >= 2:
            variance = float(clean.var(ddof=1))
            sd_val = float(variance ** 0.5)
            if mean_val != 0:
                cv_percent = float((sd_val / abs(mean_val)) * 100)
        stats.append({"genotype": genotype, "mean": mean_val, "sd": sd_val, "cv_percent": cv_percent})
    return stats


def classify_precision_level(cv_percent: Optional[float]) -> str:
    """Classify experimental precision based on coefficient of variation (Legacy/ANOVA mapping)."""
    if cv_percent is None:
        return "low"
    if cv_percent < 10.0:
        return "good"
    elif cv_percent <= 20.0:
        return "moderate"
    else:
        return "low"