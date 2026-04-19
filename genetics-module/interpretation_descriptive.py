from typing import List, Dict, Any, Optional

def classify_cv_precision(cv_percent: Optional[float]) -> str:
    """Classify experimental precision based on VivaSense standard descriptive thresholds."""
    if cv_percent is None: return "unknown"
    if cv_percent < 10: return "excellent"
    if cv_percent <= 20: return "good"
    if cv_percent <= 30: return "moderate"
    return "high (caution)"

def generate_trait_flags(stats: dict, n: int) -> List[str]:
    """Identify potential data quality issues and anomalies for the trait."""
    flags = []
    cv = stats.get("cv_percent")
    if cv is not None and cv > 30:
        flags.append("High variability (CV > 30%)")
    if stats.get("missing_count", 0) > 0:
        flags.append("Missing observations detected")
    if stats.get("zero_count", 0) > 0:
        flags.append("Zero values present")
    if n < 10:
        flags.append("Low sample size (n < 10)")
    mean_val = stats.get("grand_mean")
    if mean_val is not None and abs(mean_val) < 0.01 and mean_val != 0:
        flags.append("Mean close to zero (unstable CV)")
    skew = stats.get("skewness")
    if skew is not None and abs(skew) > 2:
        flags.append("High skewness detected")
    return flags

def generate_trait_interpretation(trait: str, stats: dict, precision_class: str) -> str:
    """Generate a scientifically neutral textual interpretation of a trait's stats."""
    mean = stats.get('grand_mean')
    if mean is None:
        return f"Insufficient data to compute descriptive statistics for {trait}."
    
    cv = stats.get('cv_percent')
    cv_str = f"{cv:.2f}" if cv is not None else "N/A"
    min_v = stats.get('min')
    max_v = stats.get('max')
    
    variability = "low" if cv is not None and cv < 15 else "moderate" if cv is not None and cv <= 30 else "high"
    
    interp = (f"The mean {trait} was {mean:.2f}, ranging from {min_v:.2f} to {max_v:.2f}, "
              f"indicating {variability} variability. ")
    if cv is not None:
        interp += f"The coefficient of variation ({cv_str}%) suggests {precision_class} experimental precision."
    else:
        interp += "The coefficient of variation could not be calculated."
    return interp

def generate_global_recommendation(reliable_traits: List[str], caution_traits: List[str], global_flags: List[str]) -> str:
    """Produce an overall statement regarding the dataset's descriptive readiness."""
    if not caution_traits and not global_flags:
        return "The dataset exhibits excellent experimental precision across all evaluated traits, making it highly suitable for further parametric and genetic analyses."
    
    recs = []
    if reliable_traits:
        recs.append(f"Traits such as {', '.join(reliable_traits[:3])} show reliable precision and are ready for downstream analysis.")
    if caution_traits:
        recs.append(f"Exercise caution with traits showing high variability (e.g., {', '.join(caution_traits[:3])}). Consider data transformation or non-parametric alternatives.")
    if any("Missing" in f for f in global_flags):
        recs.append("Address missing values before proceeding to multivariate analyses.")
        
    return " ".join(recs) if recs else "Proceed with caution due to the noted global flags."