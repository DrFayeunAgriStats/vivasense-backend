"""Validated dynamic one-to-three-factor CRD adapter for crop protection."""
from __future__ import annotations
import json, shutil, subprocess, uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
R_ADAPTER = MODULE_DIR / "crop_protection_factorial.R"
R_SCRIPT = shutil.which("Rscript") or r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe"

class FactorialCrdValidationError(ValueError): pass

def _run_r_adapter(records: Sequence[Dict[str, Any]], factor_names: list[str]) -> Dict[str, Any]:
    if not Path(R_SCRIPT).exists(): raise RuntimeError(f"Rscript not found at {R_SCRIPT}")
    run_id = uuid.uuid4().hex
    input_path, runner_path = (MODULE_DIR / f".crop-protection-{run_id}{suffix}" for suffix in (".json", ".R"))
    try:
        input_path.write_text(json.dumps(records, allow_nan=False), encoding="utf-8")
        runner_path.write_text("\n".join([
            "suppressPackageStartupMessages(library(jsonlite))",
            f"source({json.dumps(str(R_ADAPTER))})",
            f"records <- fromJSON({json.dumps(str(input_path))}, simplifyDataFrame = TRUE)",
            "result <- compute_crop_protection_factorial_crd(records, c(" +
            ", ".join(json.dumps(name) for name in factor_names) + "))",
            "cat(toJSON(result, auto_unbox = TRUE, na = 'null', digits = 15))",
        ]), encoding="utf-8")
        proc = subprocess.run([str(R_SCRIPT), str(runner_path)], capture_output=True, text=True,
                              encoding="utf-8", timeout=120, check=False)
    finally:
        input_path.unlink(missing_ok=True); runner_path.unlink(missing_ok=True)
    if proc.returncode: raise RuntimeError(proc.stderr.strip() or "Crop-protection R adapter failed.")
    return json.loads(proc.stdout)

def analyze_factorial_crd(df: pd.DataFrame, *, replicate_column: str, response_column: str,
    factor_columns: Optional[Sequence[str]] = None, factor_display_names: Optional[Sequence[str]] = None,
    display_column: Optional[str] = None, control_column: Optional[str] = None, control_level: Any = None,
    treatment_column: Optional[str] = None, dose_column: Optional[str] = None,
    expected_dose_series: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Fit response ~ Factor1 * ... * FactorN; Rep identifies units only."""
    legacy = factor_columns is None
    factors = list(factor_columns or [treatment_column, dose_column])
    if not 1 <= len(factors) <= 3 or any(not f for f in factors):
        raise FactorialCrdValidationError("Declare between one and three experimental factors.")
    if len(set(factors)) != len(factors) or replicate_column in factors:
        raise FactorialCrdValidationError("Experimental factors and replicate must be distinct.")
    labels = list(factor_display_names or factors)
    columns = list(dict.fromkeys(factors + [replicate_column, response_column] + ([display_column] if display_column else [])))
    missing = [c for c in columns if c not in df.columns]
    if missing: raise FactorialCrdValidationError(f"Mapped columns not found: {missing}")
    work = df[columns].copy()
    work["_inference"] = pd.to_numeric(work[response_column], errors="coerce")
    work["_display"] = pd.to_numeric(work[display_column], errors="coerce") if display_column else work["_inference"]
    if work[["_inference", "_display"]].isna().any().any():
        bad = work.index[work[["_inference", "_display"]].isna().any(axis=1)].tolist()
        raise FactorialCrdValidationError(f"Non-numeric or missing response values at row indices {bad}.")
    if control_level is not None:
        control_col = control_column or factors[0]
        mask = work[control_col].astype(str) == str(control_level)
        if not mask.any(): raise FactorialCrdValidationError(f"Control level {control_level!r} was not found.")
    else: mask = pd.Series(False, index=work.index)
    control, factorial = work.loc[mask], work.loc[~mask].copy()
    if legacy and expected_dose_series:
        expected = {float(v) for v in expected_dose_series}
        factorial[dose_column] = pd.to_numeric(factorial[dose_column], errors="coerce")
        for level, group in factorial.groupby(treatment_column, observed=True):
            missing_doses = sorted(expected - set(group[dose_column]))
            if missing_doses: raise FactorialCrdValidationError(f"Treatment {level!r} is missing expected dose levels {missing_doses}.")
    unit_columns = factors + [replicate_column]
    duplicated = factorial.duplicated(unit_columns, keep=False)
    if duplicated.any():
        prefix = "Duplicate Treatment × Dose × Replicate experimental units" if legacy else "Duplicate factorial experimental units"
        raise FactorialCrdValidationError(f"{prefix}: {factorial.loc[duplicated, unit_columns].to_dict('records')}.")
    counts = factorial.groupby(factors, observed=True).size(); balanced = counts.nunique() == 1
    internal = [f"factor_{i+1}" for i in range(len(factors))]
    records = []
    for _, row in factorial.iterrows():
        record = {name: str(row[col]) for name, col in zip(internal, factors)}
        record.update(replicate=str(row[replicate_column]), inference_value=float(row["_inference"]), display_value=float(row["_display"]))
        records.append(record)
    result = _run_r_adapter(records, internal); source_map = dict(zip(internal, labels))
    for row in result["anova"]:
        if row["source"] != "Error":
            parts = [source_map.get(p, p) for p in row["source"].split(":")]
            row["source"] = ":".join(parts) if legacy else " × ".join(parts)
    for mean in result["cell_means"]:
        mean["factor_levels"] = {label: mean.pop(name) for name, label in zip(internal, labels)}
        if legacy:
            mean["treatment"] = mean["factor_levels"][labels[0]]
            mean["dose"] = float(mean["factor_levels"][labels[1]])
        mean["mean"] = mean["mean_display_scale"]; mean["se"] = mean["se_display_scale"] if display_column else mean["se_inference_scale"]
        mean["tukey_letter"] = mean["letter"]
    marginal = {}
    for key, rows in result["marginal_means"].items():
        if isinstance(rows, dict): rows = list(rows.values())
        display_key = " × ".join(source_map.get(p, p) for p in key.split(":"))
        for row in rows: row["factor_levels"] = {source_map.get(n, n): row.pop(n) for n in key.split(":")}
        marginal[display_key] = rows
    result["marginal_means"] = marginal; result["interaction"]["means"] = result["cell_means"]
    if legacy:
        result["marginal_means"] = {
            "treatment": marginal[labels[0]], "dose": marginal[labels[1]]
        }
        for row in result["marginal_means"]["treatment"]:
            row["level"] = row["factor_levels"][labels[0]]
        for row in result["marginal_means"]["dose"]:
            row["level"] = row["factor_levels"][labels[1]]
    result.update({"design": {"design_type":"factorial_crd","factor_count":len(factors),
        "factors":[{"column":c,"display_name":l,"levels":int(factorial[c].nunique())} for c,l in zip(factors,labels)],
        "factorial_n":len(factorial),"control_n":len(control),"cells":len(counts),"balanced":bool(balanced),
        "cell_replication":int(counts.iloc[0]) if balanced else None,
        "cell_counts":[{"factor_levels":dict(zip(labels,k if isinstance(k,tuple) else (k,))),"n":int(n)} for k,n in counts.items()]},
        "factorial_rows_used":factorial.index.tolist(),"rows_excluded":control.index.tolist(),
        "provenance":{"factor_columns":factors,"factor_display_names":labels,"replicate_column":replicate_column,
                      "replicate_model_role":"experimental_unit_identifier","control_level":None if control_level is None else str(control_level),
                      "control_excluded_from_factorial":control_level is not None},
        "diagnostic_metadata":{"model_formula":result["model_formula"],"residual_df":result["error_df"],
                               "residual_mean_square":result["residual_mean_square"],"replicate_in_model":False},
        "warnings":[] if balanced else ["Full-factor cells have unequal replication."]})
    result["provenance"].update({"inference_column": response_column,
                                 "display_column": display_column or response_column})
    if legacy:
        result["model_formula"] = "response ~ treatment * dose"
        result["diagnostic_metadata"]["model_formula"] = result["model_formula"]
        result["design"] = {"design_type":"factorial_crd",
                            "treatment_levels":int(factorial[treatment_column].nunique()),
                            "dose_levels":int(factorial[dose_column].nunique()),
                            "factorial_n":len(factorial),"control_n":len(control),
                            "balanced":bool(balanced)}
    return result
