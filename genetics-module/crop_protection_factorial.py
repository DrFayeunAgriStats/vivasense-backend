"""Validated factorial-CRD adapter for future crop-protection workflows.

This is an internal adapter, not a public API route.  It isolates the two
scientific invariants needed before the wider Crop Protection module exists:
control exclusion and Rep-as-experimental-unit (never an implicit block).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
R_ADAPTER = MODULE_DIR / "crop_protection_factorial.R"
R_SCRIPT = shutil.which("Rscript") or r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe"


class FactorialCrdValidationError(ValueError):
    """Raised when the declared factorial CRD cannot be analysed faithfully."""


def _normalise_dose(value: Any) -> float:
    try:
        dose = float(value)
    except (TypeError, ValueError) as exc:
        raise FactorialCrdValidationError(f"Dose value {value!r} is not numeric.") from exc
    if not math.isfinite(dose):
        raise FactorialCrdValidationError(f"Dose value {value!r} is not finite.")
    return dose


def _run_r_adapter(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not Path(R_SCRIPT).exists():
        raise RuntimeError(f"Rscript not found at {R_SCRIPT}")

    run_id = uuid.uuid4().hex
    input_path = MODULE_DIR / f".crop-protection-{run_id}.json"
    runner_path = MODULE_DIR / f".crop-protection-{run_id}.R"
    try:
        input_path.write_text(json.dumps(records, allow_nan=False), encoding="utf-8")
        runner_path.write_text(
            "\n".join(
                [
                    "suppressPackageStartupMessages(library(jsonlite))",
                    f"source({json.dumps(str(R_ADAPTER))})",
                    f"records <- fromJSON({json.dumps(str(input_path))}, simplifyDataFrame = TRUE)",
                    "result <- compute_crop_protection_factorial_crd(records)",
                    "cat(toJSON(result, auto_unbox = TRUE, na = 'null', digits = 15))",
                ]
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(R_SCRIPT), str(runner_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)
        runner_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Crop-protection R adapter failed.")
    return json.loads(proc.stdout)


def analyze_factorial_crd(
    df: pd.DataFrame,
    *,
    treatment_column: str,
    dose_column: str,
    replicate_column: str,
    response_column: str,
    control_level: Any,
    expected_dose_series: Iterable[Any],
    display_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyse a declared crop-protection Treatment × Dose factorial CRD.

    Replicate is validated as part of the experimental-unit key but is never
    included in the model formula.  The untreated control is retained in the
    returned provenance and excluded before R receives the factorial records.
    """

    columns = [treatment_column, dose_column, replicate_column, response_column]
    if display_column:
        columns.append(display_column)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise FactorialCrdValidationError(f"Mapped columns not found: {missing}")

    expected_doses = tuple(_normalise_dose(value) for value in expected_dose_series)
    if not expected_doses or len(set(expected_doses)) != len(expected_doses):
        raise FactorialCrdValidationError("expected_dose_series must contain unique numeric doses.")

    work = df[columns].copy()
    work["_dose"] = pd.to_numeric(work[dose_column], errors="coerce")
    work["_inference"] = pd.to_numeric(work[response_column], errors="coerce")
    if display_column:
        work["_display"] = pd.to_numeric(work[display_column], errors="coerce")
    else:
        work["_display"] = work["_inference"]

    required_numeric = ["_dose", "_inference", "_display"]
    if work[required_numeric].isna().any().any():
        bad_rows = work.index[work[required_numeric].isna().any(axis=1)].tolist()
        raise FactorialCrdValidationError(
            f"Non-numeric or missing dose/response values at row indices {bad_rows}."
        )

    control_mask = work[treatment_column].astype(str) == str(control_level)
    control = work.loc[control_mask].copy()
    if control.empty:
        raise FactorialCrdValidationError(f"Control level {control_level!r} was not found.")

    treated = work.loc[~control_mask].copy()
    unexpected_doses = sorted(set(treated["_dose"]) - set(expected_doses))
    if unexpected_doses:
        raise FactorialCrdValidationError(
            f"Treated observations contain doses outside expected_dose_series: {unexpected_doses}."
        )

    treatment_levels = treated[treatment_column].astype(str).unique().tolist()
    for treatment in treatment_levels:
        observed = set(treated.loc[treated[treatment_column].astype(str) == treatment, "_dose"])
        missing_doses = sorted(set(expected_doses) - observed)
        if missing_doses:
            raise FactorialCrdValidationError(
                f"Treatment {treatment!r} is missing expected dose levels {missing_doses}."
            )

    duplicate_mask = treated.duplicated(
        subset=[treatment_column, "_dose", replicate_column], keep=False
    )
    if duplicate_mask.any():
        duplicate_keys = treated.loc[
            duplicate_mask, [treatment_column, "_dose", replicate_column]
        ].to_dict("records")
        raise FactorialCrdValidationError(
            f"Duplicate Treatment × Dose × Replicate experimental units: {duplicate_keys}."
        )

    cell_counts = treated.groupby([treatment_column, "_dose"], observed=True).size()
    balanced = cell_counts.nunique() == 1
    warnings = []
    if not balanced:
        warnings.append(
            "Treatment × Dose cells have unequal replication. Cell-specific n and SE are "
            "reported; no common interaction SE is available."
        )

    records = [
        {
            "treatment": str(row[treatment_column]),
            "dose": str(float(row["_dose"])),
            "replicate": str(row[replicate_column]),
            "inference_value": float(row["_inference"]),
            "display_value": float(row["_display"]),
        }
        for _, row in treated.iterrows()
    ]
    result = _run_r_adapter(records)

    for mean in result["interaction"]["means"]:
        # Existing VivaSense consumers expect a concise display-scale mean/SE,
        # while the explicit fields preserve both statistical scales.
        mean["mean"] = mean["mean_display_scale"]
        mean["se"] = mean["se_display_scale"] if display_column else mean["se_inference_scale"]
        mean["dose"] = float(mean["dose"])
        mean["tukey_letter"] = mean["letter"]

    result.update(
        {
            "design": {
                "design_type": "factorial_crd",
                "treatment_levels": len(treatment_levels),
                "dose_levels": len(expected_doses),
                "factorial_n": len(treated),
                "control_n": len(control),
                "balanced": balanced,
            },
            "factorial_rows_used": treated.index.tolist(),
            "rows_excluded": control.index.tolist(),
            "provenance": {
                "inference_scale": "inference_column",
                "display_scale": "display_column" if display_column else "inference_column",
                "inference_column": response_column,
                "display_column": display_column or response_column,
                "control_level": str(control_level),
                "control_excluded_from_factorial": True,
                "replicate_column": replicate_column,
                "replicate_model_role": "experimental_unit_identifier",
            },
            "diagnostic_metadata": {
                "model_formula": result["model_formula"],
                "residual_df": result["error_df"],
                "residual_mean_square": result["residual_mean_square"],
                "replicate_in_model": False,
            },
            "warnings": warnings,
        }
    )
    return result
