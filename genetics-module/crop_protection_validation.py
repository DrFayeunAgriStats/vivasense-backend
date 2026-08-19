"""Request-level validation helpers for crop-protection orchestration."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from crop_protection_schemas import BioassayAnalysisRequest


class BioassayValidationError(ValueError):
    pass


def validate_bioassay_dataframe(
    df: pd.DataFrame, request: BioassayAnalysisRequest
) -> Dict[str, Any]:
    design = request.design
    response_columns: List[str] = []
    for response in request.responses:
        response_columns.extend(
            column for column in [
                response.raw_column, response.inference_column, response.display_column,
                response.transformed_column, response.corrected_column,
            ] if column
        )
    required = {
        design.replicate_column,
        *(factor.column for factor in design.factor_columns),
        *response_columns,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise BioassayValidationError(f"Mapped columns not found: {missing}")
    if df[design.replicate_column].isna().any():
        raise BioassayValidationError("Replicate identifiers contain missing values.")
    dose_factor = next((factor for factor in design.factor_columns
                        if factor.id == design.dose_factor_id), None)
    if dose_factor and design.expected_dose_series:
        dose = pd.to_numeric(df[dose_factor.column], errors="coerce")
        if dose.isna().any():
            raise BioassayValidationError("Dose factor contains missing or non-numeric values.")
    control_factor = next((factor for factor in design.factor_columns
                           if factor.semantic_role == "treatment"), design.factor_columns[0])
    if design.control_treatment_level is not None and not (
        df[control_factor.column].astype(str) == design.control_treatment_level
    ).any():
        raise BioassayValidationError(
            f"Declared control level {design.control_treatment_level!r} was not found."
        )
    duplicate = df.duplicated(
        [*(factor.column for factor in design.factor_columns), design.replicate_column], keep=False
    )
    # Exact repeated control blocks are handled by the explicit Phase 2 policy;
    # duplicate treated experimental units are always invalid.
    treated_duplicate = duplicate & (
        (design.control_treatment_level is None) |
        (df[control_factor.column].astype(str) != design.control_treatment_level)
    )
    if treated_duplicate.any():
        keys = df.loc[
            treated_duplicate,
            [*(factor.column for factor in design.factor_columns), design.replicate_column],
        ].to_dict("records")
        raise BioassayValidationError(
            f"Duplicate Treatment × Dose × Replicate experimental units: {keys}."
        )

    response_issues = []
    for response in request.responses:
        for column in {response.raw_column, response.inference_column}:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.isna().any():
                response_issues.append(
                    {"response_id": response.id, "column": column,
                     "rows": numeric.index[numeric.isna()].tolist()}
                )
        raw = pd.to_numeric(df[response.raw_column], errors="coerce")
        if response.type == "mortality" and ((raw < 0) | (raw > 100)).any():
            raise BioassayValidationError(
                f"Mortality response {response.id!r} must use raw percentages from 0 to 100."
            )
        if response.type == "count" and (raw < 0).any():
            raise BioassayValidationError(
                f"Count response {response.id!r} contains negative observations."
            )
    if response_issues:
        raise BioassayValidationError(
            f"Selected responses contain missing/non-numeric values: {response_issues}"
        )
    return {"validated_rows": len(df), "validated_response_ids": [r.id for r in request.responses]}


def warning(
    code: str, message: str, *, severity: str = "warning",
    response_id: str | None = None, details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "response_id": response_id,
        "message": message,
        "details": details or {},
    }
