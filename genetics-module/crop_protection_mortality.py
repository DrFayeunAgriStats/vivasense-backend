"""Control-aware mortality preparation for crop-protection bioassays.

All mappings are explicit.  Column names and observation times are never
inferred, and Abbott correction is always calculated from raw percentages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd


class MortalityValidationError(ValueError):
    """Raised when mortality correction cannot be performed scientifically."""


ControlPolicy = Literal["require_unique", "deduplicate_identical_replicates"]


@dataclass(frozen=True)
class MortalityResponseMapping:
    raw_column: str
    observation_time: float
    time_unit: str
    transformed_column: Optional[str] = None
    corrected_column: Optional[str] = None


def _validate_percentage(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        rows = numeric.index[numeric.isna()].tolist()
        raise MortalityValidationError(f"{label} contains missing/non-numeric values at rows {rows}.")
    outside = numeric[(numeric < 0) | (numeric > 100)]
    if not outside.empty:
        raise MortalityValidationError(
            f"{label} must be on a 0–100 percentage scale; invalid rows {outside.index.tolist()}."
        )
    return numeric.astype(float)


def select_control_reference(
    df: pd.DataFrame,
    *,
    treatment_column: str,
    dose_column: str,
    replicate_column: str,
    control_level: Any,
    mortality_responses: Sequence[MortalityResponseMapping],
    control_policy: ControlPolicy = "require_unique",
    control_row_indices: Optional[Iterable[Any]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Select one declared untreated-control reference set.

    Repeated identical blocks are never silently counted as extra replicates.
    They require either explicit row selection or the explicit deduplication
    policy, whose key is Rep plus every mapped raw mortality response.
    """

    controls = df.loc[df[treatment_column].astype(str) == str(control_level)].copy()
    if controls.empty:
        raise MortalityValidationError(f"Declared control level {control_level!r} was not found.")

    control_doses = controls[dose_column].dropna().unique().tolist()
    if len(control_doses) != 1:
        raise MortalityValidationError(
            f"Multiple control definitions detected across dose levels {control_doses}."
        )

    original_n = len(controls)
    selected_by_indices = control_row_indices is not None
    if selected_by_indices:
        requested = list(control_row_indices or [])
        controls = controls.loc[controls.index.intersection(requested)].copy()
        if controls.empty or set(requested) - set(controls.index):
            raise MortalityValidationError(
                "control_row_indices must identify rows belonging to the declared control."
            )

    dedupe_columns = [replicate_column] + [mapping.raw_column for mapping in mortality_responses]
    duplicate_mask = controls.duplicated(subset=dedupe_columns, keep=False)
    repeated_blocks_detected = bool(duplicate_mask.any())
    duplicates_removed = 0
    if repeated_blocks_detected and not selected_by_indices:
        if control_policy != "deduplicate_identical_replicates":
            raise MortalityValidationError(
                "Repeated control blocks were detected. Select the intended control rows or explicitly "
                "use control_policy='deduplicate_identical_replicates'."
            )
        before = len(controls)
        controls = controls.drop_duplicates(subset=dedupe_columns, keep="first")
        duplicates_removed = before - len(controls)

    if controls.duplicated(subset=[replicate_column], keep=False).any():
        raise MortalityValidationError(
            "The selected control reference contains repeated replicate labels with differing mortality "
            "profiles; select one intended control block explicitly."
        )

    provenance = {
        "control_level": str(control_level),
        "control_dose": control_doses[0],
        "control_rows_available": original_n,
        "control_rows_used": controls.index.tolist(),
        "n_control": len(controls),
        "repeated_control_blocks_detected": repeated_blocks_detected,
        "duplicates_removed": duplicates_removed,
        "selection_rule": "explicit_row_indices" if selected_by_indices else control_policy,
    }
    return controls, provenance


def prepare_mortality_responses(
    df: pd.DataFrame,
    *,
    treatment_column: str,
    dose_column: str,
    replicate_column: str,
    control_level: Any,
    mortality_responses: Sequence[MortalityResponseMapping],
    floor_at_zero: bool = False,
    control_policy: ControlPolicy = "require_unique",
    control_row_indices: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Calculate time-specific Abbott correction and scale provenance."""

    if not mortality_responses:
        raise MortalityValidationError("At least one explicit mortality response mapping is required.")
    times = [(m.observation_time, m.time_unit) for m in mortality_responses]
    if len(times) != len(set(times)):
        raise MortalityValidationError("Observation time mappings must be unique.")

    required = {treatment_column, dose_column, replicate_column}
    for mapping in mortality_responses:
        required.add(mapping.raw_column)
        if mapping.transformed_column:
            required.add(mapping.transformed_column)
        if mapping.corrected_column:
            required.add(mapping.corrected_column)
    missing = sorted(required - set(df.columns))
    if missing:
        raise MortalityValidationError(f"Mapped mortality columns not found: {missing}")

    controls, control_provenance = select_control_reference(
        df,
        treatment_column=treatment_column,
        dose_column=dose_column,
        replicate_column=replicate_column,
        control_level=control_level,
        mortality_responses=mortality_responses,
        control_policy=control_policy,
        control_row_indices=control_row_indices,
    )

    selected_control_indices = set(controls.index)
    # Remove repeated, unselected control rows from the analytical view while
    # preserving their count and original indices in provenance.
    analytical = df.loc[
        (df[treatment_column].astype(str) != str(control_level))
        | df.index.isin(selected_control_indices)
    ].copy()

    response_results: List[Dict[str, Any]] = []
    all_long_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for mapping in mortality_responses:
        raw = _validate_percentage(analytical[mapping.raw_column], mapping.raw_column)
        control_raw = _validate_percentage(controls[mapping.raw_column], mapping.raw_column)
        control_mean = float(control_raw.mean())
        if control_mean >= 100:
            raise MortalityValidationError(
                f"Abbott correction is undefined at {mapping.observation_time} {mapping.time_unit}: "
                "mean control mortality is 100%."
            )
        if control_mean > 0:
            warnings.append(
                f"Observed control mortality at {mapping.observation_time} {mapping.time_unit} was "
                f"{control_mean:.4g}%. Assay-validity thresholds depend on the biological protocol."
            )

        calculated_for_verification: List[float] = []
        supplied_for_verification: List[float] = []
        response_rows: List[Dict[str, Any]] = []
        for idx, row in analytical.iterrows():
            is_control = str(row[treatment_column]) == str(control_level)
            raw_value = float(raw.loc[idx])
            transformed = None
            if mapping.transformed_column:
                transformed_value = pd.to_numeric(
                    pd.Series([row[mapping.transformed_column]]), errors="coerce"
                ).iloc[0]
                transformed = None if pd.isna(transformed_value) else float(transformed_value)

            if is_control:
                raw_abbott = display_abbott = None
                floor_applied = False
                status = "reference_control"
            else:
                raw_abbott = 100.0 * (raw_value - control_mean) / (100.0 - control_mean)
                floor_applied = bool(floor_at_zero and raw_abbott < 0)
                display_abbott = max(0.0, raw_abbott) if floor_at_zero else raw_abbott
                status = "calculated"
                if mapping.corrected_column:
                    supplied = pd.to_numeric(
                        pd.Series([row[mapping.corrected_column]]), errors="coerce"
                    ).iloc[0]
                    if not pd.isna(supplied):
                        calculated_for_verification.append(float(display_abbott))
                        supplied_for_verification.append(float(supplied))

            output_row = {
                "source_row": idx,
                "treatment": str(row[treatment_column]),
                "dose": float(row[dose_column]),
                "replicate": str(row[replicate_column]),
                "observation_time": mapping.observation_time,
                "time_unit": mapping.time_unit,
                "raw_mortality": raw_value,
                "transformed_mortality": transformed,
                "raw_abbott_value": raw_abbott,
                "display_abbott_value": display_abbott,
                "floor_applied": floor_applied,
                "abbott_status": status,
            }
            response_rows.append(output_row)
            all_long_rows.append(output_row)

        if mapping.corrected_column:
            differences = np.abs(
                np.asarray(calculated_for_verification) - np.asarray(supplied_for_verification)
            )
            mismatch_count = int(np.sum(differences > 1e-6))
            verification = {
                "supplied_column": mapping.corrected_column,
                "max_absolute_difference": float(differences.max()) if len(differences) else None,
                "mismatch_count": mismatch_count,
                "verification_status": "matched" if mismatch_count == 0 else "mismatch",
            }
        else:
            verification = {
                "supplied_column": None,
                "max_absolute_difference": None,
                "mismatch_count": 0,
                "verification_status": "not_supplied",
            }

        response_results.append(
            {
                "observation_time": mapping.observation_time,
                "time_unit": mapping.time_unit,
                "control_mean_raw_mortality": control_mean,
                "rows": response_rows,
                "supplied_correction_verification": verification,
                "provenance": {
                    "raw_column": mapping.raw_column,
                    "transformed_column": mapping.transformed_column,
                    "corrected_column": mapping.corrected_column,
                    "abbott_source": "raw_mortality_percentage",
                    "control_matching": "time_specific_declared_control_mean",
                    "floor_at_zero": floor_at_zero,
                },
            }
        )

    return {
        "responses": response_results,
        "long_rows": all_long_rows,
        "control": control_provenance,
        "provenance": {
            "anova_inference_scale": "explicit_transformed_column_when_mapped",
            "mortality_raw_scale": "percentage_0_100",
            "abbott_scale": "corrected_percentage_0_100" if floor_at_zero else "unbounded_corrected_percentage",
            "cotoxicity_scale": "abbott_corrected_percentage",
        },
        "warnings": warnings,
    }


def validate_cumulative_mortality(
    long_rows: Sequence[Dict[str, Any]], *, cumulative: bool
) -> Dict[str, Any]:
    """Flag raw mortality decreases within Treatment × Dose × Rep trajectories."""

    if not cumulative:
        return {"checked": False, "decrease_count": 0, "decreases": [], "warnings": []}
    frame = pd.DataFrame(long_rows)
    decreases: List[Dict[str, Any]] = []
    for key, group in frame.groupby(["treatment", "dose", "replicate"], sort=False):
        ordered = group.sort_values("observation_time")
        values = ordered["raw_mortality"].to_numpy(dtype=float)
        times = ordered["observation_time"].tolist()
        for position in np.where(np.diff(values) < 0)[0]:
            decreases.append(
                {
                    "treatment": key[0],
                    "dose": float(key[1]),
                    "replicate": key[2],
                    "from_time": times[position],
                    "to_time": times[position + 1],
                    "from_raw_mortality": float(values[position]),
                    "to_raw_mortality": float(values[position + 1]),
                }
            )
    warnings = []
    if decreases:
        warnings.append(
            "Mortality decreases across successive observation times in some experimental units. "
            "Confirm whether mortality was recorded cumulatively before interpreting time-course results."
        )
    return {
        "checked": True,
        "scale_checked": "raw_mortality_percentage",
        "decrease_count": len(decreases),
        "decreases": decreases,
        "warnings": warnings,
    }
