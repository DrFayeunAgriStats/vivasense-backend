"""Bliss-independence joint-action analysis for corrected bioassay mortality."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


class CotoxicityValidationError(ValueError):
    """Raised when explicitly declared joint-action roles are invalid."""


def bliss_expected(component_a: float, component_b: float) -> float:
    """Expected mortality percentage under Bliss independence."""

    return component_a + component_b - (component_a * component_b / 100.0)


def bootstrap_bliss_excess(
    component_a: Sequence[float],
    component_b: Sequence[float],
    mixture: Sequence[float],
    *,
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Stratified nonparametric bootstrap of observed minus Bliss expected.

    Each treatment group is independently resampled at its original size.
    Observations are never pooled across the three biological roles.
    """

    if iterations < 100:
        raise CotoxicityValidationError("bootstrap_iterations must be at least 100.")
    if not 0 < confidence_level < 1:
        raise CotoxicityValidationError("confidence_level must lie strictly between 0 and 1.")
    arrays = [np.asarray(values, dtype=float) for values in (component_a, component_b, mixture)]
    if any(len(values) == 0 for values in arrays):
        raise CotoxicityValidationError("All three treatment roles require observations for bootstrap.")

    rng = np.random.default_rng(seed)
    means = []
    for values in arrays:
        draws = rng.choice(values, size=(iterations, len(values)), replace=True)
        means.append(draws.mean(axis=1))
    expected = means[0] + means[1] - (means[0] * means[1] / 100.0)
    excess = means[2] - expected
    alpha = 1.0 - confidence_level
    low, high = np.quantile(excess, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "low": float(low),
        "high": float(high),
        "confidence_level": confidence_level,
        "bootstrap_iterations": iterations,
        "resampling": "independent_within_component_a_component_b_and_mixture",
        "seed": seed,
    }


def analyze_bliss_joint_action(
    corrected_rows: Sequence[Dict[str, Any]],
    *,
    component_a_level: Any,
    component_b_level: Any,
    mixture_level: Any,
    expected_dose_series: Optional[Iterable[float]] = None,
    bootstrap_iterations: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: Optional[int] = None,
    ceiling_threshold: float = 99.5,
) -> Dict[str, Any]:
    """Analyse explicitly declared component A, component B and mixture roles."""

    levels = list(map(str, [component_a_level, component_b_level, mixture_level]))
    if len(set(levels)) != 3:
        raise CotoxicityValidationError(
            "component_a_level, component_b_level and mixture_level must be distinct."
        )
    if not 0 <= ceiling_threshold <= 100:
        raise CotoxicityValidationError("ceiling_threshold must be between 0 and 100.")

    frame = pd.DataFrame(corrected_rows)
    required = {
        "treatment", "dose", "replicate", "observation_time", "time_unit",
        "display_abbott_value", "abbott_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CotoxicityValidationError(f"Corrected mortality fields missing: {missing}")
    frame = frame[frame.treatment.isin(levels)].copy()
    available = set(frame.treatment)
    missing_levels = sorted(set(levels) - available)
    if missing_levels:
        raise CotoxicityValidationError(f"Declared joint-action levels not found: {missing_levels}")
    if frame.display_abbott_value.isna().any():
        raise CotoxicityValidationError("Joint-action rows require calculated Abbott mortality.")
    corrected = frame.display_abbott_value.astype(float)
    if ((corrected < 0) | (corrected > 100)).any():
        raise CotoxicityValidationError(
            "Bliss analysis requires corrected mortality on a 0–100 scale. Apply the explicitly "
            "declared zero-floor policy before joint-action analysis when Abbott values are negative."
        )

    if expected_dose_series is None:
        doses = sorted(float(value) for value in frame.dose.unique())
    else:
        doses = [float(value) for value in expected_dose_series]
    times = sorted(
        {(float(row.observation_time), str(row.time_unit)) for row in frame.itertuples()},
        key=lambda item: (item[1], item[0]),
    )

    cells: List[Dict[str, Any]] = []
    summary_counts: Dict[tuple[float, str], Dict[str, int]] = defaultdict(
        lambda: {
            "number_of_matched_doses": 0,
            "number_positive": 0,
            "number_additive": 0,
            "number_negative": 0,
            "number_supporting_synergy": 0,
            "number_supporting_antagonism": 0,
            "number_inconclusive": 0,
            "number_ceiling_limited": 0,
        }
    )

    for time, unit in times:
        for dose in doses:
            subset = frame[(frame.observation_time.astype(float) == time) & (frame.dose.astype(float) == dose)]
            role_values = {
                level: subset.loc[subset.treatment == level, "display_abbott_value"].astype(float).to_numpy()
                for level in levels
            }
            missing_roles = [level for level, values in role_values.items() if len(values) == 0]
            if missing_roles:
                cells.append(
                    {
                        "dose": dose,
                        "observation_time": time,
                        "time_unit": unit,
                        "available": False,
                        "missing_roles": missing_roles,
                        "warnings": [
                            "Bliss comparison unavailable because the same dose/time cell is missing: "
                            + ", ".join(missing_roles)
                        ],
                    }
                )
                continue

            a_values, b_values, mixture_values = (role_values[level] for level in levels)
            a_mean = float(a_values.mean())
            b_mean = float(b_values.mean())
            observed = float(mixture_values.mean())
            expected = float(bliss_expected(a_mean, b_mean))
            excess = observed - expected
            ratio = observed / expected if expected > 0 else None
            if excess > 1e-12:
                direction = "positive_deviation"
            elif excess < -1e-12:
                direction = "negative_deviation"
            else:
                direction = "additive_or_equal"

            # Derive a deterministic but distinct stream per sorted cell.
            cell_seed = None if bootstrap_seed is None else int(bootstrap_seed + len(cells))
            ci = bootstrap_bliss_excess(
                a_values,
                b_values,
                mixture_values,
                iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                seed=cell_seed,
            )
            ceiling = expected >= ceiling_threshold
            cell_warnings: List[str] = []
            if ceiling:
                inference = "ceiling_limited"
                cell_warnings.append(
                    "Expected mortality is at or near 100%; this dose/time cannot meaningfully "
                    "distinguish synergy from additivity."
                )
            elif ci["low"] > 0:
                inference = "supports_synergy_under_bliss"
            elif ci["high"] < 0:
                inference = "supports_antagonism_under_bliss"
            else:
                inference = "not_distinguishable_from_additivity"

            cell = {
                "dose": dose,
                "observation_time": time,
                "time_unit": unit,
                "available": True,
                "component_a": {
                    "level": levels[0], "n": len(a_values),
                    "mean_corrected_mortality": a_mean,
                },
                "component_b": {
                    "level": levels[1], "n": len(b_values),
                    "mean_corrected_mortality": b_mean,
                },
                "mixture": {
                    "level": levels[2], "n": len(mixture_values),
                    "mean_corrected_mortality": observed,
                },
                "bliss_expected": expected,
                "excess_observed_minus_expected": excess,
                "observed_expected_ratio": ratio,
                "bootstrap_ci": ci,
                "descriptive_direction": direction,
                "inference": inference,
                "ceiling_effect": ceiling,
                "warnings": cell_warnings,
            }
            cells.append(cell)

            counts = summary_counts[(time, unit)]
            counts["number_of_matched_doses"] += 1
            counts[
                "number_positive" if direction == "positive_deviation"
                else "number_negative" if direction == "negative_deviation"
                else "number_additive"
            ] += 1
            if inference == "supports_synergy_under_bliss":
                counts["number_supporting_synergy"] += 1
            elif inference == "supports_antagonism_under_bliss":
                counts["number_supporting_antagonism"] += 1
            elif inference == "ceiling_limited":
                counts["number_ceiling_limited"] += 1
            else:
                counts["number_inconclusive"] += 1

    summaries = [
        {"observation_time": time, "time_unit": unit, **counts}
        for (time, unit), counts in sorted(summary_counts.items())
    ]
    return {
        "roles": {
            "component_a_level": levels[0],
            "component_b_level": levels[1],
            "mixture_level": levels[2],
            "role_assignment": "explicit",
        },
        "cells": cells,
        "time_summaries": summaries,
        "provenance": {
            "input_scale": "abbott_corrected_percentage",
            "expected_model": "Bliss independence: A + B - AB/100",
            "bootstrap": "independent nonparametric resampling within each treatment role",
            "ceiling_threshold": ceiling_threshold,
        },
    }
