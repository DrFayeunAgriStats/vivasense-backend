"""Backend orchestration for Crop Protection Bioassay / Efficacy Analysis."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from crop_protection_cotoxicity import analyze_bliss_joint_action
from crop_protection_factorial import analyze_factorial_crd
from crop_protection_mortality import (
    MortalityResponseMapping,
    prepare_mortality_responses,
    validate_cumulative_mortality,
)
from crop_protection_schemas import BioassayAnalysisRequest
from crop_protection_validation import BioassayValidationError, validate_bioassay_dataframe, warning


class UnsupportedBioassayAnalysis(NotImplementedError):
    pass


def _anova_map(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["source"]: row for row in result["anova"]}


def _diagnostics(
    df: pd.DataFrame, factors: List[str], response: str, response_id: str
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    work = df[[*factors, response]].copy()
    work[response] = pd.to_numeric(work[response], errors="coerce")
    cell_mean = work.groupby(factors, observed=True)[response].transform("mean")
    residuals = (work[response] - cell_mean).to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    shapiro = None
    if 3 <= len(residuals) <= 5000:
        statistic, p_value = stats.shapiro(residuals)
        shapiro = {"test": "Shapiro-Wilk", "statistic": float(statistic),
                   "p_value": float(p_value), "passed": bool(p_value >= 0.05)}
    groups = [
        group[response].to_numpy(dtype=float)
        for _, group in work.groupby(factors, observed=True)
        if len(group) >= 2
    ]
    levene = None
    if len(groups) >= 2:
        statistic, p_value = stats.levene(*groups, center="median")
        levene = {"test": "Levene (median-centered)",
                  "grouping": " × ".join(factors) + " cell",
                  "statistic": float(statistic), "p_value": float(p_value),
                  "passed": bool(p_value >= 0.05)}
    warnings = []
    if shapiro and not shapiro["passed"]:
        warnings.append(warning(
            "residual_non_normality",
            "Residual normality remains questionable; interpret conventional ANOVA cautiously.",
            response_id=response_id, details=shapiro,
        ))
    if levene and not levene["passed"]:
        warnings.append(warning(
            "variance_heterogeneity",
            "Variance heterogeneity was detected across Treatment × Dose cells.",
            response_id=response_id, details=levene,
        ))
    return {"residual_normality": shapiro, "homogeneity": levene}, warnings


def _correlations(
    df: pd.DataFrame, request: BioassayAnalysisRequest, response_by_id: Dict[str, Any]
) -> List[Dict[str, Any]]:
    design = request.design
    treatment = next((factor for factor in design.factor_columns
                      if factor.semantic_role == "treatment"), design.factor_columns[0])
    treated = (df if design.control_treatment_level is None else
               df[df[treatment.column].astype(str) != design.control_treatment_level])
    output = []
    ids = request.correlation_response_ids
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1:]:
            left = response_by_id[left_id]
            right = response_by_id[right_id]
            left_col = left.display_column or left.raw_column
            right_col = right.display_column or right.raw_column
            pair = treated[[left_col, right_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair[left_col].nunique() < 2 or pair[right_col].nunique() < 2:
                output.append({"response_a": left_id, "response_b": right_id, "n": len(pair),
                               "r": None, "p_value": None, "status": "insufficient_variation"})
                continue
            r_value, p_value = stats.pearsonr(pair[left_col], pair[right_col])
            output.append({"response_a": left_id, "response_b": right_id, "n": len(pair),
                           "r": float(r_value), "p_value": float(p_value),
                           "status": "success", "scale": "raw_display", "population": "treated_only"})
    return output


def _regressions(
    df: pd.DataFrame, request: BioassayAnalysisRequest, response_by_id: Dict[str, Any]
) -> List[Dict[str, Any]]:
    design = request.design
    factors = design.factor_columns
    factor_columns = [factor.column for factor in factors]
    factor_labels = [factor.display_name or factor.column for factor in factors]
    treatment_factor = next((factor for factor in factors if factor.semantic_role == "treatment"), factors[0])
    dose_factor = next((factor for factor in factors if factor.id == design.dose_factor_id), None)
    treatment_column = treatment_factor.column
    dose_column = dose_factor.column if dose_factor else None
    treated = df[df[design.treatment_column].astype(str) != design.control_treatment_level]
    output = []
    for response_id in request.regression_response_ids:
        response = response_by_id[response_id]
        column = response.display_column or response.raw_column
        for treatment, group in treated.groupby(design.treatment_column, sort=False):
            pair = group[[design.dose_column, column]].apply(pd.to_numeric, errors="coerce").dropna()
            base = {"treatment": str(treatment), "response_id": response_id, "n": len(pair),
                    "scale": "raw_display", "control_included": False}
            if len(pair) < 3 or pair[design.dose_column].nunique() < 2:
                output.append({**base, "status": "insufficient_dose_variation", "intercept": None,
                               "slope": None, "r_squared": None, "p_value": None,
                               "significance": "unavailable", "direction": "unavailable"})
            elif pair[column].nunique() < 2:
                output.append({**base, "status": "constant_response", "intercept": float(pair[column].iloc[0]),
                               "slope": 0.0, "r_squared": None, "p_value": None,
                               "significance": "unavailable", "direction": "constant"})
            else:
                fit = stats.linregress(pair[design.dose_column], pair[column])
                output.append({**base, "status": "success", "intercept": float(fit.intercept),
                               "slope": float(fit.slope), "r_squared": float(fit.rvalue ** 2),
                               "p_value": float(fit.pvalue),
                               "significance": "significant" if fit.pvalue < request.options.alpha else "not_significant",
                               "direction": "increasing" if fit.slope > 0 else "decreasing" if fit.slope < 0 else "constant"})
    return output


def _interpretation_priority(anova_rows, factor_count, alpha):
    interactions = [row for row in anova_rows if " × " in row["source"] or ":" in row["source"]]
    order_of = lambda source: max(source.count(" × "), source.count(":")) + 1
    significant_highest = [row for row in interactions if order_of(row["source"]) == factor_count
                           and row["p_value"] is not None and row["p_value"] < alpha]
    significant_two_way = [row for row in interactions if order_of(row["source"]) == 2
                           and row["p_value"] is not None and row["p_value"] < alpha]
    if factor_count == 3 and significant_highest:
        priority = "three_way_interaction"
    elif significant_two_way:
        priority = "two_way_interaction" if factor_count == 3 else "interaction"
    else:
        priority = "main_effects"
    return priority, significant_highest, significant_two_way


def orchestrate_bioassay(
    df: pd.DataFrame, request: BioassayAnalysisRequest
) -> Dict[str, Any]:
    validation = validate_bioassay_dataframe(df, request)
    design = request.design
    factors = design.factor_columns
    factor_columns = [factor.column for factor in factors]
    factor_labels = [factor.display_name or factor.column for factor in factors]
    treatment_factor = next((factor for factor in factors if factor.semantic_role == "treatment"), factors[0])
    dose_factor = next((factor for factor in factors if factor.id == design.dose_factor_id), None)
    treatment_column = treatment_factor.column
    dose_column = dose_factor.column if dose_factor else None
    response_by_id = {response.id: response for response in request.responses}
    warnings: List[Dict[str, Any]] = []

    mortality_definitions = [response for response in request.responses if response.type == "mortality"]
    mortality = None
    mortality_by_time = {}
    if mortality_definitions:
        if design.control_treatment_level is None or dose_column is None:
            raise BioassayValidationError(
                "Mortality correction requires an explicit control and a factor with semantic role 'dose'."
            )
        mappings = [
            MortalityResponseMapping(
                raw_column=response.raw_column,
                observation_time=float(response.observation_time),
                time_unit=str(response.time_unit),
                transformed_column=response.transformed_column or (
                    response.inference_column if response.inference_column != response.raw_column else None
                ),
                corrected_column=response.corrected_column,
            )
            for response in mortality_definitions
        ]
        mortality = prepare_mortality_responses(
            df,
            treatment_column=treatment_column,
            dose_column=dose_column,
            replicate_column=design.replicate_column,
            control_level=design.control_treatment_level,
            mortality_responses=mappings,
            floor_at_zero=request.options.floor_abbott_at_zero,
            control_policy=request.options.control_policy,
            control_row_indices=request.options.control_row_indices,
        )
        mortality_by_time = {
            (item["observation_time"], item["time_unit"]): item for item in mortality["responses"]
        }
        if mortality["control"]["repeated_control_blocks_detected"]:
            warnings.append(warning(
                "repeated_control_blocks", "Repeated control blocks were detected and resolved by the declared policy.",
                details=mortality["control"],
            ))
        threshold = request.options.high_control_mortality_warning_threshold
        for item in mortality["responses"]:
            if threshold is not None and item["control_mean_raw_mortality"] >= threshold:
                warnings.append(warning(
                    "high_control_mortality",
                    "Control mortality met or exceeded the researcher-declared warning threshold.",
                    response_id=next(r.id for r in mortality_definitions if r.observation_time == item["observation_time"]),
                    details={"observed": item["control_mean_raw_mortality"], "threshold": threshold},
                ))
            if any(row["floor_applied"] for row in item["rows"]):
                warnings.append(warning(
                    "abbott_floor_applied", "Negative Abbott values were floored at zero under the declared policy.",
                    details={"observation_time": item["observation_time"]},
                ))

    response_results = []
    first_design = None
    interpretation_by_response = {}
    for response in request.responses:
        analysis_kwargs = dict(
            replicate_column=design.replicate_column,
            response_column=response.inference_column,
            display_column=((response.display_column or response.raw_column)
                            if (response.display_column or response.raw_column) != response.inference_column else None),
            control_level=design.control_treatment_level,
            control_column=treatment_column,
            expected_dose_series=design.expected_dose_series,
        )
        if design.treatment_column and design.dose_column:
            result = analyze_factorial_crd(
                df, treatment_column=design.treatment_column, dose_column=design.dose_column,
                **analysis_kwargs,
            )
        else:
            result = analyze_factorial_crd(
                df, factor_columns=factor_columns, factor_display_names=factor_labels,
                **analysis_kwargs,
            )
        if first_design is None:
            first_design = result["design"]
        treated = df.loc[result["factorial_rows_used"]]
        diagnostics, diagnostic_warnings = _diagnostics(
            treated, factor_columns, response.inference_column, response.id,
        )
        warnings.extend(diagnostic_warnings)
        if not result["design"]["balanced"]:
            warnings.append(warning(
                "unequal_cell_replication", "Treatment × Dose cells have unequal replication.",
                response_id=response.id,
            ))
        anova = _anova_map(result)
        priority, significant_highest, significant_two_way = _interpretation_priority(
            result["anova"], len(factors), request.options.alpha
        )
        interaction_significant = bool(significant_highest or significant_two_way)
        mortality_details = None
        if response.type == "mortality":
            prepared = mortality_by_time[(float(response.observation_time), str(response.time_unit))]
            mortality_details = {
                "abbott_applied": response.abbott_correction,
                "scales": {
                    "raw": "percent_mortality",
                    "inference": (
                        "explicit_transformed_column"
                        if response.inference_column != response.raw_column
                        else "percent_mortality"
                    ),
                    "corrected": "abbott_percent" if response.abbott_correction else None,
                },
                "control_n": mortality["control"]["n_control"],
                "control_mean_raw_mortality": prepared["control_mean_raw_mortality"],
                "control_policy": mortality["control"]["selection_rule"],
                "duplicates_removed": mortality["control"]["duplicates_removed"],
                "rows": prepared["rows"] if response.abbott_correction else [],
                "verification": prepared["supplied_correction_verification"],
                "warnings": mortality["warnings"],
            }
        facts = {
            "treatment_significant": bool(anova.get(factor_labels[0], {}).get("p_value", 1) < request.options.alpha),
            "dose_significant": bool(
                len(factor_labels) > 1 and
                anova.get(factor_labels[1], {}).get("p_value", 1) < request.options.alpha
            ),
            "main_effect_significance": {
                label: bool(anova.get(label, {}).get("p_value", 1) < request.options.alpha)
                for label in factor_labels
            },
            "interaction_significant": interaction_significant,
            "significant_two_way_interactions": [row["source"] for row in significant_two_way],
            "interpretation_priority": priority,
        }
        interpretation_by_response[response.id] = facts
        response_results.append({
            "response_id": response.id,
            "biological_type": response.type,
            "anova": result["anova"],
            "interaction": result["interaction"],
            "primary_mean_separation": result["interaction"]["means"] if priority != "main_effects" else result["marginal_means"],
            "cell_means": result["cell_means"],
            "marginal_means": result["marginal_means"],
            "treatment_marginal_means": result["marginal_means"].get("treatment", result["marginal_means"].get(factor_labels[0], [])),
            "dose_marginal_means": result["marginal_means"].get("dose", result["marginal_means"].get(factor_labels[1], []) if len(factors)>1 else []),
            "diagnostics": diagnostics,
            "mortality_correction": mortality_details,
            "interpretation_metadata": facts,
            "provenance": {
                "response_id": response.id,
                "biological_type": response.type,
                "raw_column": response.raw_column,
                "inference_column": response.inference_column,
                "display_column": response.display_column or response.raw_column,
                "transformation": "explicit" if response.inference_column != response.raw_column else "none",
                "abbott_applied": response.abbott_correction,
                "control_rows_used": mortality["control"]["control_rows_used"] if mortality else result["rows_excluded"],
                "factorial_rows_used": result["factorial_rows_used"],
                "excluded_rows": result["rows_excluded"],
                "alpha": request.options.alpha,
                "software_engine": "VivaSense crop-protection factorial CRD adapter (R aov/agricolae)",
            },
        })

    control_mask = (df[treatment_column].astype(str) == design.control_treatment_level
                    if design.control_treatment_level is not None else pd.Series(False, index=df.index))
    control_n = mortality["control"]["n_control"] if mortality else int(control_mask.sum())
    control_rows = mortality["control"]["control_rows_used"] if mortality else df.index[control_mask].tolist()
    cell_counts = df[~control_mask].groupby(factor_columns, observed=True).size()
    design_result = {
        "design_type": "factorial_crd",
        "total_rows": len(df),
        "factorial_rows": first_design["factorial_n"],
        "control_rows": control_n,
        "factor_count": len(factors),
        "factors": first_design.get("factors", [
            {"column": factor.column, "display_name": factor.display_name or factor.column,
             "levels": int(df.loc[~control_mask, factor.column].nunique())}
            for factor in factors
        ]),
        "factorial_treatments": df.loc[~control_mask, treatment_column].astype(str).drop_duplicates().tolist(),
        "dose_levels": design.expected_dose_series,
        "cells": int(len(cell_counts)),
        "balanced": bool(cell_counts.nunique() == 1),
        "cell_replication": int(cell_counts.iloc[0]) if cell_counts.nunique() == 1 else None,
        "cell_counts": [
            {"factor_levels": dict(zip(factor_labels, key if isinstance(key, tuple) else (key,))), "n": int(value)}
            for key, value in cell_counts.items()
        ],
        "replicate_role": "experimental_unit_identifier",
        "control_rows_used": control_rows,
    }

    cumulative_result = None
    cumulative_definitions = [response for response in mortality_definitions if response.cumulative]
    if len(cumulative_definitions) >= 2:
        selected = {(float(r.observation_time), str(r.time_unit)) for r in cumulative_definitions}
        cumulative_rows = [
            row for row in mortality["long_rows"]
            if (float(row["observation_time"]), str(row["time_unit"])) in selected
        ]
        cumulative_result = validate_cumulative_mortality(cumulative_rows, cumulative=True)
        for decrease in cumulative_result["decreases"]:
            warnings.append(warning(
                "non_monotonic_cumulative_mortality",
                "Raw cumulative mortality decreased between successive observation times.",
                details=decrease,
            ))

    cotoxicity_result = None
    if request.cotoxicity and request.cotoxicity.enabled:
        if len(factors) > 2:
            raise UnsupportedBioassayAnalysis("not_supported_for_multifactor_cotoxicity")
        config = request.cotoxicity
        if config.method.lower() != "bliss":
            raise UnsupportedBioassayAnalysis(
                f"Co-toxicity method {config.method!r} is not implemented; Bliss was not substituted."
            )
        selected_responses = [response_by_id[rid] for rid in config.response_ids]
        if any(response.type != "mortality" or not response.abbott_correction for response in selected_responses):
            raise BioassayValidationError(
                "Co-toxicity response IDs must reference mortality responses with Abbott correction enabled."
            )
        selected_times = {(float(r.observation_time), str(r.time_unit)) for r in selected_responses}
        corrected_rows = [
            row for row in mortality["long_rows"]
            if (float(row["observation_time"]), str(row["time_unit"])) in selected_times
        ]
        raw_cotoxicity = analyze_bliss_joint_action(
            corrected_rows,
            component_a_level=config.component_a_level,
            component_b_level=config.component_b_level,
            mixture_level=config.mixture_level,
            expected_dose_series=design.expected_dose_series,
            bootstrap_iterations=config.bootstrap_iterations,
            confidence_level=config.confidence_level,
            bootstrap_seed=config.seed,
            ceiling_threshold=config.ceiling_threshold,
        )
        by_time = []
        for summary in raw_cotoxicity["time_summaries"]:
            time = summary["observation_time"]
            cells = [cell for cell in raw_cotoxicity["cells"] if cell["observation_time"] == time]
            by_time.append({"observation_time": time, "time_unit": summary["time_unit"],
                            "cells": cells, "summary": summary})
            for cell in cells:
                if cell.get("ceiling_effect"):
                    warnings.append(warning(
                        "co_toxicity_ceiling_effect", cell["warnings"][0],
                        details={"dose": cell["dose"], "observation_time": time},
                    ))
                if not cell.get("available", True):
                    warnings.append(warning(
                        "co_toxicity_missing_matched_dose", cell["warnings"][0],
                        details={"dose": cell["dose"], "observation_time": time},
                    ))
        cotoxicity_result = {
            "method": "bliss_independence",
            "component_a": config.component_a_level,
            "component_b": config.component_b_level,
            "mixture": config.mixture_level,
            "by_time": by_time,
            "provenance": {**raw_cotoxicity["provenance"],
                           "bootstrap_iterations": config.bootstrap_iterations,
                           "confidence_level": config.confidence_level, "seed": config.seed},
        }

    if request.regression_response_ids and (dose_factor is None or len(factors) > 2):
        raise UnsupportedBioassayAnalysis("Dose-response regression is not supported for multifactor pooling.")
    regressions = _regressions(df, request, response_by_id) if dose_factor else []
    correlations = _correlations(df, request, response_by_id)
    return {
        "status": "success",
        "analysis_type": "crop_protection_bioassay",
        "design": design_result,
        "warnings": warnings,
        "response_results": response_results,
        "cotoxicity": cotoxicity_result,
        "regression": regressions,
        "correlation": correlations,
        "cumulative_mortality_validation": cumulative_result,
        "interpretation_metadata": {
            "by_response": interpretation_by_response,
            "dose_response_direction": {
                f"{row['response_id']}:{row['treatment']}": row["direction"] for row in regressions
            },
            "control_warning_present": any(w["code"] == "high_control_mortality" for w in warnings),
            "assumption_warning_present": any(
                w["code"] in {"residual_non_normality", "variance_heterogeneity"} for w in warnings
            ),
        },
        "result_order": [
            "design", "warnings", "response_results.anova", "response_results.primary_mean_separation",
            "response_results.marginal_means", "response_results.mortality_correction", "cotoxicity",
            "regression", "correlation", "response_results.diagnostics", "interpretation_metadata",
        ],
        "provenance": {
            "validation": validation,
            "request_contract": "crop_protection_bioassay_phase_3",
            "statistical_services": ["factorial_crd", "abbott", "bliss_independence"],
        },
    }
