"""API contracts for Crop Protection Bioassay / Efficacy Analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class BioassayDataset(BaseModel):
    base64_content: str
    file_type: Literal["csv", "xlsx", "xls"]


class BioassayDesign(BaseModel):
    design_type: Literal["crd"]
    treatment_column: str
    dose_column: str
    replicate_column: str
    control_treatment_level: str
    expected_dose_series: List[float] = Field(min_length=1)


class BioassayResponseDefinition(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["mortality", "count", "continuous"]
    raw_column: str
    inference_column: str
    display_column: Optional[str] = None
    observation_time: Optional[float] = None
    time_unit: Optional[str] = None
    transformed_column: Optional[str] = None
    corrected_column: Optional[str] = None
    abbott_correction: bool = False
    cumulative: bool = False

    @model_validator(mode="after")
    def validate_response_role(self):
        if self.type == "mortality" and (
            self.observation_time is None or not (self.time_unit or "").strip()
        ):
            raise ValueError("Mortality responses require explicit observation_time and time_unit.")
        if self.abbott_correction and self.type != "mortality":
            raise ValueError("Abbott correction is available only for mortality responses.")
        return self


class BioassayCotoxicity(BaseModel):
    enabled: bool = False
    method: str = "bliss"
    component_a_level: str
    component_b_level: str
    mixture_level: str
    response_ids: List[str] = Field(default_factory=list)
    bootstrap_iterations: int = Field(default=10_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    seed: Optional[int] = None
    ceiling_threshold: float = Field(default=99.5, ge=0, le=100)


class BioassayOptions(BaseModel):
    alpha: float = Field(default=0.05, gt=0, lt=1)
    floor_abbott_at_zero: bool = True
    control_policy: Literal[
        "require_unique", "deduplicate_identical_replicates"
    ] = "require_unique"
    control_row_indices: Optional[List[int]] = None
    high_control_mortality_warning_threshold: Optional[float] = Field(
        default=None, ge=0, le=100
    )


class BioassayAnalysisRequest(BaseModel):
    dataset: BioassayDataset
    design: BioassayDesign
    responses: List[BioassayResponseDefinition] = Field(min_length=1)
    cotoxicity: Optional[BioassayCotoxicity] = None
    correlation_response_ids: List[str] = Field(default_factory=list)
    regression_response_ids: List[str] = Field(default_factory=list)
    options: BioassayOptions = Field(default_factory=BioassayOptions)

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [response.id for response in self.responses]
        if len(ids) != len(set(ids)):
            raise ValueError("Response IDs must be unique.")
        known = set(ids)
        requested = set(self.correlation_response_ids) | set(self.regression_response_ids)
        if self.cotoxicity:
            requested |= set(self.cotoxicity.response_ids)
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown response IDs referenced: {unknown}")
        roles = [
            self.design.treatment_column,
            self.design.dose_column,
            self.design.replicate_column,
        ]
        if len(set(roles)) != len(roles):
            raise ValueError("Treatment, dose and replicate columns must be distinct.")
        return self


class BioassayAnalysisResponse(BaseModel):
    status: Literal["success"] = "success"
    analysis_type: Literal["crop_protection_bioassay"] = "crop_protection_bioassay"
    design: Dict[str, Any]
    warnings: List[Dict[str, Any]]
    response_results: List[Dict[str, Any]]
    cotoxicity: Optional[Dict[str, Any]] = None
    regression: List[Dict[str, Any]] = Field(default_factory=list)
    correlation: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_mortality_validation: Optional[Dict[str, Any]] = None
    interpretation_metadata: Dict[str, Any]
    result_order: List[str]
    provenance: Dict[str, Any]
