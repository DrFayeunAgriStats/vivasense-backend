"""VivaSense field layout generator routes.

Exposes POST /field-layout/generate for all supported designs.
Pro gate is enforced here — the engine itself (field_layout_generator.py)
has no knowledge of users or entitlements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from field_layout_generator import DESIGN_REGISTRY, generate_field_layout

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Field Layout"])


class FieldLayoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    design_type: str
    treatments: Optional[List[str]] = None
    replications: Optional[int] = None
    block_size: Optional[int] = None
    main_treatments: Optional[List[str]] = None
    sub_treatments: Optional[List[str]] = None
    sub_sub_treatments: Optional[List[str]] = None
    factors: Optional[Dict[str, List[str]]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    plot_width_m: Optional[float] = 5.0
    plot_length_m: Optional[float] = 10.0
    aisle_width_m: Optional[float] = 1.0
    seed: Optional[int] = 42


class FieldLayoutResponse(BaseModel):
    # Standardized response fields
    design: str
    seed: int
    treatments: List[str]
    replications: Optional[int] = None
    total_plots: int
    layout_matrix: List[Any]
    plot_labels: List[str]
    timestamp: str
    export_data: Dict[str, Any]

    # Backward-compatible fields
    design_type: str
    plot_matrix: List[Any]
    fieldbook: List[Dict[str, Any]]
    layout_summary: Dict[str, Any]
    alpha_value: Optional[float] = None


@router.get("/field-layout/designs")
async def list_designs():
    """Return all supported designs with their Pro requirements."""
    return {
        "designs": [
            {
                "design_type": key,
                "requires_pro": config["requires_pro"],
                "label": key.replace("_", " ").title(),
            }
            for key, config in DESIGN_REGISTRY.items()
        ]
    }


@router.post("/field-layout/generate", response_model=FieldLayoutResponse)
async def generate_layout(
    payload: Dict[str, Any],
    http_request: Request,
):
    """Generate a field layout for the requested design.

    Free users may access CRD and RCBD only.
    Pro users may access all designs including Latin square,
    split plot, split-split plot, factorial RCBD,
    balanced lattice, and alpha lattice.
    """
    try:
        validated_payload = FieldLayoutRequest.model_validate(payload)
    except ValidationError as exc:
        issues: List[str] = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            issues.append(f"{loc}: {msg}" if loc else msg)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "malformed_payload",
                "message": "Malformed request payload. Fix the listed fields and retry.",
                "issues": issues,
            },
        ) from exc

    design_type = (validated_payload.design_type or "").strip().lower()

    # Resolve design config from registry
    design_config = DESIGN_REGISTRY.get(design_type)
    if design_config is None:
        supported = sorted(DESIGN_REGISTRY.keys())
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_design",
                "message": f"Unknown design type '{design_type}'.",
                "supported_designs": supported,
            },
        )

    # Pro gate — checked at route level, never inside the engine
    if design_config["requires_pro"]:
        vivasense_mode = http_request.headers.get("X-VivaSense-Mode", "free")
        if vivasense_mode != "pro":
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "pro_required",
                    "message": (
                        f"The '{design_type}' layout design requires VivaSense Pro. "
                        "Upgrade to access Latin square, split plot, factorial RCBD, "
                        "balanced lattice, and alpha lattice designs."
                    ),
                    "upgrade_url": "/pricing",
                },
            )

    # Build engine request from payload
    engine_request: Dict[str, Any] = validated_payload.model_dump(exclude_none=True)

    # Call engine — ValueError means invalid user input, not a server error
    try:
        result = generate_field_layout(engine_request)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_failed",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error in field layout generation for design '%s': %s",
            design_type,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "generation_failed",
                "message": "An unexpected error occurred during layout generation. "
                           "Please check your input parameters and try again.",
            },
        ) from exc

    logger.info(
        "Field layout generated successfully:\n"
        "design=%s\n"
        "treatments=%s\n"
        "replications=%s\n"
        "plots=%s\n"
        "seed=%s",
        result.get("design", design_type).upper(),
        len(result.get("treatments", [])),
        result.get("replications"),
        result.get("total_plots", result["layout_summary"].get("n_plots")),
        result.get("seed"),
    )

    return FieldLayoutResponse(**result)
