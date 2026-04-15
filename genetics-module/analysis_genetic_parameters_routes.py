"""
VivaSense – Genetic Parameters Analysis Module

POST /analysis/genetic-parameters

Returns per-trait:
  • Variance components (σ²g, σ²e, σ²ge, σ²p)
  • Broad-sense heritability (H²) + heritability class
  • GCV, PCV
  • GA (Genetic Advance), GAM (GA as % of mean)
  • Breeding implication text
  • Interpretation text
  • Data warnings

The R engine (vivasense_genetics.R) performs all computation.
If /analysis/anova was already called for the same dataset_token + trait,
the result is read from dataset_cache — no duplicate R subprocess call.
"""

import asyncio
import base64
import logging
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from genetics_schemas import GeneticsResponse
from multitrait_upload_routes import build_observations, check_balance, read_file
from module_schemas import (
    GeneticParametersModuleResponse,
    GeneticParametersTraitResult,
    ModuleRequest,
)
import dataset_cache
from interpretation import InterpretationEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


def _build_gp_text(trait: str, res) -> Tuple[str, str]:
    """Build precise Genetic Parameters text, completely decoupled from ANOVA."""
    try:
        hp = res.heritability if isinstance(res.heritability, dict) else {}
        gp = res.genetic_parameters if isinstance(res.genetic_parameters, dict) else {}
        
        h2_val = hp.get("h2_broad_sense")
        
        engine = InterpretationEngine()
        support = engine.generate_decision_support(
            trait_name=trait,
            h2=float(h2_val) if h2_val is not None else 0.0,
            gam=float(gp.get("GAM_percent") or 0.0),
            gcv=float(gp.get("GCV") or 0.0),
            pcv=float(gp.get("PCV") or 0.0)
        )
        return support["interpretation"], support["recommendation"]
    except Exception as exc:
        logger.warning("Failed to build GP interpretation for '%s': %s", trait, exc)
        return "Genetic parameters interpretation not available.", "Breeding implication not available."

@router.post(
    "/analysis/genetic-parameters",
    response_model=GeneticParametersModuleResponse,
    summary="Run Genetic Parameters analysis for selected traits",
)
async def analysis_genetic_parameters(request: ModuleRequest):
    """
    For each requested trait column, run the R genetics engine and return
    the genetic-parameters slice of the result:

      variance_components, heritability, GCV, PCV, GA, GAM,
      breeding_implication, interpretation, data_warnings

    Requires a dataset_token from POST /upload/dataset.
    Reuses cached R results from any prior /analysis/* call for the same
    dataset_token + trait to avoid duplicate computation.
    """
    import app_genetics
    if app_genetics.r_engine is None:
        raise HTTPException(status_code=503, detail="R genetics engine not ready")
    r_engine = app_genetics.r_engine

    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset token '{request.dataset_token}' not found. "
                "Re-upload via POST /upload/dataset to get a new token."
            ),
        )

    try:
        file_bytes = base64.b64decode(ctx["base64_content"])
        df = read_file(file_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc

    missing = [c for c in request.trait_columns if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Trait columns not found in dataset: {missing}"
        )

    mode       = ctx["mode"]
    env_col    = ctx["environment_column"] if mode == "multi" else None
    geno_col   = ctx["genotype_column"]
    rep_col    = ctx["rep_column"]
    random_env = ctx["random_environment"]

    trait_results: Dict[str, GeneticParametersTraitResult] = {}
    failed_traits: List[str] = []

    MAX_CONCURRENT_R_PROCESSES = 4
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_R_PROCESSES)

    async def process_trait(trait: str):
        try:
            # ── Cache check ──────────────────────────────────────────────────
            cached: Optional[GeneticsResponse] = dataset_cache.get_analysis(
                request.dataset_token, trait
            )

            if cached is None:
                async with semaphore:
                    balance_warnings = check_balance(df, geno_col, rep_col, trait, env_col)
                    observations     = build_observations(df, geno_col, rep_col, trait, env_col)

                    result_dict = await asyncio.to_thread(
                        r_engine.run_analysis,
                        data=observations,
                        mode=mode,
                        trait_name=trait,
                        random_environment=random_env,
                    )

                if result_dict.get("status") == "ERROR":
                    r_errors = result_dict.get("errors") or {}
                    raise RuntimeError(
                        result_dict.get("interpretation")
                        or next(iter(r_errors.values()), None)
                        or "R analysis returned ERROR"
                    )

                cached = GeneticsResponse(**result_dict)
                dataset_cache.put_analysis(request.dataset_token, trait, cached)
            else:
                balance_warnings = []

            res = cached.result
            if res is None:
                raise RuntimeError("R returned status OK but result object is empty")

            gp = res.genetic_parameters if isinstance(res.genetic_parameters, dict) else {}
            
            # Build pure Genetic Parameters text
            interp_text, breeding_text = _build_gp_text(trait, res)

            result_obj = GeneticParametersTraitResult(
                trait=trait,
                status="success",
                grand_mean=res.grand_mean,
                variance_components=res.variance_components
                    if isinstance(res.variance_components, dict) else {},
                heritability=res.heritability
                    if isinstance(res.heritability, dict) else {},
                gcv=gp.get("GCV"),
                pcv=gp.get("PCV"),
                ga=gp.get("GAM"),        # GA absolute value
                gam=gp.get("GAM_percent"),
                breeding_implication=breeding_text,
                interpretation=interp_text,
                data_warnings=balance_warnings,
            )
            return trait, "success", result_obj

        except Exception as exc:
            logger.warning("GeneticParameters: trait '%s' failed — %s", trait, exc)
            result_obj = GeneticParametersTraitResult(
                trait=trait,
                status="failed",
                error=str(exc),
            )
            return trait, "failed", result_obj

    # Execute all trait analyses concurrently
    tasks = [process_trait(trait) for trait in request.trait_columns]
    results = await asyncio.gather(*tasks)

    for trait, status, result_obj in results:
        trait_results[trait] = result_obj
        if status == "failed":
            failed_traits.append(trait)

    return GeneticParametersModuleResponse(
        dataset_token=request.dataset_token,
        mode=mode,
        trait_results=trait_results,
        failed_traits=failed_traits,
    )
