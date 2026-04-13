"""
VivaSense – ANOVA Analysis Module

POST /analysis/anova

Returns per-trait:
  • ANOVA table (source, df, SS, MS, F, p)
  • Descriptive statistics
  • Assumption tests (Shapiro-Wilk, Levene — if available from R)
  • Mean separation (Tukey HSD or LSD)
  • Interpretation text
  • Data warnings (balance / completeness issues)

The R engine (vivasense_genetics.R) performs all computation.
Python re-uses the dataset context stored by POST /upload/dataset,
builds flat observation records, dispatches to R, and slices the result.

Analysis results are cached in dataset_cache so that a subsequent call
to /analysis/genetic-parameters for the same dataset_token + traits
does not trigger a second R subprocess call.
"""

import base64
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from genetics_schemas import GeneticsResponse
from multitrait_upload_routes import build_observations, check_balance, read_file
from module_schemas import AnovaModuleResponse, AnovaTraitResult, ModuleRequest
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


@router.post(
    "/analysis/anova",
    response_model=AnovaModuleResponse,
    summary="Run ANOVA analysis for selected traits",
)
async def analysis_anova(request: ModuleRequest):
    """
    For each requested trait column, run the R genetics engine and return
    the ANOVA-specific slice of the result:

      anova_table, descriptive_stats, assumption_tests,
      mean_separation, interpretation, data_warnings

    Requires a dataset_token issued by POST /upload/dataset.
    If another module already ran R for a trait under the same token,
    the result is read from cache — no duplicate R call.
    """
    import app_genetics  # lazy: r_engine assigned on startup
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

    mode         = ctx["mode"]
    env_col      = ctx["environment_column"] if mode == "multi" else None
    geno_col     = ctx["genotype_column"]
    rep_col      = ctx["rep_column"]
    random_env   = ctx["random_environment"]

    trait_results: Dict[str, AnovaTraitResult] = {}
    failed_traits: List[str] = []

    for trait in request.trait_columns:
        try:
            # ── Cache check ──────────────────────────────────────────────────
            cached: Optional[GeneticsResponse] = dataset_cache.get_analysis(
                request.dataset_token, trait
            )

            if cached is None:
                balance_warnings = check_balance(df, geno_col, rep_col, trait, env_col)
                observations     = build_observations(df, geno_col, rep_col, trait, env_col)

                result_dict = r_engine.run_analysis(
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

            trait_results[trait] = AnovaTraitResult(
                trait=trait,
                status="success",
                grand_mean=res.grand_mean,
                n_genotypes=res.n_genotypes,
                n_reps=res.n_reps,
                n_environments=res.n_environments,
                anova_table=res.anova_table,
                descriptive_stats=res.descriptive_stats,
                assumption_tests=res.assumption_tests,
                mean_separation=res.mean_separation,
                interpretation=cached.interpretation,
                data_warnings=balance_warnings,
            )

        except Exception as exc:
            logger.warning("ANOVA: trait '%s' failed — %s", trait, exc)
            failed_traits.append(trait)
            trait_results[trait] = AnovaTraitResult(
                trait=trait,
                status="failed",
                error=str(exc),
            )

    return AnovaModuleResponse(
        dataset_token=request.dataset_token,
        mode=mode,
        trait_results=trait_results,
        failed_traits=failed_traits,
    )
