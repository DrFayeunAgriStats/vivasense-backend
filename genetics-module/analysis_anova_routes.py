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

import asyncio
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


def _build_anova_interpretation(trait: str, res) -> str:
    """Build a precise, ANOVA-only interpretation string (no genetic parameters)."""
    parts = []
    
    # 1. Experimental Structure
    n_g = getattr(res, "n_genotypes", None)
    n_e = getattr(res, "n_environments", None)
    
    if n_g is not None:
        if n_e is not None and n_e > 1:
            parts.append(f"An analysis of variance was conducted for {trait} evaluated across {n_g} genotypes and {n_e} environments.")
        else:
            parts.append(f"An analysis of variance was conducted for {trait} evaluated across {n_g} genotypes.")

    # 2. ANOVA Significance
    if res.anova_table and "genotype" in res.anova_table.source:
        try:
            idx = res.anova_table.source.index("genotype")
            p = res.anova_table.p_value[idx]
            f = res.anova_table.f_value[idx]
            if p is not None:
                sig = "significant" if p < 0.05 else "not significant"
                p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
                f_str = f"{f:.3f}" if f is not None else "—"
                parts.append(f"The effect of genotype on {trait} was {sig} in this experiment (F = {f_str}, p = {p_str}).")
        except (ValueError, IndexError):
            pass
            
    # 3. Mean Separation
    if res.mean_separation and res.mean_separation.genotype:
        try:
            top_g = res.mean_separation.genotype[0]
            top_m = res.mean_separation.mean[0]
            bot_g = res.mean_separation.genotype[-1]
            bot_m = res.mean_separation.mean[-1]
            parts.append(f"The highest mean among the genotypes tested was recorded in {top_g} ({top_m:.2f}), while the lowest was recorded in {bot_g} ({bot_m:.2f}).")
        except (IndexError, TypeError):
            pass
            
    return " ".join(parts) if parts else "ANOVA interpretation not available."


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

            result_obj = AnovaTraitResult(
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
                interpretation=_build_anova_interpretation(trait, res),
                data_warnings=balance_warnings,
            )
            return trait, "success", result_obj

        except Exception as exc:
            logger.warning("ANOVA: trait '%s' failed — %s", trait, exc)
            result_obj = AnovaTraitResult(
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

    return AnovaModuleResponse(
        dataset_token=request.dataset_token,
        mode=mode,
        trait_results=trait_results,
        failed_traits=failed_traits,
    )
