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
    DescriptiveStats,
    ModuleRequest,
)
from analysis_anova_routes import compute_descriptive_stats
import dataset_cache
from genetics_interpretation import generate_genetics_interpretation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


def _get_anova_flags(anova_table) -> Tuple[bool, bool]:
    """
    Return (environment_significant, gxe_significant) from an AnovaTable.
    Returns (False, False) when the table is absent or source is not listed.
    """
    if anova_table is None or not hasattr(anova_table, "source"):
        return False, False

    def _is_sig(source_name: str) -> bool:
        try:
            idx = anova_table.source.index(source_name)
            p = anova_table.p_value[idx]
            return p is not None and float(p) < 0.05
        except (ValueError, IndexError, TypeError):
            return False

    env_sig = _is_sig("environment")
    gxe_sig = any(_is_sig(t) for t in ["genotype:environment", "environment:genotype", "GxE", "gxe"])
    return env_sig, gxe_sig


def _build_gp_text(trait: str, res) -> Tuple[str, str, bool, bool]:
    """
    Build academic-grade genetic parameters interpretation text.
    Replaces legacy R engine output with validated interpretation.

    Returns:
        (interpretation_text, breeding_implication_text,
         environment_significant, gxe_significant)
    """
    try:
        hp = res.heritability if isinstance(res.heritability, dict) else {}
        gp = res.genetic_parameters if isinstance(res.genetic_parameters, dict) else {}

        h2_val  = hp.get("h2_broad_sense")
        gam_val = gp.get("GAM_percent")
        gcv_val = gp.get("GCV")
        pcv_val = gp.get("PCV")

        env_significant, gxe_significant = _get_anova_flags(res.anova_table)

        interpretation, breeding_implication = generate_genetics_interpretation(
            trait_name=trait,
            h2=float(h2_val) if h2_val is not None else None,
            gam=float(gam_val) if gam_val is not None else None,
            gcv=float(gcv_val) if gcv_val is not None else None,
            pcv=float(pcv_val) if pcv_val is not None else None,
            gxe_significant=gxe_significant,
            environment_significant=env_significant,
        )

        logger.info(
            "[genetic_parameters] generated interpretation for trait '%s': %d chars "
            "(env_sig=%s, gxe_sig=%s)",
            trait, len(interpretation), env_significant, gxe_significant,
        )

        return interpretation, breeding_implication, env_significant, gxe_significant
    except Exception as exc:
        logger.warning("Failed to build GP interpretation for '%s': %s", trait, exc)
        return (
            "Genetic parameters interpretation not available.",
            "Breeding implication not available.",
            False,
            False,
        )

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

    mode          = ctx["mode"]
    env_col       = ctx["environment_column"] if mode == "multi" else None
    geno_col      = ctx["genotype_column"]
    rep_col       = ctx["rep_column"]          # may be None for CRD datasets
    factor_col    = ctx.get("factor_column") if mode == "single" else None
    main_plot_col = ctx.get("main_plot_column")
    sub_plot_col  = ctx.get("sub_plot_column")
    design_type   = ctx.get("design_type")
    random_env    = ctx["random_environment"]
    crd_mode      = (rep_col is None) and (mode == "single")

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
                    balance_warnings = check_balance(
                        df,
                        geno_col,
                        rep_col,
                        trait,
                        env_col,
                        factor_col=factor_col,
                        design_type=design_type,
                        main_plot_col=main_plot_col,
                        sub_plot_col=sub_plot_col,
                    )
                    observations = build_observations(
                        df,
                        geno_col,
                        rep_col,
                        trait,
                        env_col,
                        factor_col=factor_col,
                        design_type=design_type,
                        main_plot_col=main_plot_col,
                        sub_plot_col=sub_plot_col,
                    )

                    result_dict = await asyncio.to_thread(
                        r_engine.run_analysis,
                        data=observations,
                        mode=mode,
                        trait_name=trait,
                        random_environment=random_env,
                        crd_mode=crd_mode,
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

            # Compute descriptive statistics from the trait data
            trait_descriptive_stats = compute_descriptive_stats(df[trait])
            desc_stats_obj = DescriptiveStats(
                grand_mean=trait_descriptive_stats.get("grand_mean"),
                standard_deviation=trait_descriptive_stats.get("standard_deviation"),
                variance=trait_descriptive_stats.get("variance"),
                standard_error=trait_descriptive_stats.get("standard_error"),
                cv_percent=trait_descriptive_stats.get("cv_percent"),
                min=trait_descriptive_stats.get("min"),
                max=trait_descriptive_stats.get("max"),
                range=trait_descriptive_stats.get("range"),
            )

            # Build interpretation text (now returns ANOVA flags too)
            interp_text, breeding_text, env_sig, gxe_sig = _build_gp_text(trait, res)

            result_obj = GeneticParametersTraitResult(
                trait=trait,
                status="success",
                grand_mean=res.grand_mean,
                descriptive_stats=desc_stats_obj,
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
                environment_significant=env_sig,
                gxe_significant=gxe_sig,
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
