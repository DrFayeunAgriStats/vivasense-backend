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
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from genetics_schemas import GeneticsResponse
from multitrait_upload_routes import build_observations, check_balance, read_file
from module_schemas import AnovaModuleResponse, AnovaTraitResult, ModuleRequest
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


def generate_anova_interpretation(
    trait: str,
    summary: Dict[str, Optional[float]],
    precision_level: Optional[str],
    cv_interpretation_flag: Optional[str],
    genotype_significant: Optional[bool],
    environment_significant: Optional[bool],
    gxe_significant: Optional[bool],
    ranking_caution: Optional[bool],
    selection_feasible: Optional[bool],
    mean_separation: Optional[Any],
    n_genotypes: Optional[int],
    n_environments: Optional[int],
    n_reps: Optional[int],
) -> str:
    """Generate academic-grade ANOVA interpretation following VivaSense standards."""
    
    parts = []
    
    # 1. Overview
    overview = []
    if n_genotypes and n_environments and n_reps:
        if n_environments > 1:
            overview.append(f"This analysis evaluated {trait} across {n_genotypes} genotypes tested in {n_environments} environments with {n_reps} replications per genotype-environment combination.")
        else:
            overview.append(f"This analysis evaluated {trait} across {n_genotypes} genotypes with {n_reps} replications per genotype.")
    
    if summary.get("grand_mean") is not None:
        overview.append(f"The overall mean performance for {trait} was {summary['grand_mean']:.2f}.")
    
    if cv_interpretation_flag == "cv_available" and summary.get("cv_percent") is not None:
        cv = summary["cv_percent"]
        overview.append(f"The coefficient of variation (CV) was {cv:.1f}%, indicating {'high' if cv > 20 else 'moderate' if cv > 10 else 'low'} experimental variability.")
    else:
        overview.append("Experimental precision could not be assessed due to insufficient data for CV calculation.")
    
    parts.append(" ".join(overview))
    
    # 2. Descriptive Interpretation
    desc = []
    if summary.get("grand_mean") is not None:
        desc.append(f"The grand mean of {summary['grand_mean']:.2f} represents the average {trait} performance across all experimental units.")
    
    if summary.get("min") is not None and summary.get("max") is not None and summary.get("range") is not None:
        desc.append(f"Performance ranged from {summary['min']:.2f} to {summary['max']:.2f}, with a total range of {summary['range']:.2f}, indicating {'substantial' if summary['range'] > summary['grand_mean'] * 0.5 else 'moderate'} variability among genotypes.")
    
    if precision_level == "good":
        desc.append("The experimental precision was good, suggesting reliable and reproducible results.")
    elif precision_level == "moderate":
        desc.append("The experimental precision was moderate, indicating acceptable but not optimal experimental control.")
    elif precision_level == "low":
        desc.append("The experimental precision was low, suggesting high variability that warrants cautious interpretation of the results.")
    elif cv_interpretation_flag == "cv_unavailable":
        desc.append("Experimental precision could not be assessed, limiting confidence in the results.")
    
    parts.append(" ".join(desc))
    
    # 3. Genotype Effect
    if genotype_significant is True:
        parts.append(f"Significant genetic variation was detected for {trait} (p < 0.05), indicating that genotypes differ in their performance and that selection for improved {trait} is feasible.")
    elif genotype_significant is False:
        parts.append(f"No significant genetic variation was detected for {trait}, suggesting that the genotypes tested do not differ sufficiently to justify selection based on this trait.")
    else:
        parts.append(f"The significance of genetic variation for {trait} could not be determined.")
    
    # 4. Environment Effect
    if environment_significant is True:
        parts.append(f"Significant environmental variation was observed for {trait}, indicating that growing conditions substantially influence performance and that results may not be transferable across environments.")
    elif environment_significant is False:
        parts.append(f"No significant environmental variation was detected for {trait}, suggesting relatively consistent performance across the tested conditions.")
    else:
        parts.append(f"The significance of environmental variation for {trait} could not be determined.")
    
    # 5. G×E Interaction
    if gxe_significant is True:
        parts.append(f"A significant genotype × environment interaction was detected for {trait}, indicating that genotype performance is not consistent across environments. This suggests that no single genotype is universally superior, and selection strategies should account for environmental stability.")
    elif gxe_significant is False:
        parts.append(f"No significant genotype × environment interaction was detected for {trait}, suggesting relatively stable genotype performance across the tested environments.")
    else:
        parts.append(f"The presence of genotype × environment interaction for {trait} could not be determined.")
    
    # 6. Mean Performance and Ranking
    ranking = []
    if mean_separation and hasattr(mean_separation, 'genotype') and mean_separation.genotype:
        try:
            top_genotype = mean_separation.genotype[0]
            top_mean = mean_separation.mean[0]
            ranking.append(f"Based on overall means, {top_genotype} exhibited the highest {trait} performance ({top_mean:.2f}).")
        except (IndexError, TypeError):
            ranking.append("Mean separation analysis was available but could not be summarized.")
    else:
        ranking.append("Detailed mean separation analysis was not available.")
    
    if ranking_caution is True:
        ranking.append("However, due to significant genotype × environment interaction, ranking based on overall means should be interpreted cautiously, as performance may vary across environments.")
    
    parts.append(" ".join(ranking))
    
    # 7. Breeding Interpretation
    breeding = []
    if selection_feasible is True:
        breeding.append("The results suggest that selection for improved {trait} is feasible.")
        if gxe_significant is False:
            breeding.append("Given the absence of significant genotype × environment interaction, breeding efforts can focus on broad adaptation across environments.")
        else:
            breeding.append("However, due to significant genotype × environment interaction, breeding strategies should prioritize stability analysis and environment-specific selection.")
    else:
        breeding.append("The lack of significant genetic variation indicates that selection for {trait} may not be effective with the current germplasm.")
    
    breeding.append("The observed variability and experimental precision should guide the design of future experiments and breeding trials.")
    parts.append(" ".join(breeding))
    
    # 8. Risk and Limitations
    risks = []
    if gxe_significant is True:
        risks.append("The significant genotype × environment interaction represents a major limitation, as it complicates genotype evaluation and selection.")
    if precision_level == "low":
        risks.append("The low experimental precision introduces uncertainty in the results and suggests potential issues with experimental control or replication.")
    if environment_significant is True:
        risks.append("Strong environmental influence may limit the generalizability of these results to other locations or conditions.")
    if not risks:
        risks.append("No major experimental limitations were identified in this analysis.")
    parts.append(" ".join(risks))
    
    # 9. Recommendation
    recs = []
    if gxe_significant is True:
        recs.append("Conduct stability analysis (e.g., AMMI or GGE biplot) to identify genotypes with consistent performance across environments.")
    if selection_feasible is True:
        recs.append("Consider advancing promising genotypes to further evaluation, with appropriate caution regarding environmental interactions.")
    if precision_level == "low":
        recs.append("Improve experimental design by increasing replication or enhancing environmental control to reduce variability.")
    recs.append("Integrate these ANOVA results with genetic parameter estimates (heritability, genetic coefficient of variation) for comprehensive trait evaluation.")
    parts.append(" ".join(recs))
    
    # Format with headers
    sections = [
        "Overview",
        "Descriptive Interpretation", 
        "Genotype Effect",
        "Environment Effect",
        "G×E Interaction",
        "Mean Performance and Ranking",
        "Breeding Interpretation",
        "Risk and Limitations",
        "Recommendation"
    ]
    
    formatted = []
    for section, content in zip(sections, parts):
        formatted.append(f"{section}\n{content}")
    
    return "\n\n".join(formatted)


def compute_descriptive_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """Compute numeric descriptive statistics for a trait series.

    This uses raw observation-level data and ignores missing values.
    If insufficient observations exist, variance/SD/SE are returned as None.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)

    if n == 0:
        return {
            "grand_mean": None,
            "standard_deviation": None,
            "variance": None,
            "standard_error": None,
            "cv_percent": None,
            "min": None,
            "max": None,
            "range": None,
        }

    grand_mean = float(clean.mean())
    min_val = float(clean.min())
    max_val = float(clean.max())
    range_val = max_val - min_val

    if n >= 2:
        variance = float(clean.var(ddof=1))
        standard_deviation = float(variance ** 0.5)
        standard_error = float(standard_deviation / (n ** 0.5))
    else:
        variance = None
        standard_deviation = None
        standard_error = None

    cv_percent = None
    if grand_mean != 0 and standard_deviation is not None:
        cv_percent = float((standard_deviation / grand_mean) * 100)

    return {
        "grand_mean": grand_mean,
        "standard_deviation": standard_deviation,
        "variance": variance,
        "standard_error": standard_error,
        "cv_percent": cv_percent,
        "min": min_val,
        "max": max_val,
        "range": range_val,
    }


def compute_per_genotype_stats(
    df: pd.DataFrame, trait_column: str, genotype_column: str
) -> List[Dict[str, Optional[float]]]:
    """Compute per-genotype descriptive statistics for the requested trait."""
    if genotype_column not in df.columns:
        return []

    grouped = df[[genotype_column, trait_column]].copy()
    grouped[trait_column] = pd.to_numeric(grouped[trait_column], errors="coerce")

    stats: List[Dict[str, Optional[float]]] = []
    for genotype, group in grouped.groupby(genotype_column, sort=True):
        clean = group[trait_column].dropna()
        n = len(clean)

        if n == 0:
            stats.append(
                {
                    "genotype": genotype,
                    "mean": None,
                    "sd": None,
                    "cv_percent": None,
                }
            )
            continue

        mean_val = float(clean.mean())
        sd_val = None
        cv_percent = None

        if n >= 2:
            variance = float(clean.var(ddof=1))
            sd_val = float(variance ** 0.5)
            if mean_val != 0:
                cv_percent = float((sd_val / mean_val) * 100)

        stats.append(
            {
                "genotype": genotype,
                "mean": mean_val,
                "sd": sd_val,
                "cv_percent": cv_percent,
            }
        )

    return stats


def classify_precision_level(cv_percent: Optional[float]) -> str:
    """Classify experimental precision based on coefficient of variation."""
    if cv_percent is None:
        return "low"  # No CV available, assume low precision
    if cv_percent < 10.0:
        return "good"
    elif cv_percent < 20.0:
        return "moderate"
    else:
        return "low"


def get_cv_interpretation_flag(cv_percent: Optional[float]) -> str:
    """Return flag indicating if CV is available for interpretation."""
    return "cv_available" if cv_percent is not None else "cv_unavailable"


def is_genotype_effect_significant(anova_table) -> bool:
    """Check if genotype effect is significant (p < 0.05)."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return False
    try:
        idx = anova_table.source.index("genotype")
        p_val = anova_table.p_value[idx]
        return p_val is not None and p_val < 0.05
    except (ValueError, IndexError):
        return False


def is_environment_effect_significant(anova_table) -> bool:
    """Check if environment effect is significant (p < 0.05)."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return False
    try:
        idx = anova_table.source.index("environment")
        p_val = anova_table.p_value[idx]
        return p_val is not None and p_val < 0.05
    except (ValueError, IndexError):
        return False


def is_gxe_effect_significant(anova_table) -> bool:
    """Check if genotype x environment interaction is significant (p < 0.05)."""
    if not anova_table or not hasattr(anova_table, "source") or not hasattr(anova_table, "p_value"):
        return False
    # Look for common GxE terms
    gxe_terms = ["genotype:environment", "environment:genotype", "GxE", "gxe"]
    for term in gxe_terms:
        try:
            idx = anova_table.source.index(term)
            p_val = anova_table.p_value[idx]
            if p_val is not None and p_val < 0.05:
                return True
        except (ValueError, IndexError):
            continue
    return False


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

            trait_descriptive_stats = compute_descriptive_stats(df[trait])
            per_genotype_stats = compute_per_genotype_stats(df, trait, geno_col)

            # Build summary from descriptive stats
            summary = {
                "grand_mean": trait_descriptive_stats["grand_mean"],
                "cv_percent": trait_descriptive_stats["cv_percent"],
                "min": trait_descriptive_stats["min"],
                "max": trait_descriptive_stats["max"],
                "range": trait_descriptive_stats["range"],
                "standard_error": trait_descriptive_stats["standard_error"],
            }

            # Classify precision and flags
            precision_level = classify_precision_level(trait_descriptive_stats["cv_percent"])
            cv_interpretation_flag = get_cv_interpretation_flag(trait_descriptive_stats["cv_percent"])
            genotype_significant = is_genotype_effect_significant(res.anova_table)
            environment_significant = is_environment_effect_significant(res.anova_table)
            gxe_significant = is_gxe_effect_significant(res.anova_table)
            # ranking_caution follows directly from GxE significance
            ranking_caution = gxe_significant
            selection_feasible = genotype_significant

            # Generate ANOVA interpretation
            anova_interpretation = generate_anova_interpretation(
                trait=trait,
                summary=summary,
                precision_level=precision_level,
                cv_interpretation_flag=cv_interpretation_flag,
                genotype_significant=genotype_significant,
                environment_significant=environment_significant,
                gxe_significant=gxe_significant,
                ranking_caution=ranking_caution,
                selection_feasible=selection_feasible,
                mean_separation=res.mean_separation,
                n_genotypes=res.n_genotypes,
                n_environments=res.n_environments,
                n_reps=res.n_reps,
            )
            logger.info(
                "ANOVA interpretation generated for trait '%s': %d characters",
                trait, len(anova_interpretation) if anova_interpretation else 0
            )

            result_obj = AnovaTraitResult(
                trait=trait,
                status="success",
                grand_mean=res.grand_mean,
                n_genotypes=res.n_genotypes,
                n_reps=res.n_reps,
                n_environments=res.n_environments,
                anova_table=res.anova_table,
                descriptive_stats=trait_descriptive_stats,
                per_genotype_stats=per_genotype_stats,
                summary=summary,
                precision_level=precision_level,
                cv_interpretation_flag=cv_interpretation_flag,
                ranking_caution=ranking_caution,
                selection_feasible=selection_feasible,
                genotype_significant=genotype_significant,
                environment_significant=environment_significant,
                gxe_significant=gxe_significant,
                assumption_tests=res.assumption_tests,
                mean_separation=res.mean_separation,
                interpretation=anova_interpretation,
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
        else:
            logger.info(
                "ANOVA response: trait '%s' has interpretation = %s",
                trait, "YES" if result_obj.interpretation else "NO"
            )

    logger.info(
        "ANOVA endpoint returning %d successful traits, %d failed traits",
        len(trait_results) - len(failed_traits), len(failed_traits)
    )

    return AnovaModuleResponse(
        dataset_token=request.dataset_token,
        mode=mode,
        trait_results=trait_results,
        failed_traits=failed_traits,
    )
