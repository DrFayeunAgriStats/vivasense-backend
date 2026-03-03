"""
GeneticsPipeline — orchestrates all breeding analysis engines.

Mirrors the role of VivaSenseBackend in main.py:
  - Validates input data
  - Checks cache before running analysis
  - Calls engines in the correct dependency order
  - Assembles and caches the final result
  - Saves result to ./results/ for later retrieval

Dependency order (respects data flow between engines):
  1. validate_trial / validate_markers
  2. MultilocationEngine.run_combined_anova   (g, l, r counts + MS values)
  3. MultilocationEngine.fit_ammi             (needs anova_result)
  4. MultilocationEngine.fit_gge              (needs anova_result)
  5. VarianceComponentEngine.estimate         (needs anova_result → MS values)
  6. StabilityEngine.eberhart_russell         (needs anova_result → cell_matrix)
  7. StabilityEngine.compute_asv              (needs ammi_result → IPCA scores)
  8. CorrelationEngine.phenotypic_correlations
  9. CorrelationEngine.genotypic_correlations (needs variance_components)
  10. CorrelationEngine.path_analysis
  11. CorrelationEngine.selection_index       (needs variance_components)
  12. MultivariateEngine.run_all
  13. GeneticsPlotter.*                       (needs all engine results)
  14. Assemble GeneticsAnalysisResult
  15. CacheManager + save_results
"""
from __future__ import annotations
import hashlib
import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import GeneticsConfig, TrialDesign, MarkerConfig
from .serializers import numpy_to_python
from .validators import validate_trial, validate_markers, detect_trait_cols
from .plotting import GeneticsPlotter
from .engines import (
    MultilocationEngine,
    VarianceComponentEngine,
    StabilityEngine,
    CorrelationEngine,
    MultivariateEngine,
    MarkerEngine,
)

logger = logging.getLogger(__name__)

# Results are saved alongside the main VivaSense results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_DIR = Path(__file__).parent.parent / "cache"


class GeneticsPipeline:
    """
    Top-level orchestrator for genetics analyses.
    Constructed with config and design; call run_trial_analysis() or run_marker_analysis().
    """

    def __init__(
        self,
        config: Optional[GeneticsConfig] = None,
        design: Optional[TrialDesign] = None,
        marker_config: Optional[MarkerConfig] = None,
    ):
        self.config = config or GeneticsConfig()
        self.design = design or TrialDesign()
        self.marker_config = marker_config or MarkerConfig()

        # Engine instances
        self.ml_engine = MultilocationEngine(self.config)
        self.vc_engine = VarianceComponentEngine(self.config)
        self.stab_engine = StabilityEngine(self.config)
        self.corr_engine = CorrelationEngine(self.config)
        self.mv_engine = MultivariateEngine(self.config)
        self.plotter = GeneticsPlotter(self.config)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # TRIAL ANALYSIS
    # ------------------------------------------------------------------

    def run_trial_analysis(
        self,
        df: pd.DataFrame,
        filename: str = "upload",
    ) -> Dict[str, Any]:
        """
        Full breeding trial analysis pipeline.
        Auto-selects single-environment or multi-environment mode.
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        warnings: List[str] = []
        design = self.design

        # ── 1. Resolve design columns ────────────────────────────────
        if design.trait_cols is None:
            trait_cols = detect_trait_cols(df, design)
        else:
            trait_cols = [c for c in design.trait_cols if c in df.columns]

        # Combine Location + Season → Environment if needed
        loc_col = design.location_col
        if loc_col and design.season_col and design.season_col in df.columns:
            df = df.copy()
            df[loc_col] = df[loc_col].astype(str) + "_" + df[design.season_col].astype(str)

        # ── 3. Validate ───────────────────────────────────────────────
        val = validate_trial(df, design)
        if val["errors"]:
            return {
                "status": "validation_error",
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "errors": val["errors"],
                "warnings": val["warnings"],
            }
        warnings.extend(val["warnings"])

        n_locations = df[loc_col].nunique() if loc_col and loc_col in df.columns else 1
        is_multi = n_locations >= 2

        # ── 4. Combined ANOVA (multi-location) ────────────────────────
        anova_results: Dict[str, Any] = {}   # trait → anova_result
        ammi_results: Dict[str, Any] = {}
        gge_results: Dict[str, Any] = {}
        vc_results: Dict[str, Any] = {}
        stab_results: Dict[str, Any] = {}

        for trait in trait_cols:
            try:
                if is_multi:
                    ar = self.ml_engine.run_combined_anova(
                        df, trait, design.genotype_col, loc_col, design.rep_col
                    )
                    anova_results[trait] = ar

                    ammi_r = self.ml_engine.fit_ammi(ar)
                    ammi_results[trait] = ammi_r

                    gge_r = self.ml_engine.fit_gge(ar)
                    gge_results[trait] = gge_r

                    vc = self.vc_engine.estimate(ar, df, trait, design.genotype_col, loc_col, design.rep_col)
                    vc_results[trait] = vc

                    er = self.stab_engine.eberhart_russell(ar)
                    er = self.stab_engine.compute_asv(er, ammi_r)
                    stab_results[trait] = er
                else:
                    # Single-location: simplified ANOVA (no G×L)
                    single_anova = _single_location_anova(df, trait, design.genotype_col, design.rep_col)
                    anova_results[trait] = single_anova
                    vc = self.vc_engine.estimate(
                        single_anova, df, trait, design.genotype_col, None, design.rep_col
                    )
                    vc_results[trait] = vc

            except Exception:
                logger.error("Analysis failed for trait %s:\n%s", trait, traceback.format_exc())
                warnings.append(f"Analysis failed for trait '{trait}' — see server logs.")

        # ── 5. Correlations, path, selection index ────────────────────
        corr_result: Dict[str, Any] = {}
        path_result: Dict[str, Any] = {}
        sel_index_result: Dict[str, Any] = {}

        if len(trait_cols) >= 2:
            try:
                ph_corr = self.corr_engine.phenotypic_correlations(df, trait_cols, loc_col)
                ge_corr = self.corr_engine.genotypic_correlations(df, trait_cols, design.genotype_col, vc_results)
                corr_result = {"phenotypic": ph_corr, "genotypic": ge_corr}
            except Exception:
                logger.error("Correlation analysis failed:\n%s", traceback.format_exc())
                warnings.append("Correlation analysis failed — check trait data.")

            try:
                target = self.config.path_target_trait or trait_cols[0]
                path_result = self.corr_engine.path_analysis(df, trait_cols, design.genotype_col, target, loc_col)
            except Exception:
                logger.error("Path analysis failed:\n%s", traceback.format_exc())
                warnings.append("Path coefficient analysis failed.")

            try:
                sel_index_result = self.corr_engine.selection_index(
                    df, trait_cols, design.genotype_col, vc_results,
                    self.config.economic_weights, loc_col,
                )
            except Exception:
                logger.error("Selection index failed:\n%s", traceback.format_exc())
                warnings.append("Selection index computation failed.")

        # ── 6. Multivariate (PCA + clustering) ───────────────────────
        mv_result: Dict[str, Any] = {}
        if len(trait_cols) >= 2:
            try:
                mv_result = self.mv_engine.run_all(df, trait_cols, design.genotype_col, loc_col)
            except Exception:
                logger.error("Multivariate analysis failed:\n%s", traceback.format_exc())
                warnings.append("PCA/clustering failed.")

        # ── 7. Plots ──────────────────────────────────────────────────
        plots = _generate_trial_plots(
            self.plotter, trait_cols, ammi_results, gge_results,
            stab_results, corr_result, path_result, mv_result,
        )

        # ── 8. Plain-English interpretation ──────────────────────────
        interpretation = _interpret_trial(
            trait_cols, vc_results, stab_results, is_multi, n_locations
        )

        # ── 9. Assemble result ────────────────────────────────────────
        result = numpy_to_python({
            "status": "success",
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "metadata": {
                "filename": filename,
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "n_genotypes": df[design.genotype_col].nunique(),
                "n_locations": n_locations,
                "mode": "multilocational" if is_multi else "single_environment",
                "trait_cols": trait_cols,
                "genotype_col": design.genotype_col,
                "location_col": loc_col,
                "rep_col": design.rep_col,
                "design_type": design.design_type,
                "config": self.config.to_dict(),
            },
            "warnings": warnings,
            "variance_components": vc_results,
            "anova_tables": {t: anova_results[t]["anova_table"] for t in anova_results},
            "ammi": ammi_results,
            "gge": gge_results,
            "stability": stab_results,
            "correlations": corr_result,
            "path_analysis": path_result,
            "selection_index": sel_index_result,
            "multivariate": mv_result,
            "plots": plots,
            "interpretation": interpretation,
        })

        # ── 10. Persist result ────────────────────────────────────────
        _save_result(analysis_id, result)

        return result

    # ------------------------------------------------------------------
    # MARKER ANALYSIS
    # ------------------------------------------------------------------

    def run_marker_analysis(
        self,
        df: pd.DataFrame,
        filename: str = "markers",
    ) -> Dict[str, Any]:
        """Full molecular marker analysis pipeline."""
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Validate
        val = validate_markers(df, self.marker_config)
        if val["errors"]:
            return {
                "status": "validation_error",
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "errors": val["errors"],
                "warnings": val["warnings"],
            }

        mk_engine = MarkerEngine(self.config, self.marker_config)
        try:
            diversity = mk_engine.analyze(df)
        except Exception:
            return {
                "status": "error",
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "error": traceback.format_exc(),
            }

        # Plots
        plots = {}
        try:
            sim = diversity.get("similarity_matrix", {})
            if sim:
                plots["similarity_heatmap"] = self.plotter.similarity_heatmap(sim, "Genetic Similarity")
            pca_mol = diversity.get("pca_molecular", {})
            if pca_mol and "scores" in pca_mol:
                plots["marker_pca"] = self.plotter.pca_biplot(
                    {"biplot_data": {
                        "x_axis_label": "PC1",
                        "y_axis_label": "PC2",
                        "genotype_points": [
                            {"id": k, "x": v.get("PC1", 0), "y": v.get("PC2", 0), "type": "genotype"}
                            for k, v in pca_mol.get("scores", {}).items()
                        ],
                        "loading_vectors": [],
                    }},
                    title="Molecular PCA Biplot",
                )
        except Exception:
            logger.warning("Marker plot generation failed: %s", traceback.format_exc())

        result = numpy_to_python({
            "status": "success",
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "metadata": {
                "filename": filename,
                "n_rows": len(df),
                "accession_col": self.marker_config.accession_col,
                "config": self.config.to_dict(),
            },
            "warnings": val["warnings"],
            "diversity": diversity,
            "plots": plots,
        })

        _save_result(analysis_id, result)
        return result


# ── Helper functions ─────────────────────────────────────────────────────────

def _single_location_anova(
    df: pd.DataFrame, trait: str, geno_col: str, rep_col: str
) -> Dict[str, Any]:
    """Simplified one-location RCBD ANOVA returning the same dict shape."""
    from scipy import stats as scipy_stats

    genotypes = sorted(df[geno_col].unique())
    g = len(genotypes)
    r = int(df.groupby(geno_col)[rep_col].nunique().mode().iloc[0])
    grand_mean = float(df[trait].mean())
    geno_means = df.groupby(geno_col)[trait].mean()

    SS_G = r * float(((geno_means - grand_mean) ** 2).sum())
    SS_total = float(((df[trait] - grand_mean) ** 2).sum())
    SS_error = max(0.0, SS_total - SS_G)
    df_G = g - 1
    df_error = max(1, g * (r - 1))
    MS_G = SS_G / df_G if df_G > 0 else 0.0
    MS_error = SS_error / df_error if df_error > 0 else 0.0
    F_G = MS_G / MS_error if MS_error > 0 else 0.0
    p_G = float(1 - scipy_stats.f.cdf(F_G, df_G, df_error)) if F_G > 0 else 1.0

    # Dummy location for compatibility with VC engine
    dummy_loc_col = "__single_loc__"
    df_dummy = df.copy()
    df_dummy[dummy_loc_col] = "Loc1"

    return {
        "anova_table": {
            "columns": ["Source", "df", "SS", "MS", "F", "p_value"],
            "rows": [
                ["Genotype", df_G, round(SS_G, 4), round(MS_G, 4), round(F_G, 3), round(p_G, 4)],
                ["Error", df_error, round(SS_error, 4), round(MS_error, 4), None, None],
                ["Total", df_G + df_error, round(SS_total, 4), None, None, None],
            ],
        },
        "grand_mean": grand_mean,
        "n_genotypes": g,
        "n_locations": 1,
        "n_reps": r,
        "genotype_names": genotypes,
        "location_names": ["Loc1"],
        "geno_means": geno_means,
        "loc_means": pd.Series({"Loc1": grand_mean}),
        "cell_matrix": geno_means.rename("Loc1").to_frame(),
        "interaction_matrix": __import__("numpy").zeros((g, 1)),
        "MS": {"G": MS_G, "L": 0.0, "GL": 0.0, "error": MS_error},
        "SS": {"G": SS_G, "L": 0.0, "GL": 0.0, "error": SS_error, "total": SS_total},
        "df": {"G": df_G, "L": 0, "GL": 0, "error": df_error, "total": df_G + df_error},
        "F": {"G": round(F_G, 3), "L": 0.0, "GL": 0.0},
        "p": {"G": round(p_G, 4), "L": 1.0, "GL": 1.0},
    }


def _generate_trial_plots(
    plotter: GeneticsPlotter,
    trait_cols: List[str],
    ammi_results: Dict,
    gge_results: Dict,
    stab_results: Dict,
    corr_result: Dict,
    path_result: Dict,
    mv_result: Dict,
) -> Dict[str, Any]:
    plots: Dict[str, Any] = {}

    # Use first trait for AMMI/GGE/stability plots
    primary = trait_cols[0] if trait_cols else None

    if primary and primary in ammi_results:
        try:
            plots[f"ammi_biplot_{primary}"] = plotter.ammi_biplot(ammi_results[primary], primary)
        except Exception:
            logger.warning("AMMI biplot failed for %s", primary)

    if primary and primary in gge_results:
        try:
            plots[f"gge_biplot_{primary}"] = plotter.gge_biplot(gge_results[primary], primary)
        except Exception:
            logger.warning("GGE biplot failed for %s", primary)

    if primary and primary in stab_results:
        try:
            plots[f"stability_regression_{primary}"] = plotter.stability_regression(stab_results[primary], primary)
            plots[f"mean_vs_bi_{primary}"] = plotter.mean_vs_bi(stab_results[primary], primary)
        except Exception:
            logger.warning("Stability plot failed for %s", primary)

    if corr_result.get("phenotypic"):
        try:
            plots["phenotypic_heatmap"] = plotter.correlation_heatmap(
                corr_result["phenotypic"], "phenotypic"
            )
        except Exception:
            logger.warning("Phenotypic heatmap failed")

    if corr_result.get("genotypic"):
        try:
            plots["genotypic_heatmap"] = plotter.correlation_heatmap(
                corr_result["genotypic"], "genotypic"
            )
        except Exception:
            logger.warning("Genotypic heatmap failed")

    if path_result and path_result.get("direct_effects"):
        try:
            target = path_result.get("target_trait", "")
            plots["path_diagram"] = plotter.path_diagram(path_result, target)
        except Exception:
            logger.warning("Path diagram failed")

    if mv_result and mv_result.get("pca"):
        try:
            cl = mv_result.get("kmeans_clustering", {}).get("cluster_labels", {})
            plots["pca_biplot"] = plotter.pca_biplot(mv_result["pca"], cl, "PCA Biplot — Genotypes")
            plots["scree_plot"] = plotter.scree_plot(mv_result["pca"])
        except Exception:
            logger.warning("PCA plot failed")

    if mv_result and mv_result.get("hierarchical_clustering"):
        try:
            dd = mv_result["hierarchical_clustering"].get("dendrogram_data", {})
            n = len(mv_result.get("kmeans_clustering", {}).get("cluster_labels", {}))
            plots["dendrogram"] = plotter.dendrogram(dd, "Hierarchical Clustering — Genotypes", n)
        except Exception:
            logger.warning("Dendrogram failed")

    return plots


def _interpret_trial(
    trait_cols: List[str],
    vc_results: Dict,
    stab_results: Dict,
    is_multi: bool,
    n_locations: int,
) -> Dict[str, Any]:
    """Generate plain-English summary for the analysis."""
    summaries = []
    recommendations = []

    for trait in trait_cols:
        vc = vc_results.get(trait, {})
        H2 = vc.get("heritability", {}).get("H2_broad", {}).get("value")
        GA_pct = vc.get("genetic_advance", {}).get("GA_percent", {}).get("value")
        GCV = vc.get("coefficients_of_variation", {}).get("GCV", {}).get("value")
        grand_mean = vc.get("grand_mean")

        if H2 is not None:
            summaries.append(
                f"{trait}: H²={H2:.2f}, GA%={GA_pct:.1f}%, GCV={GCV:.1f}% — "
                f"{vc.get('heritability', {}).get('H2_broad', {}).get('interpretation', '')}"
            )

        stab = stab_results.get(trait, {})
        gs = stab.get("genotype_stability", [])
        if gs:
            top = gs[0]  # already sorted by yield
            recommendations.append(
                f"Highest yielding genotype for {trait}: {top['genotype']} "
                f"(mean={top['grand_mean']}, class={top['classification']})"
            )
            stable = [e for e in gs if "stable_high" in e.get("classification", "")]
            if stable:
                recommendations.append(
                    f"Broad-release candidates ({trait}): {', '.join(e['genotype'] for e in stable[:3])}"
                )

    mode_text = (
        f"Multi-environment trial across {n_locations} locations."
        if is_multi
        else "Single-environment trial."
    )

    return {
        "overall": mode_text + f" Analysed {len(trait_cols)} trait(s).",
        "trait_summaries": summaries,
        "recommendations": recommendations,
    }


def _cache_key(df: pd.DataFrame, config: GeneticsConfig, design: TrialDesign) -> str:
    h = hashlib.md5()
    h.update(df.to_csv(index=False).encode())
    h.update(json.dumps(config.to_dict(), sort_keys=True).encode())
    h.update(json.dumps(design.to_dict(), sort_keys=True).encode())
    return h.hexdigest()


def _load_cache(key: str) -> Optional[Dict]:
    path = CACHE_DIR / f"genetics_{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _save_cache(key: str, result: Dict):
    path = CACHE_DIR / f"genetics_{key}.json"
    try:
        path.write_text(json.dumps(result, default=str))
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)


def _save_result(analysis_id: str, result: Dict):
    path = RESULTS_DIR / f"{analysis_id}.json"
    try:
        path.write_text(json.dumps(result, default=str, indent=2))
    except Exception as exc:
        logger.warning("Result persist failed: %s", exc)
