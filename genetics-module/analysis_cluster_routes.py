"""
VivaSense – Cluster Analysis Module

POST /analysis/cluster

Groups genotypes by phenotypic similarity across multiple traits using
hierarchical clustering with Ward linkage (or user-specified method).
Optimal cluster number is determined by maximising the average silhouette
coefficient when k is not supplied explicitly.

Provides:
  • Cluster assignments per genotype
  • Per-cluster mean profile (trait × cluster summary)
  • Per-genotype silhouette score
  • Dendrogram data (linkage matrix) for visualisation
  • Plain-English interpretation

Implementation:
  scipy.cluster.hierarchy for hierarchical clustering
  sklearn.metrics.silhouette_score / silhouette_samples for validation

Reference:
  Ward, J.H. (1963). Hierarchical grouping to optimise an objective function.
  Journal of the American Statistical Association, 58(301), 236-244.

  Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation
  and validation of cluster analysis. Journal of Computational and Applied
  Mathematics, 20, 53-65.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

import dataset_cache
from module_schemas import (
    ClusterRequest,
    ClusterResponse,
    ClusterSummary,
    GenotypeCluster,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

# Map user-friendly method names to scipy linkage method names
_LINKAGE_MAP: Dict[str, str] = {
    "ward": "ward",
    "ward.D2": "ward",
    "ward.d2": "ward",
    "complete": "complete",
    "average": "average",
    "single": "single",
}


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_clusters(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str,
    method: str,
    k: Optional[int],
    scale: bool,
) -> Dict[str, Any]:
    """
    Aggregate to genotype means, apply hierarchical clustering, determine
    optimal k via silhouette, and return full cluster summary.
    """
    # Aggregate to genotype means
    geno_means = (
        df.groupby(genotype_col)[trait_cols]
        .mean()
        .dropna(how="all")
    )
    geno_means = geno_means.dropna(axis=1, how="all")
    available_traits = geno_means.columns.tolist()

    if len(available_traits) < 2:
        raise ValueError(
            "Cluster analysis requires at least 2 traits with non-missing data."
        )

    geno_means = geno_means.dropna()
    n_genotypes = len(geno_means)

    if n_genotypes < 4:
        raise ValueError(
            f"Cluster analysis requires at least 4 genotypes. "
            f"Only {n_genotypes} found with complete data."
        )

    X = geno_means.values.astype(float)
    genotype_labels = [str(g) for g in geno_means.index.tolist()]

    # Standardise
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Resolve linkage method
    method_lower = method.strip().lower().replace(".", "").replace("_", "").replace("-", "")
    scipy_method = _LINKAGE_MAP.get(method, None)
    if scipy_method is None:
        for key, val in _LINKAGE_MAP.items():
            if key.lower().replace(".", "") == method_lower:
                scipy_method = val
                break
    if scipy_method is None:
        scipy_method = "ward"   # default fallback

    # Build linkage matrix
    Z = linkage(X_scaled, method=scipy_method, metric="euclidean")

    # Determine optimal k
    max_k = min(10, n_genotypes - 1)
    min_k = 2

    if k is not None:
        # User-specified k
        optimal_k = max(min_k, min(int(k), max_k))
        labels = fcluster(Z, optimal_k, criterion="maxclust")
        if len(set(labels)) > 1:
            sil_avg = float(silhouette_score(X_scaled, labels))
        else:
            sil_avg = 0.0
    else:
        # Auto-detect via silhouette
        best_k = min_k
        best_sil = -1.0
        for candidate_k in range(min_k, max_k + 1):
            candidate_labels = fcluster(Z, candidate_k, criterion="maxclust")
            if len(set(candidate_labels)) < 2:
                continue
            try:
                s = float(silhouette_score(X_scaled, candidate_labels))
            except Exception:
                continue
            if s > best_sil:
                best_sil = s
                best_k = candidate_k
        optimal_k = best_k
        labels = fcluster(Z, optimal_k, criterion="maxclust")

    # Per-genotype silhouette scores
    if len(set(labels)) > 1:
        sil_samples = silhouette_samples(X_scaled, labels).tolist()
    else:
        sil_samples = [0.0] * n_genotypes

    # Cluster assignments
    assignments: List[Dict[str, Any]] = []
    for i, geno in enumerate(genotype_labels):
        assignments.append({
            "genotype": geno,
            "cluster_id": int(labels[i]),
            "silhouette_score": round(float(sil_samples[i]), 4),
        })
    assignments.sort(key=lambda r: (r["cluster_id"], r["genotype"]))

    # Cluster summary: mean per trait
    cluster_ids = sorted(set(int(l) for l in labels))
    cluster_summary: List[Dict[str, Any]] = []
    for cid in cluster_ids:
        members_idx = [i for i, l in enumerate(labels) if int(l) == cid]
        X_cluster = X[members_idx]   # use un-scaled for interpretable means
        size = len(members_idx)
        mean_per_trait: Dict[str, float] = {}
        for j, trait in enumerate(available_traits):
            mean_per_trait[trait] = round(float(np.mean(X_cluster[:, j])), 4)
        cluster_summary.append({
            "cluster_id": cid,
            "size": size,
            "mean_per_trait": mean_per_trait,
        })

    # Dendrogram data: linkage matrix rows as lists for JSON serialisation
    dendrogram_data = {
        "linkage_matrix": Z.tolist(),
        "labels": genotype_labels,
        "method": scipy_method,
    }

    return {
        "n_genotypes": n_genotypes,
        "n_traits": len(available_traits),
        "method": scipy_method,
        "optimal_k": optimal_k,
        "assignments": assignments,
        "cluster_summary": cluster_summary,
        "silhouette_scores": [round(float(s), 4) for s in sil_samples],
        "dendrogram_data": dendrogram_data,
        "trait_names": available_traits,
        "cluster_ids": cluster_ids,
    }


def _generate_cluster_interpretation(
    result: Dict[str, Any],
) -> str:
    """Generate plain-English thesis-quality interpretation."""
    sections: List[tuple] = []
    n_genos = result["n_genotypes"]
    n_traits = result["n_traits"]
    optimal_k = result["optimal_k"]
    method = result["method"]
    cluster_summary = result["cluster_summary"]
    sil_scores = result["silhouette_scores"]
    assignments = result["assignments"]

    avg_sil = float(np.mean(sil_scores)) if sil_scores else 0.0
    trait_names = result["trait_names"]

    # 1. Overview
    overview = (
        f"Hierarchical cluster analysis ({method} linkage, Euclidean distance) "
        f"was performed on {n_genos} genotypes using {n_traits} standardised "
        f"phenotypic traits. "
        f"The optimal number of clusters was determined as k = {optimal_k} "
        f"by maximising the average silhouette coefficient "
        f"(Ward, 1963; Rousseeuw, 1987). "
        f"The average silhouette score was {avg_sil:.3f} "
        f"({'strong' if avg_sil >= 0.5 else 'moderate' if avg_sil >= 0.25 else 'weak'} "
        f"cluster structure)."
    )
    sections.append(("Overview", overview))

    # 2. Cluster Profiles
    cluster_desc = []
    for cs in cluster_summary:
        cid = cs["cluster_id"]
        size = cs["size"]
        means = cs["mean_per_trait"]
        # Find the highest and lowest trait mean within this cluster
        if means:
            top_trait = max(means, key=lambda t: means[t])
            low_trait = min(means, key=lambda t: means[t])
            cluster_desc.append(
                f"Cluster {cid} ({size} genotype{'s' if size != 1 else ''}): "
                f"highest mean for '{top_trait}' ({means[top_trait]:.3f}); "
                f"lowest mean for '{low_trait}' ({means[low_trait]:.3f})."
            )
        else:
            cluster_desc.append(f"Cluster {cid} ({size} genotype{'s' if size != 1 else ''}).")
    sections.append(("Cluster Profiles", " ".join(cluster_desc)))

    # 3. Silhouette Interpretation
    low_sil = [a["genotype"] for a in assignments if a["silhouette_score"] < 0.0]
    sil_text = (
        f"Silhouette coefficients range from {min(sil_scores):.3f} to "
        f"{max(sil_scores):.3f} (mean: {avg_sil:.3f}). "
    )
    if low_sil:
        sil_text += (
            f"Genotypes with negative silhouette scores "
            f"({', '.join(low_sil[:5])}{'...' if len(low_sil) > 5 else ''}) "
            f"may be outliers or intermediate phenotypes that do not fit "
            f"cleanly into any one cluster."
        )
    else:
        sil_text += (
            "All genotypes have non-negative silhouette scores, indicating "
            "well-defined cluster membership."
        )
    sections.append(("Silhouette Validation", sil_text))

    # 4. Breeding Implications
    breed_text = (
        f"From a breeding perspective, genotypes within the same cluster "
        f"exhibit similar phenotypic profiles and may be grouped into the "
        f"same breeding pool. Crosses within clusters will reduce diversity, "
        f"while crosses between clusters (especially those most phenotypically "
        f"divergent) maximise heterosis potential and broaden the genetic base. "
        f"The cluster with the highest means across key yield traits represents "
        f"the elite breeding pool; the most divergent cluster provides parents "
        f"for wide-cross programmes aimed at introducing novel variation."
    )
    sections.append(("Breeding Implications", breed_text))

    # 5. Methodological Note
    method_note = (
        f"Ward's minimum variance linkage (or the selected '{method}' method) "
        f"minimises within-cluster sum of squares at each merge step, producing "
        f"compact, well-separated clusters. The Euclidean distance metric on "
        f"standardised data ensures that all traits contribute equally regardless "
        f"of measurement scale."
    )
    sections.append(("Methodological Note", method_note))

    return "\n\n".join(f"{h}\n{c}" for h, c in sections)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/cluster",
    response_model=ClusterResponse,
    summary="Hierarchical cluster analysis for genotype grouping",
)
async def analysis_cluster(request: ClusterRequest) -> ClusterResponse:
    """
    Group genotypes by phenotypic similarity using hierarchical clustering.

    Optimal cluster number k is determined automatically by silhouette
    analysis when not provided explicitly.

    Requires:
      - A dataset_token from POST /upload/dataset
      - At least 2 trait columns
      - At least 4 genotypes with complete data
    """
    if len(request.trait_columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="Cluster analysis requires at least 2 trait columns.",
        )

    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset token not found. Please upload your file via "
                "POST /upload/dataset first."
            ),
        )

    try:
        raw_bytes = __import__("base64").b64decode(ctx["base64_content"])
        df = read_file(raw_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read dataset: {exc}") from exc

    genotype_col: Optional[str] = ctx.get("genotype_column")
    if not genotype_col or genotype_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Genotype column not found in dataset.",
        )

    missing_traits = [t for t in request.trait_columns if t not in df.columns]
    if missing_traits:
        raise HTTPException(
            status_code=400,
            detail=f"Trait column(s) not found: {missing_traits}",
        )

    # Coerce traits to numeric
    df = df.copy()
    for t in request.trait_columns:
        df[t] = pd.to_numeric(df[t], errors="coerce")

    # Validate method
    valid_methods = set(_LINKAGE_MAP.keys())
    if request.method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown linkage method '{request.method}'. "
                f"Valid options: {sorted(valid_methods)}"
            ),
        )

    # Validate k if provided
    if request.k is not None and request.k < 2:
        raise HTTPException(
            status_code=400,
            detail="k must be at least 2.",
        )

    try:
        result = _compute_clusters(
            df=df,
            trait_cols=request.trait_columns,
            genotype_col=genotype_col,
            method=request.method,
            k=request.k,
            scale=request.scale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Cluster computation error")
        raise HTTPException(
            status_code=503,
            detail=f"Cluster analysis failed: {exc}",
        ) from exc

    interpretation = _generate_cluster_interpretation(result)

    cluster_assignments = [
        GenotypeCluster(
            genotype=a["genotype"],
            cluster_id=a["cluster_id"],
            silhouette_score=a["silhouette_score"],
        )
        for a in result["assignments"]
    ]

    cluster_summary_pydantic = [
        ClusterSummary(
            cluster_id=cs["cluster_id"],
            size=cs["size"],
            mean_per_trait=cs["mean_per_trait"],
        )
        for cs in result["cluster_summary"]
    ]

    return ClusterResponse(
        status="success",
        n_genotypes=result["n_genotypes"],
        n_traits=result["n_traits"],
        method=result["method"],
        optimal_k=result["optimal_k"],
        cluster_assignments=cluster_assignments,
        cluster_summary=cluster_summary_pydantic,
        silhouette_scores=result["silhouette_scores"],
        dendrogram_data=result["dendrogram_data"],
        interpretation=interpretation,
    )
