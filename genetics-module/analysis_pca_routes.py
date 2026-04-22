"""
VivaSense – Principal Component Analysis (PCA) Module

POST /analysis/pca

Performs PCA on genotype-mean data across multiple traits to:
  • Identify major axes of variation
  • Reveal trait groupings (positively/negatively correlated traits)
  • Produce genotype scores on PC axes for biplot visualisation

Implementation uses scikit-learn PCA with optional standardisation.

References:
  Pearson, K. (1901). On lines and planes of closest fit to systems of
  points in space. Philosophical Magazine, 2(11), 559-572.

  Jolliffe, I.T. (2002). Principal Component Analysis (2nd ed.). Springer.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import dataset_cache
from module_schemas import (
    BiplotData,
    GenotypeScore,
    PCARequest,
    PCAResponse,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_pca(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str,
    scale: bool,
    n_components: Optional[int],
) -> Dict[str, Any]:
    """
    Aggregate to genotype means, then run PCA.

    Returns a dict with loadings, scores, variance_explained, etc.
    """
    # Aggregate to genotype means
    geno_means = (
        df.groupby(genotype_col)[trait_cols]
        .mean()
        .dropna(how="all")
    )

    # Drop traits that are entirely NaN
    geno_means = geno_means.dropna(axis=1, how="all")
    available_traits = geno_means.columns.tolist()
    if len(available_traits) < 2:
        raise ValueError(
            "PCA requires at least 2 traits with non-missing data after aggregation."
        )

    # Drop genotypes with any remaining NaN
    geno_means = geno_means.dropna()
    n_genotypes = len(geno_means)
    if n_genotypes < 3:
        raise ValueError(
            f"PCA requires at least 3 genotypes with complete data. "
            f"Only {n_genotypes} found after removing missing values."
        )

    X = geno_means.values.astype(float)
    genotype_labels = geno_means.index.tolist()

    # Standardise if requested
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Determine number of components
    n_traits = len(available_traits)
    max_components = min(n_genotypes, n_traits)
    if n_components is None:
        n_components_final = max_components
    else:
        n_components_final = min(int(n_components), max_components)
        if n_components_final < 1:
            raise ValueError(f"n_components must be ≥ 1 (got {n_components}).")

    # Fit PCA
    pca = PCA(n_components=n_components_final)
    scores_array = pca.fit_transform(X_scaled)

    # Variance explained (%)
    variance_explained = [float(round(v * 100, 4)) for v in pca.explained_variance_ratio_]
    cumulative_variance = []
    cum = 0.0
    for v in variance_explained:
        cum += v
        cumulative_variance.append(round(cum, 4))

    # Loadings: shape (n_traits, n_components)
    # pca.components_ has shape (n_components, n_traits); transpose to (n_traits, n_components)
    loadings_array = pca.components_.T   # shape: (n_traits, n_components)
    loadings: Dict[str, List[float]] = {}
    for i, trait in enumerate(available_traits):
        loadings[trait] = [round(float(loadings_array[i, j]), 6) for j in range(n_components_final)]

    # Scores per genotype
    scores: List[Dict[str, Any]] = []
    for geno_idx, geno in enumerate(genotype_labels):
        scores.append({
            "genotype": str(geno),
            "scores": [round(float(scores_array[geno_idx, j]), 6) for j in range(n_components_final)],
        })

    return {
        "n_traits": len(available_traits),
        "n_genotypes": n_genotypes,
        "n_components": n_components_final,
        "variance_explained": variance_explained,
        "cumulative_variance": cumulative_variance,
        "loadings": loadings,
        "scores": scores,
        "trait_names": available_traits,
        "genotype_labels": [str(g) for g in genotype_labels],
    }


def _classify_loading(loading: float) -> str:
    """Return a human-readable description of loading magnitude."""
    abs_l = abs(loading)
    if abs_l >= 0.7:
        return "strongly"
    if abs_l >= 0.4:
        return "moderately"
    return "weakly"


def _generate_pca_interpretation(
    trait: str,
    result: Dict[str, Any],
) -> str:
    """Generate plain-English PCA interpretation."""
    sections: List[tuple] = []
    n_traits = result["n_traits"]
    n_genos = result["n_genotypes"]
    var_exp = result["variance_explained"]
    cum_var = result["cumulative_variance"]
    loadings = result["loadings"]
    n_pcs = result["n_components"]

    # 1. Overview
    pc1_var = var_exp[0] if var_exp else 0.0
    pc2_var = var_exp[1] if len(var_exp) > 1 else 0.0
    pc12_var = cum_var[1] if len(cum_var) > 1 else cum_var[0] if cum_var else 0.0

    overview = (
        f"Principal Component Analysis (PCA) was performed on {n_traits} traits "
        f"across {n_genos} genotypes using genotype means "
        f"{'standardised to unit variance' if True else 'without standardisation'}. "
        f"A total of {n_pcs} principal components were extracted. "
        f"PC1 explained {pc1_var:.1f}% and PC2 explained {pc2_var:.1f}% of total "
        f"phenotypic variance (combined: {pc12_var:.1f}%). "
    )
    # Find how many PCs to explain 80%
    n_80 = next((i + 1 for i, cv in enumerate(cum_var) if cv >= 80.0), n_pcs)
    overview += (
        f"The first {n_80} PC(s) account for \u2265 80% of total variance, "
        f"indicating {'good' if n_80 <= 3 else 'moderate'} dimensionality reduction."
    )
    sections.append(("Overview", overview))

    # 2. PC1 contributions
    if loadings:
        pc1_loadings = {t: loadings[t][0] for t in loadings}
        sorted_by_pc1 = sorted(pc1_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
        top_pc1 = sorted_by_pc1[:min(5, n_traits)]
        pc1_text = (
            f"PC1 ({pc1_var:.1f}% variance) is driven primarily by: "
            + "; ".join(
                f"{t} ({_classify_loading(l)}, loading = {l:+.3f})"
                for t, l in top_pc1
            )
            + ". "
        )
        # Identify direction groups
        pos_traits = [t for t, l in pc1_loadings.items() if l >= 0.4]
        neg_traits = [t for t, l in pc1_loadings.items() if l <= -0.4]
        if pos_traits and neg_traits:
            pc1_text += (
                f"Traits loading positively on PC1 ({', '.join(pos_traits)}) "
                f"are negatively correlated with traits loading negatively "
                f"({', '.join(neg_traits)}), suggesting a trade-off relationship."
            )
        elif pos_traits:
            pc1_text += (
                f"Traits {', '.join(pos_traits)} all load positively on PC1, "
                f"indicating they are positively correlated."
            )
        sections.append(("PC1 Interpretation", pc1_text))

    # 3. PC2 contributions (if available)
    if len(var_exp) > 1 and loadings:
        pc2_loadings = {t: loadings[t][1] for t in loadings}
        sorted_by_pc2 = sorted(pc2_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
        top_pc2 = sorted_by_pc2[:min(5, n_traits)]
        pc2_text = (
            f"PC2 ({pc2_var:.1f}% variance) is driven by: "
            + "; ".join(
                f"{t} (loading = {l:+.3f})"
                for t, l in top_pc2
            )
            + ". "
            "PC2 captures variation orthogonal to PC1 and may reveal a secondary "
            "axis of phenotypic differentiation among genotypes."
        )
        sections.append(("PC2 Interpretation", pc2_text))

    # 4. Genotype Clustering
    cluster_note = (
        f"Genotype scores on PC1 and PC2 can be visualised as a biplot. "
        f"Genotypes that plot close together in PC space are phenotypically "
        f"similar and may originate from related breeding pools. "
        f"Genotypes at the extremes of PC1 or PC2 represent the most "
        f"divergent phenotypes and may contribute complementary traits to "
        f"a crossing programme."
    )
    sections.append(("Genotype Differentiation", cluster_note))

    # 5. Practical Guidance
    guide = (
        "In a plant breeding context, PCA results inform multi-trait selection "
        "strategy: traits with high loadings on the same PC are candidates for "
        "index selection, as improving one tends to improve the others. "
        "Trait pairs with opposite signs on the same PC should be targeted "
        "for genetic recombination to break negative associations. "
        "The biplot data (scores + loadings) can be used directly by the "
        "frontend to render an interactive PC1 vs PC2 biplot."
    )
    sections.append(("Breeding Implications", guide))

    return "\n\n".join(f"{h}\n{c}" for h, c in sections)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/pca",
    response_model=PCAResponse,
    summary="Principal Component Analysis across multiple traits",
)
async def analysis_pca(request: PCARequest) -> PCAResponse:
    """
    Compute PCA on per-genotype means for the selected traits.

    Requires:
      - A dataset_token from POST /upload/dataset
      - At least 2 trait columns
      - At least 3 genotypes with complete trait data
    """
    if len(request.trait_columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="PCA requires at least 2 trait columns.",
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

    # Validate trait columns
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

    try:
        result = _compute_pca(
            df=df,
            trait_cols=request.trait_columns,
            genotype_col=genotype_col,
            scale=request.scale,
            n_components=request.n_components,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PCA computation error")
        raise HTTPException(
            status_code=503,
            detail=f"PCA analysis failed: {exc}",
        ) from exc

    interpretation = _generate_pca_interpretation(
        trait="multi-trait",
        result=result,
    )

    scores_pydantic = [
        GenotypeScore(genotype=s["genotype"], scores=s["scores"])
        for s in result["scores"]
    ]

    biplot_data = BiplotData(
        loadings=result["loadings"],
        scores=scores_pydantic,
    )

    return PCAResponse(
        status="success",
        n_traits=result["n_traits"],
        n_genotypes=result["n_genotypes"],
        variance_explained=result["variance_explained"],
        cumulative_variance=result["cumulative_variance"],
        loadings=result["loadings"],
        scores=scores_pydantic,
        biplot_data=biplot_data,
        interpretation=interpretation,
    )
