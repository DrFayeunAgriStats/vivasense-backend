"""
VivaSense Genetics Analysis Package  — V2.2 aligned
=====================================================
Journal-grade plant breeding statistics with the V2.2 response envelope
{meta, tables, plots, interpretation, strict_template, intelligence}.

Integration (already done in app/main.py):
    from genetics import genetics_router
    app.include_router(genetics_router)

Endpoints (prefix: /analyze/genetics):
    POST /analyze/genetics/trial                ← full trial pipeline (CSV/Excel, FormData)
    POST /analyze/genetics/variance-components  ← σ²g, σ²e, σ²gl, H², GA, GCV, PCV
    POST /analyze/genetics/stability            ← Eberhart-Russell + ASV
    POST /analyze/genetics/ammi                 ← AMMI model + IPCA biplot
    POST /analyze/genetics/gge                  ← GGE biplot + which-won-where
    POST /analyze/genetics/correlations         ← phenotypic/genotypic corr + path + index
    POST /analyze/genetics/multivariate         ← PCA + hierarchical + k-means
    POST /analyze/genetics/markers              ← molecular diversity (Jaccard/Dice/PIC/UPGMA)
    GET  /analyze/genetics/health
"""
from .router import genetics_router
from .pipeline import GeneticsPipeline

__all__ = ["genetics_router", "GeneticsPipeline"]
