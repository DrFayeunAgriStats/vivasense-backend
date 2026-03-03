"""
Configuration dataclasses for the VivaSense genetics analysis package.
Mirrors the AnalysisConfig pattern from main.py.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


@dataclass
class GeneticsConfig:
    """Global configuration for all genetics/breeding analyses."""
    alpha: float = 0.05
    figure_dpi: int = 170                       # V2.2 standard DPI
    figure_format: str = "png"
    n_ammi_axes: int = 2                        # IPCA axes to retain in AMMI
    stability_method: str = "both"              # "eberhart_russell" | "ammi" | "both"
    selection_intensity: float = 2.063          # k for 5% selection pressure
    economic_weights: Optional[Dict[str, float]] = None   # trait -> weight for selection index
    path_target_trait: Optional[str] = None     # dependent variable for path analysis
    n_clusters: int = 3                         # k for k-means + hierarchical cut
    p_value_correction: str = "bonferroni"      # "bonferroni" | "fdr_bh" | "none"
    similarity_metric: str = "both"             # "jaccard" | "dice" | "both"
    include_per_location: bool = True           # compute per-location sub-analyses

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrialDesign:
    """Declares experimental structure column mappings for a breeding trial."""
    genotype_col: str = "Genotype"
    location_col: Optional[str] = "Location"    # None → single-environment mode
    season_col: Optional[str] = None            # if set, Location+Season are concatenated
    rep_col: str = "Rep"
    block_col: Optional[str] = None             # required for Alpha lattice designs
    trait_cols: Optional[List[str]] = None      # None → all remaining numeric columns
    design_type: str = "RCBD"                   # "CRD" | "RCBD" | "Alpha"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarkerConfig:
    """Configuration for molecular marker (SSR/RAPD/AFLP) analysis."""
    accession_col: str = "Accession"
    marker_prefix: Optional[str] = None         # filter marker columns by prefix
    similarity_metric: str = "both"             # "jaccard" | "dice" | "both"
    n_clusters: int = 3

    def to_dict(self) -> dict:
        return asdict(self)
