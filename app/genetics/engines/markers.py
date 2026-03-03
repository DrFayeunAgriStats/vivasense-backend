"""
Molecular marker analysis (SSR, RAPD, AFLP, SNP binary matrices).

Formulas
--------
Jaccard similarity:    J(A,B)  = |A∩B| / |A∪B|     = n11 / (n11 + n10 + n01)
Dice similarity:       D(A,B)  = 2|A∩B| / (|A|+|B|) = 2·n11 / (2·n11 + n10 + n01)
Genetic distance:      GD      = 1 − J(A,B)

Shannon diversity:     H'      = −Σ pᵢ·ln(pᵢ)        per locus, then averaged
Simpson diversity:     λ       = 1 − Σ pᵢ²            per locus, then averaged
PIC (Polymorphism      PIC     = 1 − Σ pᵢ²            biallelic — same as Simpson
  Information Content):
Nei gene diversity:    Ĥ       = [n/(n−1)] · (1 − Σ pᵢ²)

Dendrogram: UPGMA (average linkage) on genetic distance matrix.
"""
from __future__ import annotations
import logging
import math
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MarkerEngine:
    """Binary marker (0/1) analysis: similarity, diversity, UPGMA dendrogram."""

    def __init__(self, config, marker_config):
        self.config = config
        self.marker_config = marker_config

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Full marker analysis pipeline.
        df must have accession_col + binary marker columns.
        """
        acc_col = self.marker_config.accession_col
        prefix = self.marker_config.marker_prefix

        marker_cols = [
            c for c in df.columns
            if c != acc_col and (prefix is None or str(c).startswith(prefix))
        ]

        df = df.copy()
        df = df.set_index(acc_col)
        M = df[marker_cols].astype(float)   # shape: n_accessions × n_markers
        accessions = list(M.index)
        n_acc = len(accessions)
        n_markers = len(marker_cols)

        # ── Polymorphism ─────────────────────────────────────────────
        freq_1 = M.mean(axis=0)   # frequency of allele 1 per locus
        freq_0 = 1 - freq_1
        polymorphic = ((freq_1 > 0) & (freq_1 < 1)).sum()
        pct_poly = round(100 * polymorphic / n_markers, 2) if n_markers > 0 else 0.0

        missing_pct = round(100 * M.isna().values.sum() / (n_acc * n_markers), 2)

        # Fill missing with locus allele frequency for diversity calcs
        M_filled = M.apply(lambda col: col.fillna(col.mean()))

        # ── Per-locus diversity ──────────────────────────────────────
        per_locus = {}
        for loc in marker_cols:
            p1 = float(freq_1[loc])
            p0 = 1 - p1
            shannon_h = -sum(p * math.log(p) for p in [p1, p0] if p > 0)
            simpson_d = 1 - (p1 ** 2 + p0 ** 2)
            pic = simpson_d  # same formula for biallelic markers
            nei_ĥ_locus = (n_acc / (n_acc - 1)) * simpson_d if n_acc > 1 else 0.0
            per_locus[loc] = {
                "allele_freq_1": round(p1, 4),
                "allele_freq_0": round(p0, 4),
                "shannon_H": {
                    "value": round(shannon_h, 4),
                    "formula": "H' = −Σ pᵢ·ln(pᵢ)",
                },
                "simpson_D": {
                    "value": round(simpson_d, 4),
                    "formula": "D = 1 − Σ pᵢ²",
                },
                "PIC": {
                    "value": round(pic, 4),
                    "formula": "PIC = 1 − Σ pᵢ²  (biallelic)",
                },
                "nei_gene_diversity": {
                    "value": round(nei_ĥ_locus, 4),
                    "formula": "Ĥ = [n/(n−1)] · (1 − Σ pᵢ²)",
                },
                "polymorphic": bool(0 < p1 < 1),
            }

        # Summary diversity (averages over loci)
        mean_H = float(np.mean([per_locus[l]["shannon_H"]["value"] for l in marker_cols]))
        mean_D = float(np.mean([per_locus[l]["simpson_D"]["value"] for l in marker_cols]))
        mean_PIC = float(np.mean([per_locus[l]["PIC"]["value"] for l in marker_cols]))
        mean_Nei = float(np.mean([per_locus[l]["nei_gene_diversity"]["value"] for l in marker_cols]))

        summary_diversity = {
            "mean_shannon_H": round(mean_H, 4),
            "mean_simpson_D": round(mean_D, 4),
            "mean_PIC": round(mean_PIC, 4),
            "mean_nei_gene_diversity": round(mean_Nei, 4),
            "n_polymorphic_loci": int(polymorphic),
            "percent_polymorphic": pct_poly,
        }

        # ── Similarity matrices ───────────────────────────────────────
        metric = self.marker_config.similarity_metric
        X = M_filled.values  # shape: n_acc × n_markers

        jac_matrix = None
        dice_matrix = None

        if metric in ("jaccard", "both"):
            jac_matrix = _jaccard_matrix(X, accessions)
        if metric in ("dice", "both"):
            dice_matrix = _dice_matrix(X, accessions)

        primary_sim = jac_matrix if jac_matrix is not None else dice_matrix
        primary_metric = "jaccard" if jac_matrix is not None else "dice"

        # ── UPGMA dendrogram on genetic distance ─────────────────────
        dist_matrix = 1.0 - np.array([
            [primary_sim["data"][a1][a2] for a2 in accessions]
            for a1 in accessions
        ])
        np.fill_diagonal(dist_matrix, 0.0)

        from scipy.spatial.distance import squareform
        dist_vec = squareform(dist_matrix, checks=False)
        Z = sch.linkage(dist_vec, method="average")  # UPGMA

        # Cut at n_clusters
        k = min(self.marker_config.n_clusters, n_acc - 1)
        labels_raw = sch.fcluster(Z, k, criterion="maxclust")
        cluster_assignments = {accessions[i]: f"Cluster_{int(labels_raw[i])}"
                               for i in range(n_acc)}

        from .multivariate import _linkage_to_tree
        tree = _linkage_to_tree(Z, accessions)

        # Cluster members
        cluster_members: Dict[str, List] = {}
        for acc, cl in cluster_assignments.items():
            cluster_members.setdefault(cl, []).append(acc)

        # ── PCA on binary matrix ─────────────────────────────────────
        pca_mol = _pca_binary(X, accessions, marker_cols)

        return {
            "status": "success",
            "n_accessions": n_acc,
            "n_markers": n_markers,
            "percent_polymorphic": pct_poly,
            "mean_missing_data_percent": missing_pct,
            "per_locus_diversity": per_locus,
            "summary_diversity": summary_diversity,
            "similarity_matrix": jac_matrix or {},
            "dice_matrix": dice_matrix,
            "dendrogram_data": {
                "linkage_method": "UPGMA (Unweighted Pair Group Method with Arithmetic Mean)",
                "distance_metric": f"1 − {primary_metric.capitalize()} similarity",
                "formula": "GD = 1 − J(A,B)",
                "tree": tree,
            },
            "cluster_groups": {
                "method": f"UPGMA clusters at k={k}",
                "n_clusters": k,
                "assignments": cluster_assignments,
                "members": cluster_members,
            },
            "pca_molecular": pca_mol,
        }


# ── Vectorised similarity functions ────────────────────────────────────────

def _jaccard_matrix(X: np.ndarray, accessions: List[str]) -> Dict[str, Any]:
    """
    Compute pairwise Jaccard similarity for all accession pairs.
    J(A,B) = n11 / (n11 + n10 + n01)
    """
    n = len(accessions)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a = X[i]
            b = X[j]
            valid = ~(np.isnan(a) | np.isnan(b))
            a_v, b_v = a[valid], b[valid]
            n11 = float(((a_v == 1) & (b_v == 1)).sum())
            n10 = float(((a_v == 1) & (b_v == 0)).sum())
            n01 = float(((a_v == 0) & (b_v == 1)).sum())
            denom = n11 + n10 + n01
            s = n11 / denom if denom > 0 else 0.0
            sim[i, j] = sim[j, i] = s
        sim[i, i] = 1.0

    mat_dict = {
        accessions[i]: {accessions[j]: round(float(sim[i, j]), 4) for j in range(n)}
        for i in range(n)
    }
    values_flat = [sim[i, j] for i in range(n) for j in range(n) if i != j]
    return {
        "metric": "jaccard",
        "formula": "J(A,B) = |A∩B| / |A∪B| = n11 / (n11 + n10 + n01)",
        "data": mat_dict,
        "range": {"min": round(float(min(values_flat)), 4),
                  "max": round(float(max(values_flat)), 4)} if values_flat else {},
    }


def _dice_matrix(X: np.ndarray, accessions: List[str]) -> Dict[str, Any]:
    """
    Dice similarity: D(A,B) = 2·n11 / (2·n11 + n10 + n01)
    """
    n = len(accessions)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = X[i], X[j]
            valid = ~(np.isnan(a) | np.isnan(b))
            a_v, b_v = a[valid], b[valid]
            n11 = float(((a_v == 1) & (b_v == 1)).sum())
            n10 = float(((a_v == 1) & (b_v == 0)).sum())
            n01 = float(((a_v == 0) & (b_v == 1)).sum())
            denom = 2 * n11 + n10 + n01
            s = (2 * n11) / denom if denom > 0 else 0.0
            sim[i, j] = sim[j, i] = s
        sim[i, i] = 1.0

    mat_dict = {
        accessions[i]: {accessions[j]: round(float(sim[i, j]), 4) for j in range(n)}
        for i in range(n)
    }
    return {
        "metric": "dice",
        "formula": "D(A,B) = 2·n11 / (2·n11 + n10 + n01)",
        "data": mat_dict,
    }


def _pca_binary(
    X: np.ndarray, accessions: List[str], marker_cols: List[str]
) -> Dict[str, Any]:
    """PCA on binary marker matrix (no standardisation — columns are 0/1)."""
    try:
        Xc = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        total = float((s ** 2).sum())
        n_pcs = min(3, len(s))
        ev = [round(100 * s[k] ** 2 / total, 2) for k in range(n_pcs)]
        cumev = [round(sum(ev[: k + 1]), 2) for k in range(n_pcs)]

        scores = {
            accessions[i]: {f"PC{k+1}": round(float(U[i, k] * s[k]), 6) for k in range(n_pcs)}
            for i in range(len(accessions))
        }
        ev_list = [{"PC": f"PC{k+1}", "percent": ev[k], "cumulative_percent": cumev[k]}
                   for k in range(n_pcs)]

        return {
            "method": "PCA on binary marker matrix (column-centred)",
            "explained_variance_ratio": ev_list,
            "scores": scores,
        }
    except Exception as exc:
        return {"error": str(exc)}
