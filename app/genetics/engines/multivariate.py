"""
Multivariate analysis:
  - PCA on standardised genotype × trait means matrix
  - Hierarchical (Ward) clustering
  - k-means clustering

All computations on the genotype-mean matrix, not on raw observations.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — falling back to numpy SVD for PCA")


class MultivariateEngine:
    """PCA + Ward hierarchical + k-means on breeding trial genotype means."""

    def __init__(self, config):
        self.config = config

    def run_all(
        self,
        df: pd.DataFrame,
        trait_cols: List[str],
        geno_col: str,
        loc_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute PCA, hierarchical clustering, and k-means on genotype means.
        Returns results + per-location PCA if loc_col is provided.
        """
        geno_means = df.groupby(geno_col)[trait_cols].mean().dropna()
        n_geno = len(geno_means)
        n_traits = len(trait_cols)

        if n_geno < 4:
            return {"status": "error", "error": "Need ≥4 genotypes for multivariate analysis"}

        k = min(self.config.n_clusters, n_geno - 1)

        pca_result = self._pca(geno_means, trait_cols)
        hier_result = self._hierarchical(geno_means, k)
        km_result = self._kmeans(geno_means, pca_result, k)

        per_loc: Dict[str, Any] = {}
        if loc_col and loc_col in df.columns and self.config.include_per_location:
            for loc, gdf in df.groupby(loc_col):
                gm_loc = gdf.groupby(geno_col)[trait_cols].mean().dropna()
                if len(gm_loc) >= 4:
                    per_loc[str(loc)] = {
                        "n_genotypes": len(gm_loc),
                        "pca": self._pca(gm_loc, trait_cols),
                    }

        return {
            "status": "success",
            "n_genotypes": n_geno,
            "n_traits": n_traits,
            "input_matrix_description": "Genotype × mean-trait matrix (averaged over locations and reps)",
            "pca": pca_result,
            "hierarchical_clustering": hier_result,
            "kmeans_clustering": km_result,
            "per_location": per_loc if per_loc else None,
        }

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def _pca(self, geno_means: pd.DataFrame, trait_cols: List[str]) -> Dict[str, Any]:
        X = geno_means[trait_cols].values
        n, p = X.shape

        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            Xz = scaler.fit_transform(X)
            pca = SklearnPCA(n_components=min(n, p))
            scores_mat = pca.fit_transform(Xz)
            loadings_mat = pca.components_.T         # shape: p × n_components
            explained = pca.explained_variance_ratio_.tolist()
        else:
            # Fallback: manual Z-score + SVD
            Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
            U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
            total_var = float((s ** 2).sum())
            explained = [(float(sv ** 2) / total_var) for sv in s]
            scores_mat = U * s
            loadings_mat = Vt.T

        n_pcs = min(p, n - 1, scores_mat.shape[1])
        genotype_names = list(geno_means.index)

        # Scores dict
        scores: Dict[str, Dict] = {}
        for i, geno in enumerate(genotype_names):
            scores[geno] = {f"PC{k+1}": round(float(scores_mat[i, k]), 6) for k in range(n_pcs)}

        # Loadings dict
        loadings: Dict[str, Dict] = {}
        for k in range(n_pcs):
            loadings[f"PC{k+1}"] = {
                trait_cols[j]: round(float(loadings_mat[j, k]), 6)
                for j in range(len(trait_cols))
            }

        cumulative = 0.0
        ev_list = []
        for k in range(n_pcs):
            pct = explained[k] * 100 if k < len(explained) else 0.0
            cumulative += pct
            ev_list.append({
                "PC": f"PC{k+1}",
                "percent": round(pct, 2),
                "cumulative_percent": round(cumulative, 2),
            })

        # Biplot data
        biplot_points = [
            {"id": geno,
             "x": scores[geno].get("PC1", 0),
             "y": scores[geno].get("PC2", 0),
             "type": "genotype"}
            for geno in genotype_names
        ]
        loading_vectors = [
            {"trait": t,
             "x": loadings.get("PC1", {}).get(t, 0),
             "y": loadings.get("PC2", {}).get(t, 0),
             "type": "loading"}
            for t in trait_cols
        ]

        pc1_pct = ev_list[0]["percent"] if ev_list else 0
        pc2_pct = ev_list[1]["percent"] if len(ev_list) > 1 else 0

        return {
            "method": "PCA on Z-score standardised genotype × mean-trait matrix",
            "n_components": n_pcs,
            "explained_variance_ratio": ev_list,
            "loadings": loadings,
            "scores": scores,
            "biplot_data": {
                "x_axis_label": f"PC1 ({pc1_pct:.1f}%)",
                "y_axis_label": f"PC2 ({pc2_pct:.1f}%)",
                "genotype_points": biplot_points,
                "loading_vectors": loading_vectors,
            },
        }

    # ------------------------------------------------------------------
    # HIERARCHICAL CLUSTERING
    # ------------------------------------------------------------------

    def _hierarchical(self, geno_means: pd.DataFrame, k: int) -> Dict[str, Any]:
        X = geno_means.values
        genotype_names = list(geno_means.index)

        dist_vec = pdist(X, metric="euclidean")
        Z = sch.linkage(dist_vec, method="ward")

        # Cut tree at k clusters
        labels_raw = sch.fcluster(Z, k, criterion="maxclust")
        cluster_labels = {genotype_names[i]: int(labels_raw[i]) for i in range(len(genotype_names))}

        # Convert linkage matrix to nested dict for frontend rendering
        tree = _linkage_to_tree(Z, genotype_names)

        # Per-cluster members
        cluster_members: Dict[str, List] = {}
        for geno, cl in cluster_labels.items():
            cluster_members.setdefault(f"Cluster_{cl}", []).append(geno)

        return {
            "method": "Ward linkage on Euclidean distance of trait means",
            "n_clusters": k,
            "cluster_labels": cluster_labels,
            "cluster_members": cluster_members,
            "dendrogram_data": {
                "description": "Nested tree for frontend dendrogram rendering",
                "linkage_method": "Ward",
                "distance_metric": "Euclidean",
                "tree": tree,
            },
        }

    # ------------------------------------------------------------------
    # K-MEANS
    # ------------------------------------------------------------------

    def _kmeans(
        self,
        geno_means: pd.DataFrame,
        pca_result: Dict[str, Any],
        k: int,
    ) -> Dict[str, Any]:
        genotype_names = list(geno_means.index)
        scores = pca_result.get("scores", {})
        n_pcs = min(2, len(pca_result.get("explained_variance_ratio", [])))

        # Use first 2 PC scores as features for k-means
        X_pc = np.array([
            [scores.get(g, {}).get(f"PC{j+1}", 0.0) for j in range(n_pcs)]
            for g in genotype_names
        ])

        if SKLEARN_AVAILABLE:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_pc)
            centroids = km.cluster_centers_
        else:
            from scipy.cluster.vq import kmeans2
            centroids, labels = kmeans2(X_pc, k, minit="points", seed=42)

        cluster_labels = {genotype_names[i]: int(labels[i]) for i in range(len(genotype_names))}
        cluster_members: Dict[str, List] = {}
        for geno, cl in cluster_labels.items():
            cluster_members.setdefault(f"Cluster_{cl}", []).append(geno)

        centroid_dict = {
            f"Cluster_{k_idx}": {f"PC{j+1}": round(float(centroids[k_idx, j]), 4)
                                  for j in range(n_pcs)}
            for k_idx in range(k)
        }

        return {
            "method": "k-means clustering on PC1/PC2 scores",
            "k": k,
            "cluster_labels": cluster_labels,
            "cluster_members": cluster_members,
            "centroids": centroid_dict,
        }


# ── Helper: scipy linkage matrix → nested dict tree ────────────────────────

def _linkage_to_tree(Z: np.ndarray, leaf_names: List[str]) -> Dict[str, Any]:
    """Convert scipy linkage matrix to a nested dict for frontend D3 dendrograms."""
    n = len(leaf_names)
    nodes: Dict[int, Any] = {i: {"id": leaf_names[i], "height": 0.0, "leaf": True}
                              for i in range(n)}
    for k, row in enumerate(Z):
        left_idx, right_idx, height, _ = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        nodes[n + k] = {
            "id": f"node_{n+k}",
            "height": round(height, 4),
            "leaf": False,
            "children": [nodes[left_idx], nodes[right_idx]],
        }
    root_idx = n + len(Z) - 1
    return nodes.get(root_idx, {})
