"""
Visualization for the genetics package.
All methods return base64-encoded PNG strings at 300 DPI — identical convention
to StatisticalAnalyzer.generate_plots() in main.py.
"""
from __future__ import annotations
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster import hierarchy as sch
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Consistent palette across all genetics plots
_GENO_COLOR = "#2563EB"   # blue
_ENV_COLOR = "#DC2626"    # red
_IDEAL_COLOR = "#D97706"  # amber


def _to_b64(fig: plt.Figure, dpi: int = 300) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


class GeneticsPlotter:
    """All genetics plot generators.  Returns base64 PNG strings."""

    def __init__(self, config):
        self.dpi = config.figure_dpi
        self.alpha = config.alpha

    # ------------------------------------------------------------------
    # AMMI BIPLOT
    # ------------------------------------------------------------------

    def ammi_biplot(self, ammi_result: Dict[str, Any], trait: str) -> str:
        """IPCA1 × IPCA2 biplot: genotype circles + environment triangles."""
        bd = ammi_result.get("biplot_data", {})
        gpts = bd.get("genotype_points", [])
        epts = bd.get("environment_points", [])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")

        for pt in gpts:
            ax.scatter(pt["x"], pt["y"], color=_GENO_COLOR, s=80, zorder=3)
            ax.annotate(pt["id"], (pt["x"], pt["y"]), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=_GENO_COLOR)

        for pt in epts:
            ax.scatter(pt["x"], pt["y"], color=_ENV_COLOR, marker="^", s=100, zorder=3)
            ax.annotate(pt["id"], (pt["x"], pt["y"]), textcoords="offset points",
                        xytext=(5, 3), fontsize=8, color=_ENV_COLOR, fontweight="bold")

        ax.set_xlabel(bd.get("x_axis_label", "IPCA1"), fontsize=11)
        ax.set_ylabel(bd.get("y_axis_label", "IPCA2"), fontsize=11)
        ax.set_title(f"AMMI Biplot — {trait}", fontsize=13, fontweight="bold")

        legend_handles = [
            mpatches.Patch(color=_GENO_COLOR, label="Genotypes"),
            mpatches.Patch(color=_ENV_COLOR, label="Environments"),
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    # ------------------------------------------------------------------
    # GGE BIPLOT
    # ------------------------------------------------------------------

    def gge_biplot(self, gge_result: Dict[str, Any], trait: str) -> str:
        """PC1 × PC2 biplot with environment vectors, polygon, ideal marker."""
        bd = gge_result.get("biplot_data", {})
        gpts = bd.get("genotype_points", [])
        evecs = bd.get("environment_vectors", [])
        polygon = bd.get("polygon_vertices", [])
        ideal = bd.get("ideal_marker", {})

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")

        # Polygon (convex hull of outermost genotypes)
        if len(polygon) >= 3:
            px = [v["x"] for v in polygon] + [polygon[0]["x"]]
            py = [v["y"] for v in polygon] + [polygon[0]["y"]]
            ax.plot(px, py, color="darkgrey", lw=1.0, ls="-", alpha=0.6, zorder=1)

        # Genotype points
        for pt in gpts:
            ax.scatter(pt["x"], pt["y"], color=_GENO_COLOR, s=70, zorder=3)
            ax.annotate(pt["id"], (pt["x"], pt["y"]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color=_GENO_COLOR)

        # Environment vectors from origin
        for ev in evecs:
            ax.annotate("", xy=(ev["x"], ev["y"]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=_ENV_COLOR, lw=1.5))
            ax.annotate(ev["id"], (ev["x"], ev["y"]), textcoords="offset points",
                        xytext=(5, 3), fontsize=9, color=_ENV_COLOR, fontweight="bold")

        # Ideal marker
        if ideal:
            ax.scatter(ideal.get("x", 0), ideal.get("y", 0),
                       marker="*", color=_IDEAL_COLOR, s=250, zorder=5, label="Ideal genotype")

        ax.set_xlabel(bd.get("x_axis_label", "PC1"), fontsize=11)
        ax.set_ylabel(bd.get("y_axis_label", "PC2"), fontsize=11)
        ax.set_title(f"GGE Biplot — {trait}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    # ------------------------------------------------------------------
    # STABILITY (Eberhart-Russell regression plot)
    # ------------------------------------------------------------------

    def stability_regression(self, stability_result: Dict[str, Any], trait: str) -> str:
        """One regression line per genotype: ȳij vs. environmental index Ij."""
        env_idx = stability_result.get("environmental_indices", {})
        geno_stab = stability_result.get("genotype_stability", [])
        if not env_idx or not geno_stab:
            return ""

        Ij_vals = np.array([d["index"] for d in env_idx.values()])
        x_range = np.linspace(Ij_vals.min() - 0.1, Ij_vals.max() + 0.1, 100)

        fig, ax = plt.subplots(figsize=(11, 7))

        colors = plt.cm.tab20(np.linspace(0, 1, len(geno_stab)))
        grand = stability_result.get("grand_mean", 0)

        for i, entry in enumerate(geno_stab):
            bi = entry.get("bi", {}).get("value", 1.0)
            mean = entry.get("grand_mean", grand)
            y_line = mean + bi * x_range
            ax.plot(x_range, y_line, color=colors[i], lw=1.2, alpha=0.8,
                    label=f"{entry['genotype']} (bi={bi:.2f})")

        ax.axhline(grand, color="black", lw=1.5, ls="--", label="Grand mean")
        ax.axvline(0, color="grey", lw=0.8, ls="--")

        ax.set_xlabel("Environmental Index (Ij = ȳ_.j − ȳ..)", fontsize=10)
        ax.set_ylabel(f"Mean {trait}", fontsize=10)
        ax.set_title(f"Eberhart & Russell Stability — {trait}", fontsize=12, fontweight="bold")

        if len(geno_stab) <= 15:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    def mean_vs_bi(self, stability_result: Dict[str, Any], trait: str) -> str:
        """Scatter: genotype mean yield vs. bi coefficient. 4 quadrants shown."""
        geno_stab = stability_result.get("genotype_stability", [])
        grand = stability_result.get("grand_mean", 0)
        if not geno_stab:
            return ""

        means = [e["grand_mean"] for e in geno_stab]
        bis = [e.get("bi", {}).get("value", 1.0) for e in geno_stab]
        labels = [e["genotype"] for e in geno_stab]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.axhline(grand, color="grey", lw=1.0, ls="--", label="Grand mean")
        ax.axvline(1.0, color="grey", lw=1.0, ls="--", label="bi = 1")
        ax.scatter(bis, means, color=_GENO_COLOR, s=80, zorder=3)
        for x, y, lab in zip(bis, means, labels):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=7)

        ax.set_xlabel("Regression Coefficient (bi)", fontsize=11)
        ax.set_ylabel(f"Mean {trait}", fontsize=11)
        ax.set_title(f"Mean Yield vs. bi — {trait}", fontsize=12, fontweight="bold")

        # Quadrant labels
        xmid, ymid = 1.0, grand
        ax.text(0.02, 0.98, "Stable\nbelow avg", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="grey", alpha=0.7)
        ax.text(0.98, 0.98, "Responsive\nabove avg", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="grey", alpha=0.7)
        ax.text(0.02, 0.02, "Stable\nbelow avg", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="grey", alpha=0.7)
        ax.text(0.98, 0.02, "Responsive\nbelow avg", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="grey", alpha=0.7)

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    # ------------------------------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------------------------------

    def correlation_heatmap(
        self,
        corr_result: Dict[str, Any],
        corr_type: str = "phenotypic",
        title_suffix: str = "",
    ) -> str:
        """Seaborn heatmap of phenotypic or genotypic correlation matrix."""
        matrix_data = corr_result.get("matrix", {})
        if not matrix_data:
            return ""

        traits = list(matrix_data.keys())
        mat = np.array([
            [matrix_data[t1].get(t2, {}).get("r") or matrix_data[t1].get(t2, {}).get("r_g") or 0.0
             for t2 in traits]
            for t1 in traits
        ], dtype=float)

        mask = np.triu(np.ones_like(mat, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(max(6, len(traits) * 1.2), max(5, len(traits))))
        sns.heatmap(
            mat,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=traits,
            yticklabels=traits,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 8},
        )
        title = f"{corr_type.capitalize()} Correlation Matrix"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    # ------------------------------------------------------------------
    # PATH COEFFICIENT DIAGRAM
    # ------------------------------------------------------------------

    def path_diagram(self, path_result: Dict[str, Any], trait: str) -> str:
        """Horizontal bar chart of direct path coefficients."""
        direct = path_result.get("direct_effects", {})
        if not direct:
            return ""

        predictors = list(direct.keys())
        values = [direct[p]["value"] for p in predictors]
        colors = [_GENO_COLOR if v >= 0 else _ENV_COLOR for v in values]

        fig, ax = plt.subplots(figsize=(9, max(4, len(predictors) * 0.6)))
        bars = ax.barh(predictors, values, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Direct Path Coefficient", fontsize=11)
        ax.set_title(f"Path Coefficients → {trait}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    # ------------------------------------------------------------------
    # PCA BIPLOT
    # ------------------------------------------------------------------

    def pca_biplot(
        self,
        pca_result: Dict[str, Any],
        cluster_labels: Optional[Dict[str, int]] = None,
        title: str = "PCA Biplot",
    ) -> str:
        """Genotype scores on PC1/PC2 with trait loading vectors."""
        bd = pca_result.get("biplot_data", {})
        gpts = bd.get("genotype_points", [])
        lvecs = bd.get("loading_vectors", [])

        n_clusters = len(set(cluster_labels.values())) if cluster_labels else 1
        cmap = plt.cm.Set1(np.linspace(0, 0.8, max(n_clusters, 2)))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.axvline(0, color="grey", lw=0.6, ls="--")

        for pt in gpts:
            cl = (cluster_labels or {}).get(pt["id"], 0)
            color = cmap[cl % len(cmap)]
            ax.scatter(pt["x"], pt["y"], color=color, s=70, zorder=3)
            ax.annotate(pt["id"], (pt["x"], pt["y"]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7)

        # Trait loading arrows (scaled)
        scale = 0.85 * max(abs(pt["x"]) for pt in gpts or [{"x": 1}]) if gpts else 1.0
        lx = [v["x"] for v in lvecs]
        ly = [v["y"] for v in lvecs]
        if lx:
            lscale = scale / max(max(abs(x) for x in lx), 1e-6)
            for v in lvecs:
                ax.annotate(
                    "", xy=(v["x"] * lscale, v["y"] * lscale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
                )
                ax.text(v["x"] * lscale * 1.08, v["y"] * lscale * 1.08,
                        v["trait"], fontsize=8, color="darkgreen")

        ax.set_xlabel(bd.get("x_axis_label", "PC1"), fontsize=11)
        ax.set_ylabel(bd.get("y_axis_label", "PC2"), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    def scree_plot(self, pca_result: Dict[str, Any], title: str = "Scree Plot") -> str:
        """Bar + line chart of explained variance per PC."""
        ev = pca_result.get("explained_variance_ratio", [])
        if not ev:
            return ""

        pcs = [e["PC"] for e in ev]
        pcts = [e["percent"] for e in ev]
        cumpcts = [e["cumulative_percent"] for e in ev]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(pcs, pcts, color=_GENO_COLOR, alpha=0.7, label="% Variance")
        ax1.set_ylabel("Variance Explained (%)", fontsize=10)
        ax1.set_xlabel("Principal Component", fontsize=10)

        ax2 = ax1.twinx()
        ax2.plot(pcs, cumpcts, color=_ENV_COLOR, marker="o", lw=2, label="Cumulative %")
        ax2.axhline(80, color="grey", ls="--", lw=0.8, label="80% threshold")
        ax2.set_ylabel("Cumulative Variance (%)", fontsize=10)
        ax2.set_ylim(0, 105)

        ax1.set_title(title, fontsize=12, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
        fig.tight_layout()
        return _to_b64(fig, self.dpi)

    # ------------------------------------------------------------------
    # DENDROGRAM
    # ------------------------------------------------------------------

    def dendrogram(
        self,
        dendrogram_data: Dict[str, Any],
        title: str = "Dendrogram",
        n_leaves: int = 20,
    ) -> str:
        """
        Render dendrogram from scipy linkage tree structure.
        Uses scipy.cluster.hierarchy.dendrogram directly.
        """
        try:
            tree = dendrogram_data.get("tree", {})
            labels = _collect_leaves(tree)
            if not labels or len(labels) < 3:
                return ""

            # Rebuild linkage matrix from tree to use scipy renderer
            from scipy.spatial.distance import squareform
            n = len(labels)
            fig_height = max(6, n * 0.28)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            # We can't easily re-derive the full linkage from the tree dict,
            # so we annotate the tree structure as a text dendrogram instead.
            # For proper rendering, we regenerate distances from leaf structure.
            _draw_tree_text(ax, tree, labels, title)
            fig.tight_layout()
            return _to_b64(fig, self.dpi)
        except Exception as exc:
            logger.warning("Dendrogram rendering failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # SIMILARITY HEATMAP (molecular markers)
    # ------------------------------------------------------------------

    def similarity_heatmap(self, sim_matrix: Dict[str, Any], title: str = "Similarity Matrix") -> str:
        """Seaborn clustermap of pairwise similarity matrix."""
        data = sim_matrix.get("data", {})
        if not data:
            return ""

        accessions = list(data.keys())
        mat = np.array([[data[a1].get(a2, 0.0) for a2 in accessions] for a1 in accessions])

        fig, ax = plt.subplots(figsize=(max(6, len(accessions) * 0.5), max(5, len(accessions) * 0.45)))
        sns.heatmap(
            mat, xticklabels=accessions, yticklabels=accessions,
            cmap="YlOrRd", vmin=0, vmax=1, ax=ax,
            annot=len(accessions) <= 20,
            fmt=".2f" if len(accessions) <= 20 else "",
            linewidths=0.3 if len(accessions) <= 30 else 0,
            annot_kws={"size": 6},
        )
        ax.set_title(f"{sim_matrix.get('metric', '').capitalize()} {title}", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        fig.tight_layout()
        return _to_b64(fig, self.dpi)


# ── Tree helpers ─────────────────────────────────────────────────────────────

def _collect_leaves(node: Dict[str, Any]) -> List[str]:
    if node.get("leaf"):
        return [node["id"]]
    leaves = []
    for child in node.get("children", []):
        leaves.extend(_collect_leaves(child))
    return leaves


def _draw_tree_text(ax: plt.Axes, tree: Dict[str, Any], labels: List[str], title: str):
    """Fallback: render a simple text-based dendrogram using matplotlib."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Linkage Distance", fontsize=10)
    n = len(labels)
    ax.set_ylim(-0.5, n - 0.5)
    for i, lbl in enumerate(labels):
        ax.text(0, i, lbl, ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.1, 1.1)
    ax.axis("off")
