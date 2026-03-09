"""
Publication-quality figure generator for VivaSense Genetics module.

Generates matplotlib figures at 300 DPI and returns them as base64-encoded
PNG strings in a list of {"name": str, "caption": str, "image_base64": str} dicts.
"""
from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ── Shared style constants ────────────────────────────────────────────────────
_DPI = 300
_GENO_COLOR  = "#2563EB"  # blue
_ENV_COLOR   = "#DC2626"  # red
_HEAT_CMAP   = "RdYlGn"
_BAR_COLOR   = "#4A90E2"
_EDGE_COLOR  = "#2c5f9e"
_FONT_FAMILY = "DejaVu Serif"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        _DPI,
})


def _encode(fig: plt.Figure) -> str:
    """Save figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


# ── Figure 1: Genotype Means Bar Chart ───────────────────────────────────────

def _fig_genotype_means(envelope: Dict, trait: str, genotype_col: str) -> Optional[str]:
    tables     = envelope.get("tables", {})
    means_recs = tables.get("genotype_means", [])
    if not means_recs:
        return None

    genotypes  = [r.get(genotype_col) or r.get("genotype") or r.get("Genotype", f"G{i}")
                  for i, r in enumerate(means_recs, 1)]
    means_vals = [_safe_float(r.get("mean")) for r in means_recs]
    letters    = [str(r.get("letter") or r.get("tukey_letter") or "") for r in means_recs]

    # sort descending
    pairs = sorted(zip(means_vals, genotypes, letters), key=lambda x: (x[0] or 0), reverse=True)
    means_s, genos_s, letters_s = zip(*pairs) if pairs else ([], [], [])

    # SE from anova (approximate: try to pull from means_recs)
    se_vals = [_safe_float(r.get("SE") or r.get("se") or r.get("sem")) for r in means_recs]
    # re-sort SEs to match sorted order
    pairs_se = sorted(zip(means_vals, se_vals), key=lambda x: (x[0] or 0), reverse=True)
    _, se_s = zip(*pairs_se) if pairs_se else ([], [])

    fig, ax = plt.subplots(figsize=(max(6, len(genos_s) * 0.7 + 2), 5))
    x = np.arange(len(genos_s))
    bars = ax.bar(x, means_s, color=_BAR_COLOR, edgecolor=_EDGE_COLOR,
                  linewidth=0.8, width=0.6, zorder=3)

    # error bars
    valid_se = [s for s in se_s if s is not None]
    if valid_se and len(valid_se) == len(means_s):
        ax.errorbar(x, means_s, yerr=se_s, fmt="none", color="black",
                    capsize=4, linewidth=1.2, zorder=4)

    # mean values inside bars
    for bar, m in zip(bars, means_s):
        if m is None:
            continue
        y_pos = bar.get_height() * 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{m:,.2f}", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")

    # Tukey letter annotations
    for xi, (m, se, let) in enumerate(zip(means_s, se_s, letters_s)):
        if not let or m is None:
            continue
        offset = (float(se) if se else 0) + (max(means_s) - min([v for v in means_s if v])) * 0.03
        ax.text(xi, m + offset, let, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(genos_s, rotation=30 if len(genos_s) > 6 else 0, ha="right")
    ax.set_ylabel(trait)
    ax.set_title(f"Figure 1: Genotype Means for {trait} (±SE)", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    note = ("Note: Bars sorted by mean (descending). Error bars = ±1 SE. "
            "Means with same letter are not significantly different (Tukey HSD, α = 0.05).")
    fig.text(0.01, -0.05, note, fontsize=7, color="#555", style="italic", wrap=True)
    fig.tight_layout()
    return _encode(fig)


# ── Figure 2: AMMI Biplot ─────────────────────────────────────────────────────

def _fig_ammi_biplot(envelope: Dict, trait: str) -> Optional[str]:
    tables    = envelope.get("tables", {})
    ipca_recs = tables.get("ammi_ipca", [])

    # Also check if plots dict already has a biplot from the pipeline
    plots = envelope.get("plots", {})
    if plots.get("ammi_biplot"):
        return None  # already generated, skip

    if not ipca_recs:
        return None

    # Split genotypes and environments
    geno_pts: List[Dict] = []
    env_pts:  List[Dict] = []

    for rec in ipca_recs:
        kind = str(rec.get("type", "")).lower()
        if "env" in kind or "loc" in kind or "environment" in kind:
            env_pts.append(rec)
        else:
            geno_pts.append(rec)

    # Infer from "id" / name if type not set
    if not geno_pts and not env_pts:
        for rec in ipca_recs:
            # environments often have numbers or location-like names
            geno_pts.append(rec)

    def _xy(rec):
        x = _safe_float(rec.get("IPCA1") or rec.get("PC1") or rec.get("x"))
        y = _safe_float(rec.get("IPCA2") or rec.get("PC2") or rec.get("y"))
        return x, y

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    for rec in geno_pts:
        x, y = _xy(rec)
        if x is None or y is None:
            continue
        label = rec.get("id") or rec.get("genotype") or rec.get("name", "")
        ax.scatter(x, y, color=_GENO_COLOR, s=60, zorder=5)
        ax.annotate(str(label), (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color=_GENO_COLOR)

    for rec in env_pts:
        x, y = _xy(rec)
        if x is None or y is None:
            continue
        label = rec.get("id") or rec.get("location") or rec.get("name", "")
        ax.scatter(x, y, color=_ENV_COLOR, marker="^", s=80, zorder=5)
        ax.annotate(str(label), (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color=_ENV_COLOR)

    exp_var = tables.get("ammi_explained_variance", [])
    pc1_pct = pc2_pct = ""
    if len(exp_var) >= 2:
        try:
            pc1_pct = f" ({float(exp_var[0].get('explained_variance', 0))*100:.1f}%)"
            pc2_pct = f" ({float(exp_var[1].get('explained_variance', 0))*100:.1f}%)"
        except Exception:
            pass

    ax.set_xlabel(f"IPCA1{pc1_pct}")
    ax.set_ylabel(f"IPCA2{pc2_pct}")
    ax.set_title(f"Figure 2: AMMI Biplot for {trait}", fontweight="bold")

    legend = [
        mpatches.Patch(color=_GENO_COLOR, label="Genotypes"),
        mpatches.Patch(color=_ENV_COLOR, label="Environments"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _encode(fig)


# ── Figure 3: GGE Biplot ──────────────────────────────────────────────────────

def _fig_gge_biplot(envelope: Dict, trait: str) -> Optional[str]:
    tables = envelope.get("tables", {})

    # Check pipeline-generated biplot first
    plots = envelope.get("plots", {})
    if plots.get("gge_biplot"):
        return None

    wwhere = tables.get("gge_which_won_where", [])
    if not wwhere:
        return None

    geno_pts: List[Dict] = []
    env_pts:  List[Dict] = []

    for rec in wwhere:
        kind = str(rec.get("type", "")).lower()
        if "env" in kind or "loc" in kind:
            env_pts.append(rec)
        else:
            geno_pts.append(rec)

    if not geno_pts and not env_pts:
        geno_pts = wwhere

    def _xy_gge(rec):
        x = _safe_float(rec.get("PC1") or rec.get("IPCA1") or rec.get("x"))
        y = _safe_float(rec.get("PC2") or rec.get("IPCA2") or rec.get("y"))
        return x, y

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    for rec in geno_pts:
        x, y = _xy_gge(rec)
        if x is None or y is None:
            continue
        lbl = rec.get("genotype") or rec.get("id") or rec.get("name", "")
        won = rec.get("won_in", [])
        color = "#D97706" if won else _GENO_COLOR
        ax.scatter(x, y, color=color, s=60, zorder=5)
        ax.annotate(str(lbl), (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)

    for rec in env_pts:
        x, y = _xy_gge(rec)
        if x is None or y is None:
            continue
        lbl = rec.get("location") or rec.get("id") or rec.get("name", "")
        ax.scatter(x, y, color=_ENV_COLOR, marker="^", s=80, zorder=5)
        ax.annotate(str(lbl), (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color=_ENV_COLOR)

    ax.set_xlabel("GGE PC1")
    ax.set_ylabel("GGE PC2")
    ax.set_title(f"Figure 3: GGE Biplot for {trait} (Which-Won-Where)", fontweight="bold")
    legend = [
        mpatches.Patch(color="#D97706", label="Sector winner"),
        mpatches.Patch(color=_GENO_COLOR, label="Genotype"),
        mpatches.Patch(color=_ENV_COLOR, label="Environment"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _encode(fig)


# ── Figure 4: Correlation Heatmap ─────────────────────────────────────────────

def _fig_correlation_heatmap(envelope: Dict) -> Optional[str]:
    tables = envelope.get("tables", {})
    corr   = tables.get("correlations")
    if not corr:
        return None

    matrix: Optional[Dict] = None
    if isinstance(corr, dict):
        matrix = corr.get("matrix") or corr.get("phenotypic_matrix")
        if matrix is None and corr:
            first = next(iter(corr.values()), None)
            if isinstance(first, dict):
                matrix = corr

    if not matrix:
        return None

    traits = list(matrix.keys())
    n = len(traits)
    if n < 2:
        return None

    # Build numeric matrix
    mat = np.zeros((n, n))
    ann = [[""] * n for _ in range(n)]

    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            if i == j:
                mat[i, j] = 1.0
                ann[i][j] = "1.000"
            else:
                cell = (matrix.get(t1) or {}).get(t2)
                if cell is None:
                    cell = (matrix.get(t2) or {}).get(t1)
                if isinstance(cell, dict):
                    r   = _safe_float(cell.get("r"))
                    p   = _safe_float(cell.get("p_value"))
                    mat[i, j] = r if r is not None else 0.0
                    stars = ("***" if p is not None and p < 0.001 else
                             "**"  if p is not None and p < 0.01  else
                             "*"   if p is not None and p < 0.05  else "ns")
                    ann[i][j] = f"{mat[i,j]:.3f}\n{stars}"
                elif isinstance(cell, (int, float)):
                    mat[i, j] = float(cell)
                    ann[i][j] = f"{mat[i,j]:.3f}"

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(5, n * 0.9 + 2), max(4, n * 0.9 + 1)))
        sns.heatmap(
            mat, annot=ann, fmt="", cmap=_HEAT_CMAP,
            vmin=-1, vmax=1, square=True,
            xticklabels=traits, yticklabels=traits,
            linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        )
        ax.set_title("Figure 4: Phenotypic Correlation Heatmap", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        fig.tight_layout()
        return _encode(fig)
    except ImportError:
        # fallback without seaborn
        fig, ax = plt.subplots(figsize=(max(5, n * 0.9 + 2), max(4, n * 0.9 + 1)))
        im = ax.imshow(mat, cmap=_HEAT_CMAP, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(traits, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(traits, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=7)
        ax.set_title("Figure 4: Phenotypic Correlation Heatmap", fontweight="bold")
        fig.tight_layout()
        return _encode(fig)


# ── Figure 5: Stability Scatter Plot ─────────────────────────────────────────

def _fig_stability_scatter(envelope: Dict, trait: str) -> Optional[str]:
    """Mean vs regression coefficient (bi) scatter — Eberhart & Russell stability."""
    tables    = envelope.get("tables", {})
    stab_recs = tables.get("stability", [])
    if not stab_recs:
        return None

    labels: List[str] = []
    means:  List[float] = []
    bis:    List[float] = []
    classes: List[str] = []

    for rec in stab_recs:
        gid  = str(rec.get("genotype") or rec.get("Genotype", "?"))
        mean = _safe_float(rec.get("grand_mean") or rec.get("mean"))
        bi   = rec.get("bi")
        if isinstance(bi, dict):
            bi = bi.get("value")
        bi = _safe_float(bi)
        cls  = str(rec.get("classification", "")).lower()
        if mean is None or bi is None:
            continue
        labels.append(gid); means.append(mean); bis.append(bi); classes.append(cls)

    if len(labels) < 2:
        return None

    color_map = {
        "stable":   "#22c55e",
        "adaptable": "#3b82f6",
        "unstable": "#ef4444",
        "":         "#94a3b8",
    }
    colors = []
    for cls in classes:
        matched = next((c for c in color_map if c and c in cls), "")
        colors.append(color_map[matched])

    grand_mean = float(np.mean(means))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bis, means, c=colors, s=80, zorder=5, edgecolors="#333", linewidth=0.5)

    for xi, yi, lbl in zip(bis, means, labels):
        ax.annotate(lbl, (xi, yi), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)

    # reference lines
    ax.axhline(grand_mean, color="grey", linestyle="--", linewidth=0.8, label=f"Grand mean ({grand_mean:.2f})")
    ax.axvline(1.0, color="#d97706", linestyle="--", linewidth=0.8, label="b_i = 1 (average)")

    # quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    mid_x = 1.0; mid_y = grand_mean
    ax.text(xlim[1], ylim[1], "Responsive\n& Productive", ha="right", va="top", fontsize=7, color="#555")
    ax.text(xlim[0], ylim[1], "Low Responsive\n& Productive", ha="left", va="top", fontsize=7, color="#555")
    ax.text(xlim[1], ylim[0], "Responsive\n& Below Avg", ha="right", va="bottom", fontsize=7, color="#555")
    ax.text(xlim[0], ylim[0], "Low Responsive\n& Below Avg", ha="left", va="bottom", fontsize=7, color="#555")

    ax.set_xlabel("Regression Coefficient (b_i)")
    ax.set_ylabel(f"Mean {trait}")
    ax.set_title(f"Figure 5: Stability Scatter Plot for {trait}\n(Eberhart & Russell, 1966)",
                 fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#22c55e", label="Stable"),
        mpatches.Patch(color="#3b82f6", label="Broadly adaptable"),
        mpatches.Patch(color="#ef4444", label="Unstable"),
        mpatches.Patch(color="#94a3b8", label="Unclassified"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _encode(fig)


# ── Figure 6: PCA Biplot ──────────────────────────────────────────────────────

def _fig_pca_biplot(envelope: Dict, trait: str) -> Optional[str]:
    """Genotype PCA scores biplot with trait loading vectors."""
    tables = envelope.get("tables", {})
    mv     = tables.get("multivariate")

    # Check pipeline-generated biplot first
    plots = envelope.get("plots", {})
    if plots.get("pca_biplot"):
        return None

    if not mv or not isinstance(mv, dict):
        return None

    pca = mv.get("pca")
    if not pca or not isinstance(pca, dict):
        return None

    scores   = pca.get("scores") or pca.get("genotype_scores")
    loadings = pca.get("loadings") or pca.get("trait_loadings")
    geno_ids = pca.get("genotype_ids") or pca.get("labels") or []
    traits   = pca.get("trait_names") or pca.get("traits") or []
    explained = pca.get("explained") or pca.get("explained_variance_ratio") or []

    if scores is None:
        return None

    try:
        sc = np.array(scores)
        if sc.ndim != 2 or sc.shape[1] < 2:
            return None
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    # Genotype scores
    ax.scatter(sc[:, 0], sc[:, 1], color=_GENO_COLOR, s=50, zorder=5, alpha=0.8)
    for i, gid in enumerate(geno_ids[:len(sc)]):
        ax.annotate(str(gid), (sc[i, 0], sc[i, 1]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)

    # Trait loading vectors
    if loadings is not None:
        try:
            lo = np.array(loadings)
            scale = np.max(np.abs(sc[:, :2])) / np.max(np.abs(lo[:, :2])) * 0.7 if lo.size else 1
            for j, tname in enumerate(traits[:len(lo)]):
                ax.annotate("",
                    xy=(lo[j, 0] * scale, lo[j, 1] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=_ENV_COLOR, lw=1.2))
                ax.text(lo[j, 0] * scale * 1.05, lo[j, 1] * scale * 1.05,
                        str(tname), fontsize=7, color=_ENV_COLOR, fontweight="bold")
        except Exception:
            pass

    pc1_pct = f" ({float(explained[0])*100:.1f}%)" if len(explained) > 0 else ""
    pc2_pct = f" ({float(explained[1])*100:.1f}%)" if len(explained) > 1 else ""
    ax.set_xlabel(f"PC1{pc1_pct}")
    ax.set_ylabel(f"PC2{pc2_pct}")
    ax.set_title("Figure 6: PCA Biplot (Genotype Scores + Trait Loadings)", fontweight="bold")

    legend = [
        mpatches.Patch(color=_GENO_COLOR, label="Genotypes"),
        mpatches.Patch(color=_ENV_COLOR, label="Trait loadings"),
    ]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()
    return _encode(fig)


# ── Figure 7: Dendrogram ──────────────────────────────────────────────────────

def _fig_dendrogram(envelope: Dict) -> Optional[str]:
    """Hierarchical clustering dendrogram from multivariate results."""
    tables = envelope.get("tables", {})
    mv     = tables.get("multivariate")

    # Check pipeline-generated dendrogram first
    plots = envelope.get("plots", {})
    if plots.get("dendrogram"):
        return None

    if not mv or not isinstance(mv, dict):
        return None

    hc = mv.get("hierarchical_clustering") or mv.get("hierarchical")
    if not hc or not isinstance(hc, dict):
        return None

    linkage_mat = hc.get("linkage_matrix") or hc.get("linkage")
    labels      = hc.get("labels") or hc.get("genotype_ids") or []

    if linkage_mat is None:
        return None

    try:
        from scipy.cluster.hierarchy import dendrogram as _dendrogram
        lm = np.array(linkage_mat)
        if lm.ndim != 2 or lm.shape[1] < 4:
            return None
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5 + 3), 5))
    _dendrogram(lm, labels=labels if labels else None,
                leaf_rotation=45, leaf_font_size=8,
                color_threshold=0.7 * max(lm[:, 2]),
                ax=ax)
    ax.set_title("Figure 7: Hierarchical Clustering Dendrogram (Genotypes)", fontweight="bold")
    ax.set_xlabel("Genotype")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return _encode(fig)


# ── Figure 8: Scree Plot ──────────────────────────────────────────────────────

def _fig_scree_plot(envelope: Dict) -> Optional[str]:
    """PCA scree plot — explained variance per component."""
    tables = envelope.get("tables", {})
    mv     = tables.get("multivariate")
    if not mv or not isinstance(mv, dict):
        return None

    pca = mv.get("pca")
    if not pca or not isinstance(pca, dict):
        return None

    explained = pca.get("explained") or pca.get("explained_variance_ratio") or []
    if not explained:
        return None

    try:
        exp = [float(v) * 100 for v in explained]
    except Exception:
        return None

    cumulative = np.cumsum(exp)
    n = len(exp)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8 + 2), 4))
    ax.bar(x, exp, color=_BAR_COLOR, edgecolor=_EDGE_COLOR, width=0.6, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color="#DC2626", marker="o", linewidth=1.5, label="Cumulative")
    ax2.axhline(80, color="#888", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Cumulative Variance Explained (%)", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    ax2.set_ylim(0, 105)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Figure 8: PCA Scree Plot", fontweight="bold")
    ax.set_xticks(x)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _encode(fig)


# ── Public API ────────────────────────────────────────────────────────────────

def build_publication_figures(
    envelope: Dict[str, Any],
    trait: str,
    genotype_col: str = "Genotype",
) -> List[Dict[str, str]]:
    """
    Generate all publication-quality figures for a genetics trial.

    Returns:
        List of {"name": str, "caption": str, "image_base64": str} dicts.
        Figures that cannot be generated (missing data) are silently omitted.
    """
    result: List[Dict[str, str]] = []

    figure_specs = [
        (
            "Genotype Means Bar Chart",
            f"Figure 1: Mean performance of genotypes for {trait} with ±SE error bars "
            "and Tukey HSD letter groupings. Bars sorted by mean (descending).",
            lambda: _fig_genotype_means(envelope, trait, genotype_col),
        ),
        (
            "AMMI Biplot",
            f"Figure 2: AMMI biplot for {trait} showing genotype (●) and environment "
            "(▲) IPCA1 vs IPCA2 scores.",
            lambda: _fig_ammi_biplot(envelope, trait),
        ),
        (
            "GGE Biplot (Which-Won-Where)",
            f"Figure 3: GGE biplot for {trait}. Polygon vertices (■) indicate "
            "which-won-where sectors for mega-environment delineation.",
            lambda: _fig_gge_biplot(envelope, trait),
        ),
        (
            "Phenotypic Correlation Heatmap",
            "Figure 4: Phenotypic correlation heatmap across traits. "
            "Cell values = Pearson r; significance codes: *** p<0.001, ** p<0.01, * p<0.05.",
            lambda: _fig_correlation_heatmap(envelope),
        ),
        (
            "Stability Scatter Plot",
            f"Figure 5: Eberhart & Russell (1966) stability scatter plot for {trait}. "
            "X-axis = regression coefficient (bᵢ); Y-axis = genotype mean. "
            "Reference lines: bᵢ = 1 and grand mean.",
            lambda: _fig_stability_scatter(envelope, trait),
        ),
        (
            "PCA Biplot",
            "Figure 6: Principal Component Analysis biplot of genotype scores "
            "with trait loading vectors. Percentages = variance explained.",
            lambda: _fig_pca_biplot(envelope, trait),
        ),
        (
            "Hierarchical Dendrogram",
            "Figure 7: UPGMA hierarchical clustering dendrogram based on "
            "standardised multi-trait Euclidean distances.",
            lambda: _fig_dendrogram(envelope),
        ),
        (
            "PCA Scree Plot",
            "Figure 8: PCA scree plot showing individual (bars) and cumulative "
            "(line) variance explained per principal component.",
            lambda: _fig_scree_plot(envelope),
        ),
    ]

    for name, caption, gen_fn in figure_specs:
        try:
            img = gen_fn()
            if img:
                result.append({"name": name, "caption": caption, "image_base64": img})
        except Exception as exc:
            logger.warning("Could not generate figure '%s': %s", name, exc)

    return result
