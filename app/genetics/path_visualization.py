"""
Publication-quality path diagram generator for VivaSense genetics module.

Outputs
-------
png_base64   : base64-encoded PNG (200 DPI, white background) — embed in JSON API
               response directly as ``data:image/png;base64,...``.
plotly_json  : Plotly figure serialised as JSON string — frontend renders with
               ``Plotly.react(div, JSON.parse(plotly_json).data, JSON.parse(plotly_json).layout)``.
               No ``kaleido`` or server-side rendering required.

Layout convention
-----------------
  Left   (x ≈ 1.5) : predictor trait boxes, evenly spaced vertically
  Right  (x ≈ 8.5) : target trait box, vertically centred
  Bottom (x ≈ 4.5) : residual box, dashed arrow up to target

Arrow style
-----------
  Positive coefficient → blue (#1565C0)
  Negative coefficient → red  (#C62828)
  Residual            → gray  (#757575), dashed
  Line width          → proportional to |coefficient|, clamped [1, 5]
  Label               → coefficient value, placed at arrow midpoint
"""
from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

logger = logging.getLogger(__name__)

# ── Style constants ────────────────────────────────────────────────────────────
_DPI        = 200
_PRED_COLOR = "#B8D4F0"   # predictor box fill (light blue)
_PRED_EDGE  = "#2C5F8A"   # predictor box border
_TGT_COLOR  = "#FFD27F"   # target box fill (amber)
_TGT_EDGE   = "#B8860B"   # target box border
_RES_COLOR  = "#E8E8E8"   # residual box fill (light grey)
_RES_EDGE   = "#808080"
_POS_ARROW  = "#1565C0"   # positive effect (blue)
_NEG_ARROW  = "#C62828"   # negative effect (red)
_RES_ARROW  = "#757575"   # residual arrow (grey)

_NODE_W = 2.4   # box half-width  (full box width = 2 × _NODE_W * ... actually used as radius)
_NODE_H = 0.30  # box half-height


def _lw(coeff: float, scale: float = 6.0) -> float:
    """Line width proportional to |coefficient|, clamped to [0.8, 5.0]."""
    return float(np.clip(abs(coeff) * scale, 0.8, 5.0))


def _encode(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Matplotlib path diagram ────────────────────────────────────────────────────

def _draw_box(
    ax: plt.Axes,
    cx: float, cy: float,
    label: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.5,
) -> None:
    """Draw a rounded rectangle with a centred text label."""
    box = FancyBboxPatch(
        (cx - _NODE_W, cy - _NODE_H),
        2 * _NODE_W, 2 * _NODE_H,
        boxstyle="round,pad=0.08",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.4, zorder=5,
    )
    ax.add_patch(box)
    # Wrap long labels
    wrapped = "\n".join(
        label[i: i + 20] for i in range(0, len(label), 20)
    ) if len(label) > 20 else label
    ax.text(
        cx, cy, wrapped,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color="#1a1a1a", zorder=6, wrap=False,
    )


def _draw_arrow(
    ax: plt.Axes,
    x0: float, y0: float,
    x1: float, y1: float,
    coeff: float,
    color: str,
    linestyle: str = "-",
) -> None:
    """Draw arrow from box-right-edge to box-left-edge with a coefficient label."""
    sx = x0 + _NODE_W   # start: right edge of source
    ex = x1 - _NODE_W   # end:   left edge of target

    lw = _lw(coeff)

    ax.annotate(
        "",
        xy=(ex, y1), xytext=(sx, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=linestyle,
            mutation_scale=14,
            shrinkA=0, shrinkB=0,
        ),
        zorder=4,
    )

    mid_x = (sx + ex) / 2.0
    mid_y = (y0 + y1) / 2.0
    # Offset label above the arrow (unless horizontal)
    dy = 0.18 if abs(y1 - y0) > 0.05 else 0.22
    ax.text(
        mid_x, mid_y + dy,
        f"{coeff:+.3f}",
        ha="center", va="bottom",
        fontsize=8.0, fontweight="bold", color=color, zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white", edgecolor="none", alpha=0.85,
        ),
    )


def make_path_diagram_png(
    target_trait: str,
    predictor_traits: List[str],
    p_direct: Dict[str, float],
    residual: float,
    r_squared: float,
) -> str:
    """
    Render a path diagram and return a base64-encoded PNG string
    (without the ``data:image/png;base64,`` prefix).

    Parameters
    ----------
    target_trait     : name of the dependent variable
    predictor_traits : ordered list of predictor names
    p_direct         : {predictor: path_coefficient}
    residual         : residual path coefficient  = sqrt(1 − R²)
    r_squared        : coefficient of determination
    """
    n = len(predictor_traits)
    fig_h = max(4.5, 1.0 + n * 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    # Coordinate system: x ∈ [−0.5, 11.5], y ∈ [−1.6, n]
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.8, n + 0.2)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Node x positions
    pred_cx = 1.5
    tgt_cx  = 9.0
    res_cx  = 5.0
    res_cy  = -1.2

    # Predictor y positions — evenly spaced so centre of distribution = target y
    if n == 1:
        pred_ys = [0.0]
    else:
        pred_ys = list(np.linspace(0, n - 1, n))

    tgt_cy = np.mean(pred_ys)

    # ── Draw predictor nodes ──────────────────────────────────────────────────
    for pred, py in zip(predictor_traits, pred_ys):
        _draw_box(ax, pred_cx, py, pred, _PRED_COLOR, _PRED_EDGE)

    # ── Draw target node ──────────────────────────────────────────────────────
    _draw_box(ax, tgt_cx, tgt_cy, target_trait, _TGT_COLOR, _TGT_EDGE, fontsize=9.5)

    # ── Draw residual node ────────────────────────────────────────────────────
    res_label = f"Residual\n({(1 - r_squared) * 100:.1f}% unmeasured)"
    _draw_box(ax, res_cx, res_cy, res_label, _RES_COLOR, _RES_EDGE, fontsize=7.5)

    # ── Draw predictor → target arrows ───────────────────────────────────────
    for pred, py in zip(predictor_traits, pred_ys):
        coeff = p_direct.get(pred, 0.0)
        color = _POS_ARROW if coeff >= 0 else _NEG_ARROW
        _draw_arrow(ax, pred_cx, py, tgt_cx, tgt_cy, coeff, color)

    # ── Draw residual → target arrow (dashed) ────────────────────────────────
    if abs(residual) > 1e-4:
        # Residual arrow: from top of res box up-right to target
        sx = res_cx + _NODE_W
        sy = res_cy
        ex = tgt_cx - _NODE_W
        ey = tgt_cy
        lw = _lw(residual, scale=4.0)
        ax.annotate(
            "",
            xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=_RES_ARROW, lw=lw,
                linestyle="dashed",
                mutation_scale=12,
                shrinkA=0, shrinkB=0,
            ),
            zorder=4,
        )
        mid_x = (sx + ex) / 2.0
        mid_y = (sy + ey) / 2.0
        ax.text(
            mid_x - 0.2, mid_y,
            f"p_res = {residual:.3f}",
            ha="right", va="center",
            fontsize=7.5, color=_RES_ARROW, fontstyle="italic", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Path Coefficient Diagram — Effects on {target_trait}   "
        f"(R\u00b2\u2009=\u2009{r_squared:.3f})",
        fontsize=11, fontweight="bold", color="#1a1a1a", pad=8,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_PRED_COLOR, edgecolor=_PRED_EDGE, label="Predictor"),
        mpatches.Patch(facecolor=_TGT_COLOR,  edgecolor=_TGT_EDGE,  label="Target"),
        mpatches.Patch(facecolor=_RES_COLOR,  edgecolor=_RES_EDGE,  label="Residual (unmeasured)"),
        mpatches.Patch(facecolor=_POS_ARROW,  label="Positive effect (→)"),
        mpatches.Patch(facecolor=_NEG_ARROW,  label="Negative effect (→)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout(pad=0.5)
    return _encode(fig)


# ── Plotly path diagram ────────────────────────────────────────────────────────

def make_path_diagram_plotly_json(
    target_trait: str,
    predictor_traits: List[str],
    p_direct: Dict[str, float],
    residual: float,
    r_squared: float,
) -> str:
    """
    Build an interactive Plotly path diagram and return it as a JSON string.
    The frontend renders it with::

        const fig = JSON.parse(plotly_json);
        Plotly.react(div, fig.data, fig.layout);

    No ``kaleido`` or server-side rendering required.
    Returns empty string ``""`` if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        import json as _json
    except ImportError:
        logger.warning("plotly not installed; path diagram JSON unavailable.")
        return ""

    n = len(predictor_traits)

    # Node positions
    PRED_X  = 1.0
    TGT_X   = 8.0
    RES_X   = 4.5

    if n == 1:
        pred_ys = [0.0]
    else:
        pred_ys = list(np.linspace(0, (n - 1) * 2.0, n))

    tgt_y = float(np.mean(pred_ys))
    res_y = min(pred_ys) - 3.0

    fig = go.Figure()

    # ── Arrow annotations (predictor → target) ───────────────────────────────
    for pred, py in zip(predictor_traits, pred_ys):
        coeff  = p_direct.get(pred, 0.0)
        color  = "#1565C0" if coeff >= 0 else "#C62828"
        lw     = float(np.clip(abs(coeff) * 8, 1.0, 8.0))

        fig.add_annotation(
            x=TGT_X, y=tgt_y,
            ax=PRED_X, ay=py,
            axref="x", ayref="y",
            xref="x",  yref="y",
            arrowcolor=color, arrowwidth=lw,
            arrowhead=2, arrowsize=1.2,
            showarrow=True, text="",
        )

        # Coefficient label
        mid_x = (PRED_X + TGT_X) / 2.0
        mid_y = (py + tgt_y) / 2.0
        offset = 0.35 if abs(py - tgt_y) > 0.1 else 0.5
        fig.add_annotation(
            x=mid_x, y=mid_y + offset,
            text=f"<b>{coeff:+.3f}</b>",
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(200,200,200,0.6)",
            borderwidth=1, borderpad=3,
            xref="x", yref="y",
        )

    # ── Residual arrow ────────────────────────────────────────────────────────
    if abs(residual) > 1e-4:
        res_lw = float(np.clip(residual * 6, 1.0, 6.0))
        fig.add_annotation(
            x=TGT_X, y=tgt_y,
            ax=RES_X, ay=res_y,
            axref="x", ayref="y",
            xref="x",  yref="y",
            arrowcolor="#757575", arrowwidth=res_lw,
            arrowhead=2, arrowsize=1.0,
            showarrow=True, text="",
        )
        mid_x = (RES_X + TGT_X) / 2.0
        mid_y = (res_y + tgt_y) / 2.0
        fig.add_annotation(
            x=mid_x - 0.5, y=mid_y,
            text=f"<i>p_res = {residual:.3f}</i>",
            showarrow=False,
            font=dict(color="#757575", size=10),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
            xref="x", yref="y",
        )

    # ── Predictor nodes ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[PRED_X] * n,
        y=pred_ys,
        mode="markers+text",
        marker=dict(
            size=52, symbol="square",
            color=_PRED_COLOR,
            line=dict(color=_PRED_EDGE, width=2),
        ),
        text=predictor_traits,
        textposition="middle center",
        textfont=dict(size=10, color="#1a1a1a"),
        hovertemplate="<b>%{text}</b> — predictor<br>"
                      "Direct path: see label<extra></extra>",
        showlegend=True, name="Predictor",
    ))

    # ── Target node ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[TGT_X], y=[tgt_y],
        mode="markers+text",
        marker=dict(
            size=60, symbol="square",
            color=_TGT_COLOR,
            line=dict(color=_TGT_EDGE, width=2),
        ),
        text=[target_trait],
        textposition="middle center",
        textfont=dict(size=11, color="#1a1a1a", family="Arial Black"),
        hovertemplate=(
            f"<b>{target_trait}</b> — target<br>"
            f"R\u00b2 = {r_squared:.4f}<extra></extra>"
        ),
        showlegend=True, name="Target",
    ))

    # ── Residual node ─────────────────────────────────────────────────────────
    unmeasured_pct = (1.0 - r_squared) * 100.0
    fig.add_trace(go.Scatter(
        x=[RES_X], y=[res_y],
        mode="markers+text",
        marker=dict(
            size=46, symbol="square",
            color=_RES_COLOR,
            line=dict(color=_RES_EDGE, width=2),
        ),
        text=["Residual"],
        textposition="middle center",
        textfont=dict(size=9, color="#555555"),
        hovertemplate=(
            f"<b>Residual (unmeasured factors)</b><br>"
            f"p_res = {residual:.4f}<br>"
            f"Explains {unmeasured_pct:.1f}% of variation<extra></extra>"
        ),
        showlegend=True, name="Residual",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    y_min = res_y - 2.0
    y_max = max(pred_ys) + 2.0

    fig.update_layout(
        title=dict(
            text=(
                f"Path Coefficient Diagram \u2014 Effects on {target_trait}"
                f"  (R\u00b2\u2009=\u2009{r_squared:.3f})"
            ),
            font=dict(size=14, color="#1a1a1a"),
            x=0.5,
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.5, 10.5],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[y_min, y_max],
        ),
        height=max(480, 250 + n * 90),
        width=920,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig.to_json()


# ── Public entry point ────────────────────────────────────────────────────────

def build_path_diagram(
    target_trait: str,
    predictor_traits: List[str],
    p_direct: Dict[str, float],
    residual: float,
    r_squared: float,
) -> Dict[str, Any]:
    """
    Generate both a PNG and a Plotly JSON path diagram.

    Returns a dict with:
      png_base64   : base64 PNG string (no data URI prefix)
      plotly_json  : Plotly figure JSON string (empty string if plotly missing)
    """
    result: Dict[str, Any] = {}

    try:
        result["png_base64"] = make_path_diagram_png(
            target_trait, predictor_traits, p_direct, residual, r_squared
        )
    except Exception as exc:
        logger.warning("Path diagram PNG failed: %s", exc)
        result["png_base64"] = ""

    try:
        result["plotly_json"] = make_path_diagram_plotly_json(
            target_trait, predictor_traits, p_direct, residual, r_squared
        )
    except Exception as exc:
        logger.warning("Path diagram Plotly JSON failed: %s", exc)
        result["plotly_json"] = ""

    return result
