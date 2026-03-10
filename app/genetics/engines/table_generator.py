"""
Publication-quality HTML table generator for VivaSense Genetics module.

Returns a list of {"name": str, "html": str} dicts that the frontend
can display directly (Word-paste-ready, LaTeX-compatible styling).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ── CSS injected once per table (inline so copy-paste works in Word/email) ──

_STYLE = """<style>
.vv-pub-table{border-collapse:collapse;width:100%;font-family:'Times New Roman',Times,serif;font-size:11pt;margin-bottom:8px}
.vv-pub-table caption{font-weight:bold;font-size:12pt;text-align:left;padding-bottom:4px;caption-side:top}
.vv-pub-table th{background-color:#1a3a5c;color:#fff;padding:6px 10px;text-align:center;border:1px solid #bbb;font-size:10pt}
.vv-pub-table td{padding:5px 10px;border:1px solid #ccc;text-align:center;vertical-align:middle}
.vv-pub-table tr:nth-child(even) td{background-color:#f0f4f8}
.vv-pub-table .td-left{text-align:left}
.vv-pub-table .total-row td{font-weight:bold;border-top:2px solid #1a3a5c}
.vv-pub-table .footnote td{font-size:9pt;color:#555;text-align:left;border:none;padding:2px 4px}
.sig-mark{color:#b00;font-weight:bold}
.sig-ns{color:#888}
</style>"""


def _fmt(v, decimals: int = 4) -> str:
    """Safe float formatter; returns '—' for None/NaN/Inf."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return "—"
        return f"{fv:.{decimals}f}"
    except Exception:
        return str(v)


def _fmt_large(v, decimals: int = 2) -> str:
    """Formatter with thousands separator for large values like kg/ha."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return "—"
        return f"{fv:,.{decimals}f}"
    except Exception:
        return str(v)


def _sig(p) -> str:
    """Significance stars."""
    if p is None:
        return ""
    try:
        pf = float(p)
        if pf < 0.001:
            return '<span class="sig-mark">***</span>'
        if pf < 0.01:
            return '<span class="sig-mark">**</span>'
        if pf < 0.05:
            return '<span class="sig-mark">*</span>'
        return '<span class="sig-ns">ns</span>'
    except Exception:
        return ""


def _html_table(
    table_number: str,
    title: str,
    headers: List[str],
    rows: List[List[str]],
    footnotes: Optional[List[str]] = None,
    total_row: Optional[List[str]] = None,
    left_cols: Optional[List[int]] = None,
) -> str:
    """Build a styled HTML table string."""
    left_cols = left_cols or [0]
    parts = [_STYLE, f'<table class="vv-pub-table">',
             f'  <caption>{table_number}: {title}</caption>',
             '  <thead><tr>']
    for h in headers:
        parts.append(f'    <th>{h}</th>')
    parts += ['  </tr></thead>', '  <tbody>']

    for row in rows:
        parts.append('    <tr>')
        for ci, cell in enumerate(row):
            cls = ' class="td-left"' if ci in left_cols else ''
            parts.append(f'      <td{cls}>{cell}</td>')
        parts.append('    </tr>')

    if total_row:
        parts.append('    <tr class="total-row">')
        for ci, cell in enumerate(total_row):
            cls = ' class="td-left"' if ci in left_cols else ''
            parts.append(f'      <td{cls}>{cell}</td>')
        parts.append('    </tr>')

    parts.append('  </tbody>')

    if footnotes:
        parts.append('  <tfoot>')
        colspan = len(headers)
        for fn in footnotes:
            parts.append(f'  <tr class="footnote"><td colspan="{colspan}">{fn}</td></tr>')
        parts.append('  </tfoot>')

    parts.append('</table>')
    return '\n'.join(parts)


# ── Individual table builders ────────────────────────────────────────────────

def _table_variance_components(envelope: Dict, trait: str) -> str:
    tables = envelope.get("tables", {})
    vc_recs = tables.get("variance_components", [])
    vc = vc_recs[0] if vc_recs else {}

    s2g  = vc.get("sigma2_g")
    s2e  = vc.get("sigma2_e")
    s2gl = vc.get("sigma2_gl")
    s2p  = vc.get("sigma2_p")
    gcv  = vc.get("GCV")
    pcv  = vc.get("PCV")
    ecv  = vc.get("ECV")
    gm   = vc.get("grand_mean")

    def _pct(v):
        if v is None or s2p is None:
            return "—"
        try:
            denom = float(s2p)
            return f"{float(v) / denom * 100:.1f}%" if denom != 0 else "—"
        except Exception:
            return "—"

    def _vi(comp, v):
        if v is None or s2p is None:
            return "—"
        try:
            fv, sp = float(v), float(s2p)
        except Exception:
            return "—"
        if comp == "g":
            r = fv / sp if sp else 0
            return ("Very High" if r > 0.70 else "High" if r > 0.50
                    else "Moderate" if r > 0.30 else "Low")
        if comp == "e":
            return ("Minimal" if fv < 0.05 else "Low" if fv < 1
                    else "Moderate" if fv < 5 else "High")
        if comp == "gl":
            return ("Low" if fv < 0.5 else "Moderate" if fv < 2 else "High")
        return "—"

    rows = [
        ["Genetic Variance",              "σ²<sub>g</sub>",  _fmt(s2g, 4), _pct(s2g),  _vi("g",  s2g)],
        ["Environmental Variance",        "σ²<sub>e</sub>",  _fmt(s2e, 4), _pct(s2e),  _vi("e",  s2e)],
        ["G × Location Interaction",      "σ²<sub>gl</sub>", _fmt(s2gl,4), _pct(s2gl), _vi("gl", s2gl)],
        ["Phenotypic Variance",           "σ²<sub>p</sub>",  _fmt(s2p, 4), "100.0%",   "Sum of all components"],
        ["Grand Mean (μ)",                "μ",               _fmt_large(gm, 2), "—",    "—"],
        ["Genotypic CV",                  "GCV (%)",         _fmt(gcv, 2), "—",         "Genetic variation / mean"],
        ["Phenotypic CV",                 "PCV (%)",         _fmt(pcv, 2), "—",         "Total variation / mean"],
        ["Environmental CV",              "ECV (%)",         _fmt(ecv, 2), "—",         "Environmental variation / mean"],
    ]
    return _html_table(
        "Table 1", f"Variance Components for {trait}",
        ["Component", "Symbol", "Estimate", "% of σ²<sub>p</sub>", "Interpretation"],
        rows,
        footnotes=[
            "σ²<sub>g</sub> = genetic variance; σ²<sub>e</sub> = error variance; "
            "σ²<sub>gl</sub> = Genotype × Location interaction; σ²<sub>p</sub> = phenotypic variance.",
            "GCV/PCV/ECV = Genotypic/Phenotypic/Environmental Coefficient of Variation (%).",
        ],
        left_cols=[0, 1, 4],
    )


def _table_heritability_ga(envelope: Dict, trait: str) -> str:
    tables = envelope.get("tables", {})
    vc_recs = tables.get("variance_components", [])
    vc = vc_recs[0] if vc_recs else {}

    h2     = vc.get("H2_broad")
    h2_pct = vc.get("H2_broad_pct")
    if h2_pct is None and h2 is not None:
        try:
            h2_pct = float(h2) * 100 if float(h2) <= 1.0 else float(h2)
        except Exception:
            h2_pct = None
    ga     = vc.get("GA")
    ga_pct = vc.get("GA_percent")
    gm     = vc.get("grand_mean")

    h2_f   = float(h2_pct) if h2_pct is not None else 0.0
    ga_f   = float(ga_pct) if ga_pct is not None else 0.0

    h2_int = ("Very High ≥70% — direct selection highly effective" if h2_f >= 70 else
              "High 60–70% — direct selection effective" if h2_f >= 60 else
              "Moderate 30–60% — multi-environment testing advised" if h2_f >= 30 else
              "Low &lt;30% — consider MAS or recurrent selection")
    ga_int = ("Excellent &gt;10% — rapid progress possible" if ga_f > 10 else
              "Good 5–10% — moderate selection response" if ga_f >= 5 else
              "Low &lt;5% — limited response to direct selection")

    rows = [
        ["Broad-sense Heritability", "H²",   f"{_fmt(h2_f, 1)}%",
         "σ²<sub>g</sub> / σ²<sub>p</sub> × 100",  h2_int],
        ["Genetic Advance (5% sel.)", "GA",   _fmt(ga, 4),
         "k × √(σ²<sub>p</sub> × H²)",              "Predicted gain per selection cycle"],
        ["Genetic Advance % Mean",    "GA%",  f"{_fmt(ga_f, 1)}%",
         "(GA / μ) × 100",                           ga_int],
        ["Grand Mean",                "μ",    _fmt_large(gm, 2),
         "—",                                        "—"],
        ["Selection Intensity",       "k",    "2.06",
         "At 5% selection intensity",                "Standard tabulated constant (Johnson et al., 1955)"],
    ]
    return _html_table(
        "Table 2", f"Heritability and Genetic Advance for {trait}",
        ["Parameter", "Symbol", "Value", "Formula", "Interpretation"],
        rows,
        footnotes=[
            "H² = broad-sense heritability (additive + dominance + epistatic variance).",
            "GA = k × √σ²<sub>p</sub> × H²  (k = 2.06 at 5% selection intensity, Allard 1960).",
            "GA% = (GA / Grand Mean) × 100. Values &gt;10%: excellent; 5–10%: good; &lt;5%: low.",
        ],
        left_cols=[0, 1, 3, 4],
    )


def _table_combined_anova(envelope: Dict, trait: str) -> str:
    tables  = envelope.get("tables", {})
    meta    = envelope.get("meta", {})
    n_locs  = meta.get("n_locations", "?")
    anova_recs = tables.get("combined_anova", [])

    rows: List[List[str]] = []
    total_row: Optional[List[str]] = None
    ss_tot = df_tot = 0.0

    for rec in anova_recs:
        src  = rec.get("source", "?")
        df_  = rec.get("df")
        ss   = rec.get("SS") or rec.get("sum_sq")
        ms   = rec.get("MS") or rec.get("mean_sq")
        f    = rec.get("F") or rec.get("F_value")
        p    = rec.get("PR(>F)") or rec.get("p_value")
        disp = rec.get("PR(>F)_display", "")
        if not disp:
            if p is not None:
                try:
                    pf = float(p)
                    disp = "< 0.001" if pf < 0.001 else _fmt(p, 4)
                except Exception:
                    disp = str(p)

        if ms is None and ss is not None and df_ and float(str(df_)) > 0:
            try:
                ms = float(ss) / float(df_)
            except Exception:
                pass
        try:
            df_tot += float(df_)
            ss_tot += float(ss) if ss else 0
        except Exception:
            pass

        rows.append([
            src,
            str(int(float(df_))) if df_ is not None else "—",
            _fmt(ss, 4), _fmt(ms, 4),
            _fmt(f, 3) if f is not None else "—",
            disp or "—",
            _sig(p),
        ])

    if rows:
        total_row = ["Total", str(int(df_tot)), _fmt(ss_tot, 4), "—", "—", "—", ""]

    return _html_table(
        "Table 3",
        f"Combined ANOVA for {trait} Across {n_locs} Location(s)",
        ["Source of Variation", "df", "SS", "MS", "F-value", "p-value", "Sig."],
        rows,
        total_row=total_row,
        footnotes=[
            "G×L = Genotype × Location interaction.",
            "Significance codes: *** p &lt; 0.001; ** p &lt; 0.01; * p &lt; 0.05; ns = not significant.",
            "Type II ANOVA; block effects included.",
        ],
        left_cols=[0],
    )


def _table_genotype_means(envelope: Dict, trait: str, genotype_col: str) -> str:
    tables     = envelope.get("tables", {})
    means_recs = tables.get("genotype_means", [])

    rows: List[List[str]] = []
    n = len(means_recs)
    for rank, rec in enumerate(means_recs, 1):
        gid   = (rec.get(genotype_col) or rec.get("genotype") or
                 rec.get("Genotype", f"G{rank}"))
        mean  = rec.get("mean")
        letter = rec.get("letter", rec.get("tukey_letter", ""))

        if rank == 1:
            rec_text = "Best performer — recommend for release/promotion"
        elif rank <= max(2, n // 3):
            rec_text = "High-performing — include in advanced trials"
        elif rank <= max(3, 2 * n // 3):
            rec_text = "Intermediate performer — evaluate in specific environments"
        else:
            rec_text = "Low-performing — consider removal from pipeline"

        rows.append([
            str(rank), str(gid), _fmt_large(mean, 2), str(letter or ""), rec_text
        ])

    return _html_table(
        "Table 4", f"Genotype Means and Rankings for {trait}",
        ["Rank", "Genotype", "Mean", "Tukey Group", "Recommendation"],
        rows,
        footnotes=[
            "Genotypes ranked by mean (descending).",
            "Tukey groups: means sharing the same letter are not significantly different (α = 0.05).",
        ],
        left_cols=[1, 4],
    )


def _table_stability(envelope: Dict, trait: str) -> str:
    tables     = envelope.get("tables", {})
    stab_recs  = tables.get("stability", [])

    rows: List[List[str]] = []
    for rec in stab_recs:
        gid  = (rec.get("genotype") or rec.get("Genotype", "—"))
        mean = rec.get("grand_mean") or rec.get("mean")
        bi   = rec.get("bi") if isinstance(rec.get("bi"), (int, float)) else \
               (rec.get("bi", {}) or {}).get("value")
        s2di = rec.get("S2di") if isinstance(rec.get("S2di"), (int, float)) else \
               (rec.get("S2di", {}) or {}).get("value")
        asv  = rec.get("ASV") if isinstance(rec.get("ASV"), (int, float)) else \
               (rec.get("ASV", {}) or {}).get("value")
        cls  = rec.get("classification", "—")

        # bi interpretation
        bi_int = "—"
        if bi is not None:
            try:
                bif = float(bi)
                bi_int = ("Highly responsive" if bif > 1.1 else
                          "Average stable" if bif >= 0.9 else "Low responsive")
            except Exception:
                pass

        rows.append([
            str(gid),
            _fmt_large(mean, 2),
            _fmt(bi, 3),
            _fmt(s2di, 4),
            _fmt(asv, 4),
            str(cls).replace("_", " ").title(),
            bi_int,
        ])

    return _html_table(
        "Table 5",
        f"Stability Analysis for {trait} (Eberhart &amp; Russell, 1966)",
        ["Genotype", "Mean", "b<sub>i</sub>", "S²<sub>di</sub>",
         "ASV", "Classification", "b<sub>i</sub> Interpretation"],
        rows,
        footnotes=[
            "b<sub>i</sub> = regression coefficient; S²<sub>di</sub> = deviation variance.",
            "ASV = AMMI Stability Value (Purchase et al., 2000).",
            "Stable: b<sub>i</sub> ≈ 1 and S²<sub>di</sub> not significant.",
            "Classification: Stable Across Environments = ideal widely-adapted genotype.",
        ],
        left_cols=[0, 5, 6],
    )


def _table_correlations(envelope: Dict) -> Optional[str]:
    """Build phenotypic correlation matrix HTML table."""
    tables = envelope.get("tables", {})
    corr   = tables.get("correlations")
    if not corr:
        return None

    # Handle nested structure: {"matrix": {"t1": {"t2": {"r": …, "p_value": …}}}}
    matrix: Optional[Dict] = None
    if isinstance(corr, dict):
        matrix = corr.get("matrix") or corr.get("phenotypic_matrix")
        if matrix is None and corr:
            # corr might be the matrix itself if pipeline stored it directly
            first_val = next(iter(corr.values()), None)
            if isinstance(first_val, dict):
                matrix = corr
    if not matrix:
        return None

    traits = list(matrix.keys())
    if not traits:
        return None

    headers = ["Trait"] + traits
    rows: List[List[str]] = []
    for t1 in traits:
        row_cells = [t1]
        for t2 in traits:
            if t1 == t2:
                row_cells.append("1.000")
            else:
                cell = (matrix.get(t1) or {}).get(t2)
                if cell is None:
                    cell = (matrix.get(t2) or {}).get(t1)
                if isinstance(cell, dict):
                    r = cell.get("r")
                    p = cell.get("p_value")
                    row_cells.append(f"{_fmt(r, 3)}{_sig(p)}")
                elif cell is not None:
                    row_cells.append(_fmt(cell, 3))
                else:
                    row_cells.append("—")
        rows.append(row_cells)

    return _html_table(
        "Table 6", "Phenotypic Correlation Matrix",
        headers, rows,
        footnotes=[
            "Pearson phenotypic correlations. Significance: *** p &lt; 0.001; "
            "** p &lt; 0.01; * p &lt; 0.05; ns = not significant.",
            "Values on diagonal = 1 (self-correlation).",
        ],
        left_cols=[0],
    )


def _table_path_analysis(envelope: Dict, target_trait: str = "") -> Optional[str]:
    """Build path analysis coefficient HTML table."""
    tables = envelope.get("tables", {})
    path   = tables.get("path_analysis")
    if not path:
        return None

    # Try to find direct/indirect effect records
    records: List[Dict] = []
    if isinstance(path, list):
        records = path
    elif isinstance(path, dict):
        records = path.get("direct_effects") or path.get("coefficients") or []
        if not records and path:
            records = [{"trait": k, **v} if isinstance(v, dict) else {"trait": k, "value": v}
                       for k, v in path.items() if k not in ("target", "formula", "r_squared")]

    if not records:
        return None

    rows: List[List[str]] = []
    for rec in records:
        trait   = rec.get("trait") or rec.get("source") or "—"
        direct  = rec.get("direct") or rec.get("direct_effect") or rec.get("coefficient") or rec.get("value")
        r_total = rec.get("r") or rec.get("total_correlation")
        indirect = rec.get("indirect") or rec.get("indirect_total")
        rows.append([
            str(trait),
            _fmt(direct, 4),
            _fmt(indirect, 4),
            _fmt(r_total, 4),
        ])

    target_label = f" on {target_trait}" if target_trait else ""
    return _html_table(
        "Table 7", f"Path Analysis Coefficients{target_label}",
        ["Causal Trait", "Direct Effect", "Indirect Effects (Total)", "Total Correlation (r)"],
        rows,
        footnotes=[
            "Path coefficients computed by stepwise multiple regression.",
            "Total correlation = direct + indirect effects.",
        ],
        left_cols=[0],
    )


def _table_selection_index(envelope: Dict, trait: str) -> Optional[str]:
    """Build selection index HTML table."""
    tables = envelope.get("tables", {})
    si     = tables.get("selection_index")
    if not si:
        return None

    records: List[Dict] = []
    if isinstance(si, list):
        records = si
    elif isinstance(si, dict):
        records = si.get("index") or si.get("genotypes") or []
        if not records and si:
            records = [{"genotype": k, "index_value": v}
                       for k, v in si.items()
                       if k not in ("weights", "traits", "formula")]

    if not records:
        return None

    rows: List[List[str]] = []
    for rank, rec in enumerate(
        sorted(records, key=lambda r: float(r.get("index_value", r.get("value", 0)) or 0), reverse=True),
        1,
    ):
        gid = rec.get("genotype") or rec.get("Genotype", f"G{rank}")
        idx = rec.get("index_value") or rec.get("value") or rec.get("index")
        rows.append([str(rank), str(gid), _fmt(idx, 4)])

    return _html_table(
        "Table 8", f"Selection Index Rankings for {trait}",
        ["Rank", "Genotype", "Selection Index Value"],
        rows,
        footnotes=[
            "Selection index (Smith-Hazel): I = Σ(b<sub>i</sub> × P<sub>i</sub>), "
            "where b<sub>i</sub> are economic weights and P<sub>i</sub> are phenotypic values.",
            "Higher index value → higher selection priority.",
        ],
        left_cols=[1],
    )


# ── Table 9: PCA Loadings ─────────────────────────────────────────────────────

def _table_pca_loadings(envelope: Dict) -> Optional[str]:
    """PCA loadings matrix — traits × principal components."""
    mv = envelope.get("tables", {}).get("multivariate")
    if not mv or not isinstance(mv, dict):
        return None
    pca = mv.get("pca")
    if not pca or not isinstance(pca, dict):
        return None
    loadings = pca.get("loadings", {})   # {PC1: {trait: val}, PC2: {...}, ...}
    if not loadings:
        return None

    pcs = sorted(loadings.keys())[:4]    # at most 4 PCs
    if not pcs:
        return None
    trait_names = list(loadings[pcs[0]].keys())
    if not trait_names:
        return None

    # Resolved explained-variance percent per PC
    ev_list = pca.get("explained_variance_ratio", [])
    ev_pct: Dict[str, float] = {}
    for item in ev_list:
        if isinstance(item, dict):
            ev_pct[str(item.get("PC", ""))] = float(item.get("percent", 0))

    # Build column headers with explained variance
    headers = ["Trait"] + [
        f"{pc} ({ev_pct.get(pc, 0):.1f}%)" if pc in ev_pct else pc
        for pc in pcs
    ]

    # Per-PC find highest-absolute-loading trait for bolding
    max_trait: Dict[str, str] = {}
    for pc in pcs:
        col = loadings.get(pc, {})
        if col:
            max_trait[pc] = max(col, key=lambda t: abs(col.get(t, 0)))

    rows: List[List[str]] = []
    for tname in trait_names:
        row = [tname]
        for pc in pcs:
            val = loadings.get(pc, {}).get(tname)
            cell = _fmt(val, 4) if val is not None else "—"
            if max_trait.get(pc) == tname:
                cell = f"<strong>{cell}</strong>"
            row.append(cell)
        rows.append(row)

    return _html_table(
        "Table 9", "PCA Loadings Matrix (Traits × Principal Components)",
        headers, rows,
        footnotes=[
            "Values are eigenvector loadings (range −1 to +1).",
            "<strong>Bold</strong> = highest absolute loading per PC (strongest contributor).",
            "Positive loading → trait increases along that PC axis; negative → decreases.",
        ],
        left_cols=[0],
    )


# ── Table 10: Cluster Membership ──────────────────────────────────────────────

def _table_cluster_membership(envelope: Dict) -> Optional[str]:
    """Cluster membership summary from hierarchical or k-means clustering."""
    mv = envelope.get("tables", {}).get("multivariate")
    if not mv or not isinstance(mv, dict):
        return None

    # Prefer hierarchical; fall back to k-means
    hc = mv.get("hierarchical_clustering")
    km = mv.get("kmeans_clustering")
    source = hc if (hc and isinstance(hc, dict)) else (km if isinstance(km, dict) else None)
    if source is None:
        return None

    cluster_members: Dict = source.get("cluster_members", {})
    if not cluster_members:
        return None

    method = source.get("method", "Cluster analysis")

    rows: List[List[str]] = []
    for cluster_name in sorted(cluster_members.keys()):
        members = cluster_members[cluster_name]
        rows.append([
            cluster_name.replace("_", " "),
            str(len(members)),
            ", ".join(sorted(str(m) for m in members)),
        ])

    return _html_table(
        "Table 10", "Cluster Membership Summary",
        ["Cluster", "n", "Genotypes"],
        rows,
        footnotes=[
            f"Clustering method: {method}.",
            "Genotypes within a cluster share similar multi-trait performance profiles.",
        ],
        left_cols=[0, 2],
    )


# ── Public API ───────────────────────────────────────────────────────────────

def build_html_tables(
    envelope: Dict[str, Any],
    trait: str,
    genotype_col: str = "Genotype",
    target_trait: str = "",
) -> List[Dict[str, str]]:
    """
    Build all publication-quality HTML tables for a genetics trial.

    Returns:
        List of {"name": str, "html": str} dicts, in table order.
        Tables with no data are silently omitted.
    """
    result: List[Dict[str, str]] = []

    def _add(name: str, html: Optional[str]) -> None:
        if html:
            result.append({"name": name, "html": html})

    _add("Variance Components",          _table_variance_components(envelope, trait))
    _add("Heritability & Genetic Advance", _table_heritability_ga(envelope, trait))
    _add("Combined ANOVA",               _table_combined_anova(envelope, trait))
    _add("Genotype Means & Rankings",    _table_genotype_means(envelope, trait, genotype_col))
    _add("Stability Analysis",           _table_stability(envelope, trait))
    _add("Phenotypic Correlations",      _table_correlations(envelope))
    _add("Path Analysis",                _table_path_analysis(envelope, target_trait))
    _add("Selection Index",              _table_selection_index(envelope, trait))
    _add("PCA Loadings Matrix",          _table_pca_loadings(envelope))
    _add("Cluster Membership",           _table_cluster_membership(envelope))

    return result
