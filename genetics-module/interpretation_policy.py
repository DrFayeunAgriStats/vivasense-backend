"""VivaSense Scientific Reporting Framework (SRF) — interpretation policy.

Phase 1 (CRD / RCBD). Pure functions, structured so they migrate cleanly into
the four SRF layers later:

  - TRAIT_DIRECTIONALITY / classify_directionality  -> Layer 2b statistical policy
  - classify_assumption_severity                    -> Layer 2b
  - compute_reliability                             -> Layer 2 composite object
  - EVIDENCE_LEVELS / evidence_level_line           -> Layer 2 -> Layer 3 footer
  - directional_phrase / rewrite_significance_claim -> Layer 3 narrative
  - enforce_institutional_language                  -> Layer 2a institutional rules

Contract: consumes Layer-1 facts (numbers / booleans) only. Never recomputes a
statistic and never renders to a medium. Every function is deterministic and
side-effect free so the Verification Panel can test policy in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ============================================================================
# Layer 2b — trait directionality
# maximize | minimize | context_dependent | descriptive_only
# ============================================================================

# Seed vocabulary for Nigerian cowpea / maize / rice / vegetable breeding,
# proposed by the biometrics lead (Dr Fayeun) 2026-07-18. Keys are normalised
# (lower-case, spaces/hyphens -> underscores). Community-contributable, subject
# to biometrics-lead review (§6.4).
#
# OPEN QUESTIONS still under co-author (Dr Adunola) review — kept at the
# conservative value below until resolved:
#   * 100/1000-seed weight: set 'maximize' (larger seed generally favoured), but
#     some markets favour smaller seed — may become context_dependent per crop.
#   * plant_height: set 'context_dependent' (cowpea/sorghum often prefer shorter
#     for lodging; maize often taller) — a crop-conditional resolution is future work.
TRAIT_DIRECTIONALITY: Dict[str, str] = {
    # ── maximize: yield & yield components ─────────────────────────────────
    "yield": "maximize",
    "yield_kg_ha": "maximize",
    "grain_yield": "maximize",
    "grain_yield_kg_ha": "maximize",
    "grain_yield_t_ha": "maximize",
    "pod_yield_kg_ha": "maximize",
    "seed_yield_kg_ha": "maximize",
    "biomass_kg_ha": "maximize",
    "dry_matter_kg_ha": "maximize",
    "fodder_yield": "maximize",
    "pods_per_plant": "maximize",
    "seeds_per_pod": "maximize",
    "panicles_per_plant": "maximize",
    "tillers_per_plant": "maximize",
    "harvest_index": "maximize",
    # seed weight — see OPEN QUESTIONS above
    "100_seed_weight": "maximize",
    "1000_grain_weight": "maximize",
    "test_weight": "maximize",
    # quality
    "protein_content_pct": "maximize",
    "oil_content_pct": "maximize",
    # ── minimize: stresses, losses, storability ────────────────────────────
    "disease_severity_score": "minimize",
    "rust_score": "minimize",
    "blast_score": "minimize",
    "striga_count": "minimize",
    "aphid_damage": "minimize",
    "bruchid_damage": "minimize",
    "lodging_score": "minimize",
    "shattering_score": "minimize",
    "moisture_content_pct": "minimize",
    # ── context_dependent: phenology, stature, physiology ──────────────────
    "days_to_flowering": "context_dependent",
    "days_to_50pct_flowering": "context_dependent",
    "days_to_maturity": "context_dependent",
    "days_to_pod_fill": "context_dependent",
    "plant_height_cm": "context_dependent",   # see OPEN QUESTIONS above
    "leaf_area_index": "context_dependent",
    "chlorophyll_content_spad": "context_dependent",
    "canopy_temperature": "context_dependent",
    # ── descriptive_only: categorical / aesthetic ─────────────────────────
    "seed_colour_score": "descriptive_only",
    "flower_colour": "descriptive_only",
    "growth_habit_score": "descriptive_only",
    "pod_colour": "descriptive_only",
}

# Keyword heuristics, applied in priority order when a trait is not in the table.
_MINIMIZE_KW = (
    "disease", "severity", "incidence", "infection", "lodging", "pest", "damage",
    "sterility", "shatter", "blight", "rust", "smut", "rot", "aphid", "borer",
    "wilt", "necrosis", "defect", "striga", "bruchid", "weevil", "infestation",
)
_CONTEXT_KW = (
    "days_to", "day_to", "flowering", "maturity", "duration", "time_to",
    "earliness", "height", "cycle", "canopy_temperature",
)
# NB: 'score'/'rating' deliberately excluded — too ambiguous (a disease score is
# minimize, a vigour score maximize). Unlisted '*_score' traits fall through to
# descriptive_only rather than being mislabelled maximize.
_MAXIMIZE_KW = (
    "yield", "weight", "biomass", "vigour", "vigor", "germination", "emergence",
    "tiller", "pod", "grain", "seed", "protein", "oil", "content", "count",
    "number", "size", "length", "width", "branch", "leaf_area", "chlorophyll",
    "survival", "harvest_index",
)

DIRECTIONS = ("maximize", "minimize", "context_dependent", "descriptive_only")


def _norm(trait_name: str) -> str:
    return str(trait_name or "").strip().lower().replace(" ", "_").replace("-", "_")


def classify_directionality(trait_name: str) -> str:
    """Best-effort trait objective. Explicit table wins; else keyword heuristics.

    Order matters: 'minimize' cues (disease, lodging) are checked before the
    generic 'maximize' cues so 'disease_score' is not misread as maximize.
    """
    t = _norm(trait_name)
    if t in TRAIT_DIRECTIONALITY:
        return TRAIT_DIRECTIONALITY[t]
    for kw in _MINIMIZE_KW:
        if kw in t:
            return "minimize"
    for kw in _CONTEXT_KW:
        if kw in t:
            return "context_dependent"
    for kw in _MAXIMIZE_KW:
        if kw in t:
            return "maximize"
    return "descriptive_only"


def directional_phrase(direction: str) -> Dict[str, str]:
    """Wording for the leading treatment mean, governed by directionality.

    Returns {top, adj, caveat}. `top` fills "the treatment with the {top} mean".
    Institutional rule §5.2: never "best/worst" for context_dependent/descriptive.
    """
    if direction == "maximize":
        return {"top": "highest", "adj": "highest-performing", "caveat": ""}
    if direction == "minimize":
        return {"top": "lowest", "adj": "lowest (most favourable)", "caveat": ""}
    if direction == "context_dependent":
        return {"top": "largest", "adj": "largest",
                "caveat": " — whether a larger or smaller value is preferable depends on the breeding objective"}
    return {"top": "largest", "adj": "largest",
            "caveat": " — reported descriptively; no directional preference is implied"}


# ============================================================================
# Layer 2b — assumption severity
# ============================================================================

def classify_assumption_severity(p_value: Optional[float],
                                 passed: Optional[bool] = None,
                                 alpha: float = 0.05,
                                 n: Optional[int] = None,
                                 w_statistic: Optional[float] = None) -> str:
    """none | mild | moderate | strong from a test p-value (CRD directive v2 bands).

      p >= alpha                         -> none
      p < 0.001                          -> strong
      0.001 <= p < 0.01                  -> moderate
      0.01  <= p < alpha (0.05):
          mild   if n >= 50 and W >= 0.95   (over-powered test, trivial departure)
          else moderate                     (default: treat a marginal violation
                                             cautiously when the sample is small or
                                             the Shapiro W is not high)

    n and W refine only the marginal [0.01, alpha) band and only apply to
    normality (Shapiro W). Homogeneity (Levene, no W) falls to the 'else moderate'
    branch in that band, which is the intended conservative behaviour.
    """
    if p_value is None:
        return "none"
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return "none"
    if p >= alpha:
        return "none"
    if p < 0.001:
        return "strong"
    if p < 0.01:
        return "moderate"
    # 0.01 <= p < alpha
    if n is not None and w_statistic is not None and n >= 50 and w_statistic >= 0.95:
        return "mild"
    return "moderate"


_SEVERITY_ORDER = {"none": 0, "mild": 1, "moderate": 2, "strong": 3}


def _worst(*sev: str) -> str:
    return max(sev, key=lambda s: _SEVERITY_ORDER.get(s, 0)) if sev else "none"


# ============================================================================
# Layer 2 — Reliability composite (§4.3)
# ============================================================================

def compute_reliability(
    *,
    normality_p: Optional[float] = None,
    homogeneity_p: Optional[float] = None,
    n_influential: Optional[int] = None,
    is_balanced: Optional[bool] = None,
    alpha: float = 0.05,
    n_observations: Optional[int] = None,
    shapiro_w: Optional[float] = None,
) -> Dict[str, Any]:
    """Composite reliability from Layer-1 facts -> {level, reasons}.

    level: High | Moderate | Low. reasons: per-check {status: pass|warn|fail, text}.
    Mapping: any 'strong' failure -> Low; any 'moderate'/'mild' failure or
    influence/imbalance warning -> Moderate; otherwise High.

    n_observations + shapiro_w refine the marginal-normality band (see
    classify_assumption_severity). They apply to normality only.
    """
    reasons: List[Dict[str, str]] = []
    norm_sev = classify_assumption_severity(normality_p, alpha=alpha,
                                            n=n_observations, w_statistic=shapiro_w)
    homo_sev = classify_assumption_severity(homogeneity_p, alpha=alpha)

    if normality_p is not None:
        reasons.append({
            "status": "pass" if norm_sev == "none" else ("fail" if norm_sev == "strong" else "warn"),
            "text": "Normality supported" if norm_sev == "none"
            else f"Normality not supported ({norm_sev})",
        })
    if homogeneity_p is not None:
        reasons.append({
            "status": "pass" if homo_sev == "none" else ("fail" if homo_sev == "strong" else "warn"),
            "text": "Homogeneity of variance supported" if homo_sev == "none"
            else f"Homogeneity of variance not supported ({homo_sev})",
        })
    if n_influential is not None:
        reasons.append({
            "status": "pass" if n_influential == 0 else "warn",
            "text": "No influential observations flagged" if n_influential == 0
            else f"{n_influential} influential observation{'' if n_influential == 1 else 's'} flagged",
        })
    if is_balanced is not None:
        reasons.append({
            "status": "pass" if is_balanced else "warn",
            "text": "Balanced design" if is_balanced else "Unbalanced design",
        })

    worst_assumption = _worst(norm_sev, homo_sev)
    has_warn = any(r["status"] == "warn" for r in reasons)
    has_fail = any(r["status"] == "fail" for r in reasons)

    if has_fail or worst_assumption == "strong":
        level = "Low"
    elif has_warn or worst_assumption in ("mild", "moderate"):
        level = "Moderate"
    else:
        level = "High"

    return {"level": level, "reasons": reasons}


# ============================================================================
# Layer 2/3 — Evidence Level (§4.4)
# ============================================================================

EVIDENCE_LEVELS: Dict[str, str] = {
    "A": "Analysis computed from the user-supplied raw dataset.",
    "B": "Backend output re-verified against raw data.",
    "C": "Report generated; not independently verified.",
    "D": "Screenshot or export inspected; underlying computation not verified.",
    "E": "Narrative summary; no numerical verification.",
}

# VivaSense-generated analyses run on user data are always Level A (§4.4).
DEFAULT_EVIDENCE_LEVEL = "A"


def evidence_level_line(level: str = DEFAULT_EVIDENCE_LEVEL) -> str:
    lvl = level if level in EVIDENCE_LEVELS else DEFAULT_EVIDENCE_LEVEL
    return f"Evidence Level {lvl} — {EVIDENCE_LEVELS[lvl]}"


# ============================================================================
# Layer 3 — narrative: significance claim rewriting (§3.3) + institutional rules
# ============================================================================

# Adjectives that overstate effect size / generalisability (§5.8). Downgraded
# unless a Layer-2 magnitude check has passed (not available in Phase 1, so we
# downgrade conservatively).
_OVERSTATED = {
    "substantial": "notable",
    "substantially": "notably",
    "dramatic": "marked",
    "dramatically": "markedly",
    "clear-cut": "apparent",
    "clearcut": "apparent",
    "huge": "large",
    "enormous": "large",
    "vast": "large",
    "definitive": "supported",
    "prove": "support",
    "proves": "supports",
    "proven": "supported",
}


def downgrade_overstated_adjectives(text: str) -> str:
    """§5.8 — soften overstated magnitude/generalisability language."""
    if not text:
        return text
    import re
    out = text
    for strong, softer in _OVERSTATED.items():
        out = re.sub(rf"\b{re.escape(strong)}\b", softer, out, flags=re.IGNORECASE)
    return out


def rewrite_significance_claim(
    base_claim: str,
    *,
    normality_p: Optional[float] = None,
    homogeneity_p: Optional[float] = None,
    transformed_applied: bool = False,
    alpha: float = 0.05,
    n_observations: Optional[int] = None,
    shapiro_w: Optional[float] = None,
) -> str:
    """§3.3 — integrate assumption caveats INTO the significance sentence rather
    than prepending a warning block. Returns the (possibly rewritten) sentence.

    A passing model returns the base claim unchanged. A violated model appends a
    'however, ...' clause naming the specific violated assumption(s) and the
    interpret-with-caution guidance, and notes when a transformation restored fit.
    """
    norm_sev = classify_assumption_severity(normality_p, alpha=alpha,
                                            n=n_observations, w_statistic=shapiro_w)
    homo_sev = classify_assumption_severity(homogeneity_p, alpha=alpha)
    violated = [name for name, sev in
                (("the normality assumption", norm_sev), ("the homogeneity-of-variance assumption", homo_sev))
                if sev in ("mild", "moderate", "strong")]

    claim = base_claim.rstrip()
    if not violated:
        return claim

    if claim.endswith("."):
        claim = claim[:-1]
    joined = violated[0] if len(violated) == 1 else " and ".join(violated)
    caveat = (
        f"; however, residual diagnostics indicate that {joined} "
        f"{'was' if len(violated) == 1 else 'were'} not supported, so this result "
        f"should be interpreted with caution"
    )
    if transformed_applied:
        caveat += (
            ", and a variance-stabilising transformation was applied whose "
            "diagnostics are reported alongside the untransformed model"
        )
    return f"{claim}{caveat}."


def enforce_institutional_language(text: str, direction: str) -> str:
    """§5.2 + §5.8 — never 'best/worst' for context_dependent/descriptive traits,
    and downgrade overstated adjectives. A conservative last-pass guard."""
    if not text:
        return text
    out = downgrade_overstated_adjectives(text)
    if direction in ("context_dependent", "descriptive_only"):
        import re
        out = re.sub(r"\bbest[- ]performing\b", "leading", out, flags=re.IGNORECASE)
        out = re.sub(r"\bthe best\b", "the largest", out, flags=re.IGNORECASE)
        out = re.sub(r"\bthe worst\b", "the smallest", out, flags=re.IGNORECASE)
    return out
