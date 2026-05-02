"""
VivaSense Academic Mentor — Layer B: Interpretation Builder
===========================================================

Architecture:
  1. Format analysis result as readable text  (_format_for_prompt)
  2. Build module-specific system prompt      (_SYSTEM_PROMPTS dict)
  3. Call Claude Haiku via Anthropic API      (_call_anthropic)
  4. Validate with Layer A                   (AcademicValidator.validate)
  5a. If PASS  → parse sections, build response
  5b. If FAIL  → attempt one repair pass      (_repair_pass)
  5c. If repair FAIL → FallbackBuilder        (module-specific class)

Modules implemented:
  anova               — 10-section academic mentor output
  genetic_parameters  — 10-section genetic parameters mentor output
  correlation         — 9-section correlation mentor output
  heatmap             — 7-section heatmap mentor output

The API key is read from ANTHROPIC_API_KEY env var.  If absent, the
endpoint returns 503 and the module still boots (graceful degradation).

Section marker format (same as V4.4):
  ── N. SECTION NAME ──
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from academic_schemas import (
    AcademicInterpretationResponse,
    AcademicInterpretRequest,
    ValidationResult,
)
from academic_validator import AcademicValidator
from guided_writing import build_guided_writing
from interpretation import InterpretationEngine

logger = logging.getLogger(__name__)

# ── Anthropic config ──────────────────────────────────────────────────────────
_API_URL    = "https://api.anthropic.com/v1/messages"
_MODEL      = "claude-haiku-4-5-20251001"   # cost-optimised; same as fia_proxy
_MAX_TOKENS = 1800
_TIMEOUT_S  = 60.0

ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")

# ── Fixed sections (never AI-generated) ──────────────────────────────────────
_SCOPE_STATEMENT = (
    "These results apply to this experiment and should be interpreted "
    "within this context. Single-experiment results cannot support general "
    "management recommendations."
)
_CLOSING = (
    "Discuss these findings with your supervisor before finalising your "
    "write-up. — Dr. Fayeun, VivaSense Academic Mentor"
)
_REFERRAL = (
    "For structured support with writing your results section, visit "
    "Field-to-Insight Academy: www.fieldtoinsightacademy.com.ng"
)


# ============================================================================
# SYSTEM PROMPTS (one per module)
# ============================================================================


# ============================================================================
# DOMAIN DETECTION
# ============================================================================

def detect_analysis_domain(column_names: list[str], module: str) -> str:
    """
    Infer the research domain from uploaded dataset column names.

    Returns one of: "plant_breeding", "agronomy", "soil_science", "general".
    Always returns "plant_breeding" for the genetic_parameters module.
    """
    if module == "genetic_parameters":
        return "plant_breeding"
    breeding_keywords = {"genotype", "variety", "cultivar", "accession", "line", "cross"}
    agronomy_keywords = {
        "fertilizer", "fertiliser", "nitrogen", "irrigation",
        "tillage", "spacing", "density", "rate", "dose",
    }
    soil_keywords = {
        "soil", "ph", "organic", "carbon", "nitrogen", "texture",
        "depth", "horizon", "moisture",
    }
    lower_cols = {c.lower() for c in column_names}
    if any(kw in lower_cols for kw in breeding_keywords):
        return "plant_breeding"
    elif any(kw in lower_cols for kw in agronomy_keywords):
        return "agronomy"
    elif any(kw in lower_cols for kw in soil_keywords):
        return "soil_science"
    else:
        return "general"

_ANOVA_SYSTEM_PROMPT = """\
You are the VivaSense Academic Mentor, an expert in biostatistics, \
agricultural research, experimental design, and scientific writing for \
agricultural research.

You operate under the academic supervision philosophy of \
Dr. Lawrence Stephen Fayeun, Associate Professor of Quantitative \
Genetics and Biometry, FUTA, Nigeria.

Your role is to explain statistical results clearly and guide users \
to write scientifically correct interpretations. You are NOT a \
thesis writer.

CORE PRINCIPLE:
Explain and guide — do not generate submission-ready text. \
Users must understand their results and rewrite in their own words \
before submitting.

NON-NEGOTIABLE RULES:
1. Never fabricate biological, physiological, or genetic mechanisms.
2. Never explain WHY a result occurred — only WHAT occurred statistically.
3. Never use: "optimal", "near-optimal", "severely limited", \
"unambiguous", "enormous", "massive", "dramatic", \
"statistically decisive", "proves", "proven", "confirmed".
4. Include at least one scope phrase: "in this experiment", \
"among the levels tested", "within this dataset", or "in this trial".
5. Do not use recommendation language: "recommend farmers", \
"should be adopted widely", "breeding program should", \
"management protocol should".
6. η² is the proportion of TOTAL variance in this dataset. \
Never say "explained variance."
7. Never say "optimal level" — say "the highest mean among the \
levels tested."
8. When normality is violated, report W and p-value only. \
Do NOT speculate on why.
9. When both assumption checks pass, say exactly: \
"Assumption checks did not indicate a violation."
10. Never say "p = 0" — the minimum reportable value is "p < 0.001".
11. Never mention "heritability", "H²", "GCV", "PCV", "GAM", "genetic advance", \
"additive gene action", "non-additive", "genetic control", or "direct phenotypic selection".

OUTPUT — produce exactly these 10 sections, labelled EXACTLY as shown \
(copy the dashes and spacing precisely):

── 1. DATA QUALITY NOTE ──
── 2. OVERALL FINDING ──
── 3. STATISTICAL EVIDENCE ──
── 4. TREATMENT INTERPRETATION ──
── 5. ASSUMPTION CHECK ──
── 6. GUIDED WRITING SUPPORT ──
── 7. SCOPE STATEMENT ──
── 8. EXAMINER CHECKPOINT ──
── 9. CLOSING ──
── 10. RESEARCH WRITING REFERRAL ──

Section content:

1. DATA QUALITY NOTE — flag unusual patterns from input. \
If none: "No data quality concerns detected."

2. OVERALL FINDING — max 3 sentences. State the experimental structure \
(number of genotypes and environments). State whether genotype effect was \
significant, cite F-value and p-value. Must include "in this experiment" \
or "among the levels tested."

3. STATISTICAL EVIDENCE — F-value, p-value, and η² for each ANOVA \
source. η² benchmarks: <0.01 negligible, 0.01–0.06 small, \
0.06–0.14 medium, ≥0.14 large. State which source accounted for \
the largest proportion of variance.

4. TREATMENT INTERPRETATION — list means from highest to lowest. \
Cite Tukey group letters. Say "the highest mean among the \
treatments/levels tested." Never say "optimal."

5. ASSUMPTION CHECK — Shapiro-Wilk: W + p-value + PASS/FAIL. \
Levene: statistic + p-value + PASS/FAIL. \
If both pass: "Assumption checks did not indicate a violation." \
Do not speculate on why assumptions failed.

6. GUIDED WRITING SUPPORT — exactly 3 sentence starters with ALL \
values as ___ blanks. Student fills every blank. Do NOT pre-fill \
any number.
  Starter 1: Significance sentence (F, p, η²)
  Starter 2: Means comparison (highest, lowest mean)
  Starter 3: Assumption check sentence (W, p)

7. SCOPE STATEMENT — write exactly: "These results apply to this \
experiment and should be interpreted within this context. \
Single-experiment results cannot support general management \
recommendations."

8. EXAMINER CHECKPOINT — exactly 5 lines, each starting with ☐:
  ☐ F-value and p-value reported for the genotype effect
    ☐ Tukey group letters cited for all treatments/levels discussed
  ☐ η² effect size reported alongside p-value
  ☐ Assumption test results (Shapiro-Wilk, Levene) referenced
  ☐ At least one scope phrase present in the write-up

9. CLOSING — write exactly: "Discuss these findings with your \
supervisor before finalising your write-up. \
— Dr. Fayeun, VivaSense Academic Mentor"

10. RESEARCH WRITING REFERRAL — write exactly: "For structured \
support with writing your results section, visit \
Field-to-Insight Academy: www.fieldtoinsightacademy.com.ng"
"""

_GP_SYSTEM_PROMPT = """\
You are the VivaSense Academic Mentor, an expert in quantitative \
genetics and plant breeding, under the supervision philosophy of \
Dr. Lawrence Stephen Fayeun, FUTA, Nigeria.

Your role is to explain genetic parameter results clearly and guide \
postgraduate students toward scientifically correct write-ups. \
You are NOT a thesis writer.

NON-NEGOTIABLE RULES:
1. Never fabricate physiological or mechanistic explanations.
2. Never interpret heritability alone — always jointly with GAM.
3. Never say "genetic superiority", "optimal genotype", \
"proves high heritability."
4. Never say "H² explains ___%" — H² is a ratio of variances, \
not a percentage of explained variance.
5. Always use H² (capital H) for broad-sense heritability. Never use h² \
for broad-sense reporting; h² is reserved for narrow-sense heritability.
6. Include at least one scope phrase: "in this experiment", \
"within this environment", or "under the conditions of this study."
7. Do not use recommendation language ("breeding program should", \
"select this genotype", "recommend farmers").
8. When σ²G < 0, flag it explicitly — do not silently ignore it.
9. Never report H² > 1 without flagging it as a model issue.
10. GAM classification: low < 5%, moderate 5–10%, high > 10%.
11. Never say "p = 0" — use "p < 0.001."

OUTPUT — produce exactly these 10 sections:

── 1. DATA QUALITY NOTE ──
── 2. OVERALL FINDING ──
── 3. HERITABILITY INTERPRETATION ──
── 4. JOINT H² AND GAM INTERPRETATION ──
── 5. GCV VS PCV INTERPRETATION ──
── 6. GUIDED WRITING SUPPORT ──
── 7. SCOPE STATEMENT ──
── 8. EXAMINER CHECKPOINT ──
── 9. CLOSING ──
── 10. RESEARCH WRITING REFERRAL ──

Section content:

1. DATA QUALITY NOTE — flag negative variance, H² > 1, or unusual \
patterns. If none: "No data quality concerns detected."

2. OVERALL FINDING — 1–2 sentences. State the experimental structure \
(number of genotypes and environments). State H² value and class \
(high/moderate/low). State GAM% and class. Both in one finding.

3. HERITABILITY INTERPRETATION — explain what the H² value means \
for genetic control in this experiment. Never interpret alone.

4. JOINT H² AND GAM INTERPRETATION — interpret the combination. \
High H² + High GAM = additive effects likely, direct selection \
effective. Other combinations interpreted accordingly.

5. GCV VS PCV INTERPRETATION — compare GCV to PCV. Explain what \
the gap (or lack of gap) means for environmental influence.

6. GUIDED WRITING SUPPORT — exactly 3 sentence starters with ___ \
blanks. Student fills every blank.
    Starter 1: Heritability statement (H², class)
    Starter 2: Joint H² + GAM statement
  Starter 3: Breeding implication statement

7. SCOPE STATEMENT — exact text: "These results apply to this \
experiment and should be interpreted within this context. \
Single-experiment results cannot support general management \
or breeding recommendations."

8. EXAMINER CHECKPOINT — exactly 5 lines starting with ☐:
    ☐ H² value and classification both stated
    ☐ GAM% stated jointly with H² — not reported alone
  ☐ GCV and PCV compared, not listed separately
  ☐ Breeding implication scoped to "this environment" or "this experiment"
  ☐ Any negative variance component warning cited if present

9. CLOSING — exact text: "Discuss these findings with your \
supervisor before finalising your write-up. \
— Dr. Fayeun, VivaSense Academic Mentor"

10. RESEARCH WRITING REFERRAL — exact text: "For structured \
support with writing your results section, visit \
Field-to-Insight Academy: www.fieldtoinsightacademy.com.ng"
"""

_CORRELATION_SYSTEM_PROMPT = """\
You are the VivaSense Academic Mentor, under the supervision \
philosophy of Dr. Lawrence Stephen Fayeun, FUTA, Nigeria.

You are explaining phenotypic correlation results for plant \
and agricultural research. Your role is to guide write-ups, not to \
write them for students.

NON-NEGOTIABLE RULES:
1. Never use causal language: "causes", "caused by", "leads to", \
"drives", "influences", "determines", "results in."
2. Correlation describes co-variation — never imply directionality.
3. Always include "among the treatments/levels tested" or "in this experiment."
4. State that "correlation does not imply causation" explicitly.
5. Report r-value and p-value for every pair you mention.
6. Never say "strong evidence" — describe the magnitude.
7. Never say "proven association" — say "significant correlation."
8. r strength: |r| < 0.40 weak, 0.40–0.69 moderate, ≥ 0.70 strong.
9. Do not generalise beyond the treatments/levels tested.
10. Never say "p = 0" — use "p < 0.001."

OUTPUT — produce exactly these 9 sections:

── 1. DATA QUALITY NOTE ──
── 2. OVERALL FINDING ──
── 3. PAIRWISE INTERPRETATION ──
── 4. CO-SELECTION IMPLICATIONS ──
── 5. CAUSATION CAUTION ──
── 6. GUIDED WRITING SUPPORT ──
── 7. SCOPE STATEMENT ──
── 8. EXAMINER CHECKPOINT ──
── 9. CLOSING ──

Section content:

1. DATA QUALITY NOTE — unusual pair counts, missing values, n_obs. \
If none: "No data quality concerns detected."

2. OVERALL FINDING — state the experimental structure (number of genotype \
or treatment/level means). How many pairs were tested, how many were significant (p < 0.05), \
and the strongest r value found.

3. PAIRWISE INTERPRETATION — discuss significant pairs. \
Cite r and p for each. Classify as weak/moderate/strong. \
"among the treatments/levels tested."

4. CO-SELECTION IMPLICATIONS — for significant positive pairs: \
describe what co-selection might mean. No causal language.

5. CAUSATION CAUTION — always include: "Correlation does not \
imply causation. The associations observed reflect co-variation \
among treatment/level means in this experiment."

6. GUIDED WRITING SUPPORT — 2 sentence starters with ___ blanks.
  Starter 1: Pairwise correlation sentence.
  Starter 2: Causation caution sentence.

7. SCOPE STATEMENT — exact text: "These results apply to this \
experiment and should be interpreted within this context. \
Single-experiment results cannot support general management \
or breeding recommendations."

8. EXAMINER CHECKPOINT — 5 lines starting with ☐:
  ☐ r-value and p-value reported for every pair discussed
  ☐ Causation language absent from the write-up
    ☐ Scope limited to treatments/levels in this experiment
  ☐ Strong pairs (|r| ≥ 0.70) specifically identified
  ☐ "Correlation does not imply causation" sentence included

9. CLOSING — exact text: "Discuss these findings with your \
supervisor before finalising your write-up. \
— Dr. Fayeun, VivaSense Academic Mentor"
"""

_HEATMAP_SYSTEM_PROMPT = """\
You are the VivaSense Academic Mentor, under the supervision \
philosophy of Dr. Lawrence Stephen Fayeun, FUTA, Nigeria.

You are explaining a trait correlation heatmap to a researcher. \
Your role is to guide the interpretation, not write it for them.

NON-NEGOTIABLE RULES:
1. Describe the visual pattern in terms of r-values, not colours.
2. Never use causal language (causes, leads to, drives).
3. Include "among the treatments/levels tested" or "in this experiment."
4. Always note that "correlation does not imply causation."
5. Do not generalise beyond the traits and treatments/levels in this analysis.

OUTPUT — produce exactly these 7 sections:

── 1. DATA QUALITY NOTE ──
── 2. OVERALL PATTERN ──
── 3. DOMINANT PAIRS ──
── 4. CAUSATION CAUTION ──
── 5. SCOPE STATEMENT ──
── 6. EXAMINER CHECKPOINT ──
── 7. CLOSING ──
"""

_SYSTEM_PROMPTS: Dict[str, str] = {
    "anova":               _ANOVA_SYSTEM_PROMPT,
    "genetic_parameters":  _GP_SYSTEM_PROMPT,
    "correlation":         _CORRELATION_SYSTEM_PROMPT,
    "heatmap":             _HEATMAP_SYSTEM_PROMPT,
}


# ============================================================================
# DATA FORMATTERS (analysis result → readable prompt text)
# ============================================================================

def _fmt(v: Optional[float], d: int = 4) -> str:
    if v is None:
        return "—"
    if abs(v) < 0.001 and v != 0:
        return f"< 0.001"
    return f"{v:.{d}f}"


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _extract_trait_result(
    module_type: str,
    trait: Optional[str],
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract the single-trait sub-dict from a full module response.

    Handles three shapes:

    Shape 1 — New AnovaTraitResult / GeneticParametersTraitResult (flat):
        {trait, status, grand_mean, anova_table, mean_separation, ...}

    Shape 2 — Old GeneticsResponse wrapper from /genetics/analyze-upload:
        {status, mode, result: {grand_mean, anova_table, variance_components, ...}}

    Shape 3 — Full module response with trait_results dict:
        {dataset_token, trait_results: {trait_name: <shape 1>}}
    """
    # ── Shape 2: GeneticsResponse wrapper ────────────────────────────────────
    # result key holds the actual data dict; status/mode are metadata wrappers
    inner = analysis_result.get("result")
    if isinstance(inner, dict) and inner:
        data_markers = ("anova_table", "variance_components", "grand_mean",
                        "mean_separation", "heritability")
        if any(k in inner for k in data_markers):
            return inner

    # ── Shape 1: Already a flat single-trait dict ─────────────────────────────
    single_trait_keys = {
        "anova":              "anova_table",
        "genetic_parameters": "variance_components",
        "correlation":        "r_matrix",
        "heatmap":            "matrix",
    }
    marker = single_trait_keys.get(module_type, "")
    if marker and marker in analysis_result:
        return analysis_result
    # grand_mean present at top level → also flat
    if "grand_mean" in analysis_result:
        return analysis_result

    # ── Shape 3: Full module response with trait_results dict ─────────────────
    trait_results = analysis_result.get("trait_results") or {}
    if trait and trait in trait_results:
        sub = trait_results[trait]
        # sub might be a TraitResult dict with analysis_result nested
        if "analysis_result" in sub and sub["analysis_result"]:
            ar = sub["analysis_result"]
            # unwrap GeneticsResponse wrapper if present
            if isinstance(ar.get("result"), dict):
                return ar["result"]
            return ar
        return sub

    # Pick the first available trait result
    if trait_results:
        first = next(iter(trait_results.values()))
        if "analysis_result" in first and first["analysis_result"]:
            ar = first["analysis_result"]
            if isinstance(ar.get("result"), dict):
                return ar["result"]
            return ar
        return first

    # Correlation / heatmap — no trait key needed
    return analysis_result


def _format_anova_prompt(trait: str, result: Dict[str, Any]) -> str:
    lines = [f"TRAIT: {trait}"]

    gm = result.get("grand_mean")
    ng = result.get("n_genotypes")
    nr = result.get("n_reps")
    ne = result.get("n_environments")
    if gm is not None:
        lines.append(f"Grand mean: {gm:.4f}")
    if ng:
        lines.append(f"Genotypes: {ng}")
    if nr:
        lines.append(f"Replications: {nr}")
    if ne:
        lines.append(f"Environments: {ne}")

    # ANOVA table with η²
    at = result.get("anova_table") or {}
    sources = at.get("source") or []
    dfs     = at.get("df") or []
    sss     = at.get("ss") or []
    mss     = at.get("ms") or []
    fvals   = at.get("f_value") or []
    pvals   = at.get("p_value") or []

    if sources:
        total_ss = sum(s for s in sss if s is not None)
        lines.append("\nANOVA TABLE:")
        lines.append(f"  {'Source':<22} {'df':>4} {'F':>8} {'p':>8} {'η²':>7}")
        lines.append("  " + "-" * 55)
        for i, src in enumerate(sources):
            f   = fvals[i] if i < len(fvals) else None
            p   = pvals[i] if i < len(pvals) else None
            ss  = sss[i]   if i < len(sss)   else None
            df  = dfs[i]   if i < len(dfs)   else None
            eta = (ss / total_ss) if (ss is not None and total_ss > 0) else None
            lines.append(
                f"  {src:<22} {str(df) if df is not None else '—':>4} "
                f"{_fmt(f, 3):>8} {_fmt_p(p):>8} {_fmt(eta, 4):>7}"
            )

    # Mean separation
    ms = result.get("mean_separation") or {}
    genos  = ms.get("genotype") or []
    means  = ms.get("mean") or []
    groups = ms.get("group") or []
    test   = ms.get("test", "Tukey HSD")
    alpha  = ms.get("alpha", 0.05)

    if genos:
        lines.append(f"\nMEAN SEPARATION ({test}, α = {alpha}):")
        for i, g in enumerate(genos[:15]):
            m = means[i] if i < len(means) else None
            gr = groups[i] if i < len(groups) else "—"
            lines.append(f"  {i+1:>3}. {g:<25} mean={_fmt(m)} group={gr}")
        if len(genos) > 15:
            lines.append(f"  ... ({len(genos)-15} more genotypes)")

    # Assumption tests
    atest = result.get("assumption_tests") or {}
    if atest:
        lines.append("\nASSUMPTION TESTS:")
        for name, val in atest.items():
            if isinstance(val, dict):
                p   = val.get("p_value") or val.get("p.value") or val.get("p")
                sta = val.get("statistic") or val.get("W") or val.get("test_stat")
                lines.append(f"  {name}: stat={_fmt(sta,4)} p={_fmt_p(p)}")

    # Data warnings
    for w in (result.get("data_warnings") or []):
        lines.append(f"\n⚠ WARNING: {w}")

    return "\n".join(lines)


def _format_gp_prompt(trait: str, result: Dict[str, Any]) -> str:
    lines = [f"TRAIT: {trait}"]

    gm = result.get("grand_mean")
    if gm is not None:
        lines.append(f"Grand mean: {gm:.4f}")

    vc = result.get("variance_components") or {}
    if vc:
        lines.append("\nVARIANCE COMPONENTS:")
        for k, v in vc.items():
            lines.append(f"  {k}: {_fmt(v, 6)}")

    hp = result.get("heritability") or {}
    if hp:
        lines.append("\nHERITABILITY:")
        for k, v in hp.items():
            lines.append(f"  {k}: {_fmt(v, 4) if isinstance(v, float) else v}")

    lines.append("\nGENETIC PARAMETERS:")
    # Flat keys from new AnovaTraitResult / GeneticParametersTraitResult
    gcv = result.get("gcv")
    pcv = result.get("pcv")
    ga  = result.get("ga")
    gam = result.get("gam")
    # Nested dict from old GeneticsResponse.result.genetic_parameters
    gp_nested = result.get("genetic_parameters") or {}
    if gcv is None:
        gcv = gp_nested.get("GCV")
    if pcv is None:
        pcv = gp_nested.get("PCV")
    if ga is None:
        # ga (absolute) stored as GAM in old engine
        ga = gp_nested.get("GAM")
    if gam is None:
        gam = gp_nested.get("GAM_percent")
    if gcv is not None:
        lines.append(f"  GCV (%): {gcv:.2f}")
    if pcv is not None:
        lines.append(f"  PCV (%): {pcv:.2f}")
    if ga is not None:
        lines.append(f"  GA (absolute): {ga:.4f}")
    if gam is not None:
        lines.append(f"  GAM (%): {gam:.2f}")

    bi = result.get("breeding_implication")
    if bi:
        lines.append(f"\nBREEDING IMPLICATION (from R engine):\n{bi}")

    for w in (result.get("data_warnings") or []):
        lines.append(f"\n⚠ WARNING: {w}")

    return "\n".join(lines)


def _format_correlation_prompt(result: Dict[str, Any]) -> str:
    lines = []
    trait_names = result.get("trait_names") or []
    n = len(trait_names)
    n_obs = result.get("n_observations", "—")
    method = result.get("method", "pearson")

    lines.append(f"METHOD: {method.capitalize()}")
    lines.append(f"TRAITS ({n}): {', '.join(trait_names)}")
    lines.append(f"N OBSERVATIONS (genotype means): {n_obs}")

    r_mat = result.get("r_matrix") or []
    p_mat = result.get("p_matrix") or []
    if r_mat and trait_names:
        lines.append("\nCORRELATION TABLE (upper triangle only):")
        lines.append(f"  {'Trait A':<20} {'Trait B':<20} {'r':>7} {'p':>8} {'Sig':>5}")
        lines.append("  " + "-" * 62)
        for i in range(n):
            for j in range(i + 1, n):
                r = r_mat[i][j] if i < len(r_mat) and j < len(r_mat[i]) else None
                p = p_mat[i][j] if i < len(p_mat) and j < len(p_mat[i]) else None
                sig = "***" if p is not None and p < 0.001 else \
                      "**"  if p is not None and p < 0.01  else \
                      "*"   if p is not None and p < 0.05  else "ns"
                lines.append(
                    f"  {trait_names[i]:<20} {trait_names[j]:<20} "
                    f"{_fmt(r, 3):>7} {_fmt_p(p):>8} {sig:>5}"
                )

    for w in (result.get("warnings") or []):
        lines.append(f"\n⚠ WARNING: {w}")

    return "\n".join(lines)


def _format_heatmap_prompt(result: Dict[str, Any]) -> str:
    labels = result.get("labels") or []
    method = result.get("method", "pearson")
    lines = [
        f"METHOD: {method.capitalize()}",
        f"TRAITS ({len(labels)}): {', '.join(labels)}",
        f"VALUE RANGE: min={_fmt(result.get('min_val'), 3)} max={_fmt(result.get('max_val'), 3)}",
    ]
    matrix = result.get("matrix") or []
    if matrix and labels:
        n = len(labels)
        lines.append("\nMATRIX (off-diagonal only, |r| ≥ 0.40):")
        for i in range(n):
            for j in range(i + 1, n):
                r = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
                if r is not None and abs(r) >= 0.40:
                    lines.append(f"  {labels[i]} × {labels[j]}: r = {_fmt(r, 3)}")
    return "\n".join(lines)


# ============================================================================
# ANTHROPIC API CALL
# ============================================================================

async def _call_anthropic(system: str, user_text: str) -> str:
    """Non-streaming Anthropic call. Raises RuntimeError on failure."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    payload = {
        "model":      _MODEL,
        "max_tokens": _MAX_TOKENS,
        "system":     system,
        "messages": [{"role": "user", "content": user_text}],
    }
    headers = {
        "Content-Type":    "application/json",
        "x-api-key":       ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        response = await client.post(_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Anthropic API error {response.status_code}: "
                f"{response.text[:300]}"
            )
        data = response.json()
        return data["content"][0]["text"]


# ============================================================================
# REPAIR PASS
# ============================================================================

_REPAIR_SYSTEM = """\
You are a text editor. The text below failed an academic standards \
validation. Revise ONLY the flagged violations — do not change any \
other part of the text. Return the complete revised text.
"""

async def _repair_pass(
    original_text: str,
    violations: list,
    module_type: str,
) -> Tuple[str, ValidationResult]:
    """
    One repair attempt.  Returns (repaired_text, new_validation_result).
    """
    violation_lines = "\n".join(
        f"  [{v.severity.upper()}] Rule {v.rule_id}: {v.message}\n"
        f"  Excerpt: «{v.excerpt}»"
        for v in violations
        if v.severity == "block"
    )
    repair_prompt = (
        f"VIOLATIONS TO FIX:\n{violation_lines}\n\n"
        f"ORIGINAL TEXT:\n{original_text}"
    )
    try:
        repaired = await _call_anthropic(_REPAIR_SYSTEM, repair_prompt)
        new_result = AcademicValidator.validate(repaired, module_type)
        return repaired, new_result
    except Exception as exc:
        logger.warning("Repair pass failed: %s", exc)
        # Return original with unchanged validation
        return original_text, AcademicValidator.validate(original_text, module_type)


# ============================================================================
# SECTION PARSER
# ============================================================================

def _parse_sections(text: str) -> Dict[str, str]:
    """
    Split AI output on ── N. SECTION NAME ── markers.
    Returns {normalised_section_name: content}.

    Accepts multiple dash/line variants so the parser is not brittle to
    Claude rendering box chars (──) vs hyphens (--) vs em-dashes (—):
      ──  U+2500 pairs  (intended format, from system prompt)
      --  ASCII hyphens
      —   single em-dash on each side
      **SECTION NAME**  bold markdown fallback
    """
    # Primary: ── or -- or — delimiters, number + section name
    pattern = r'(?:──|--|—)\s*\d+\.\s*([A-Z][A-Z0-9\s/²&]+?)\s*(?:──|--|—)'
    parts = re.split(pattern, text)

    # Fallback: **NUMBER. SECTION NAME** bold markdown
    if len(parts) <= 1:
        pattern = r'\*\*\d+\.\s+([A-Z][A-Z0-9\s/²&]+?)\*\*'
        parts = re.split(pattern, text)
    sections: Dict[str, str] = {}

    # parts[0] = preamble (ignore)
    # parts[1::2] = section names
    # parts[2::2] = section content
    for i in range(1, len(parts) - 1, 2):
        name_raw = parts[i].strip().lower()
        name_key = re.sub(r'\s+', '_', name_raw)
        content  = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        sections[name_key] = content

    return sections


# ============================================================================
# FALLBACK BUILDERS (deterministic, no AI)
# ============================================================================

class _AnovaFallback:
    """Build safe ANOVA interpretation from raw data, no AI."""

    @staticmethod
    def build(trait: str, result: Dict[str, Any]) -> Dict[str, str]:
        s: Dict[str, str] = {}

        at = result.get("anova_table") or {}
        sources = at.get("source") or []
        fvals   = at.get("f_value") or []
        pvals   = at.get("p_value") or []
        sss     = at.get("ss") or []
        dfs     = at.get("df") or []

        geno_idx = next(
            (i for i, src in enumerate(sources) if src == "genotype"), None
        )

        total_ss = sum(x for x in sss if x is not None)

        n_g = result.get("n_genotypes")
        n_e = result.get("n_environments")
        
        exp_str = ""
        if n_g is not None:
            if n_e is not None and n_e > 1:
                exp_str = f" evaluated across {n_g} genotypes and {n_e} environments"
            else:
                exp_str = f" evaluated across {n_g} genotypes"

        # ── DATA QUALITY ────────────────────────────────────────────────────
        dq_lines = []
        for w in (result.get("data_warnings") or []):
            dq_lines.append(f"• {w}")
        s["data_quality_note"] = "\n".join(dq_lines) if dq_lines else \
            "No data quality concerns detected."

        # ── OVERALL FINDING ──────────────────────────────────────────────────
        if geno_idx is not None:
            f  = fvals[geno_idx] if geno_idx < len(fvals) else None
            p  = pvals[geno_idx] if geno_idx < len(pvals) else None
            ss = sss[geno_idx]   if geno_idx < len(sss)   else None
            eta = (ss / total_ss) if (ss and total_ss) else None
            sig = "significant" if (p is not None and p < 0.05) else "not significant"
            s["overall_finding"] = (
                f"An analysis of variance was conducted for {trait}{exp_str}. "
                f"The effect of genotype was {sig} (F = {_fmt(f, 3)}, p = {_fmt_p(p)}). "
                + (f"Genotype accounted for η² = {_fmt(eta, 4)} of total variance." if eta else "")
            )
        else:
            s["overall_finding"] = (
                f"An analysis of variance was conducted for {trait}{exp_str}. "
                f"Refer to the ANOVA table for genotype F-value, p-value, and η²."
            )

        # ── STATISTICAL EVIDENCE ─────────────────────────────────────────────
        evidence_lines = []
        for i, src in enumerate(sources):
            f  = fvals[i] if i < len(fvals) else None
            p  = pvals[i] if i < len(pvals) else None
            ss = sss[i]   if i < len(sss)   else None
            df = dfs[i]   if i < len(dfs)   else None
            eta = (ss / total_ss) if (ss and total_ss) else None
            evidence_lines.append(
                f"{src}: df={df}, F={_fmt(f,3)}, p={_fmt_p(p)}, η²={_fmt(eta,4)}"
            )
        s["statistical_evidence"] = "\n".join(evidence_lines) if evidence_lines else \
            "See ANOVA table."

        # ── GENOTYPE INTERPRETATION ──────────────────────────────────────────
        ms = result.get("mean_separation") or {}
        genos  = ms.get("genotype") or []
        means  = ms.get("mean") or []
        groups = ms.get("group") or []
        test   = ms.get("test", "post-hoc test")
        alpha  = ms.get("alpha", 0.05)

        if genos:
            top_g = genos[0]
            top_m = means[0] if means else None
            bot_g = genos[-1]
            bot_m = means[-1] if means else None
            s["genotype_interpretation"] = (
                f"The highest mean {trait} among the genotypes tested was "
                f"recorded in {top_g} (mean = {_fmt(top_m)}). "
                f"The lowest mean was recorded in {bot_g} "
                f"(mean = {_fmt(bot_m)}). "
                f"Means were separated using {test} at α = {alpha}. "
                f"Tukey group letters are shown in the Mean Separation table."
            )
        else:
            s["genotype_interpretation"] = \
                "Refer to the Mean Separation table for ranked means and Tukey group letters."

        # ── ASSUMPTION CHECK ─────────────────────────────────────────────────
        atest = result.get("assumption_tests") or {}
        alines = []
        all_pass = True
        for name, val in atest.items():
            if isinstance(val, dict):
                p   = val.get("p_value") or val.get("p.value") or val.get("p")
                sta = val.get("statistic") or val.get("W")
                verdict = "PASS" if (p is not None and p >= 0.05) else "FAIL"
                if verdict == "FAIL":
                    all_pass = False
                alines.append(
                    f"{name}: stat = {_fmt(sta, 4)}, p = {_fmt_p(p)} → {verdict}"
                )
        if not alines:
            s["assumption_check"] = "Assumption test data not available."
        elif all_pass:
            s["assumption_check"] = (
                "Assumption checks did not indicate a violation.\n" +
                "\n".join(alines)
            )
        else:
            s["assumption_check"] = "\n".join(alines)

        return s


class _GpFallback:
    @staticmethod
    def build(trait: str, result: Dict[str, Any]) -> Dict[str, str]:
        s: Dict[str, str] = {}

        vc  = result.get("variance_components") or {}
        hp  = result.get("heritability") or {}
        gcv = result.get("gcv")
        pcv = result.get("pcv")
        ga  = result.get("ga")
        gam = result.get("gam")
        h2  = hp.get("h2_broad_sense") if isinstance(hp, dict) else None

        h2_class = (
            "high" if h2 is not None and h2 >= 0.60 else
            "moderate" if h2 is not None and h2 >= 0.30 else
            "low" if h2 is not None else "not computed"
        )
        gam_class = (
            InterpretationEngine.classify_gam(gam)
            if gam is not None
            else "not computed"
        )

        n_g = result.get("n_genotypes")
        n_e = result.get("n_environments")
        
        exp_str = ""
        if n_g is not None:
            if n_e is not None and n_e > 1:
                exp_str = f" evaluated across {n_g} genotypes and {n_e} environments"
            else:
                exp_str = f" evaluated across {n_g} genotypes"

        s["data_quality_note"] = (
            "⚠ Negative genotypic variance detected. Treat estimates cautiously."
            if (vc.get("sigma2_genotype") or 0) < 0
            else "No data quality concerns detected."
        )

        s["overall_finding"] = (
            f"For {trait}{exp_str}: "
            f"H² = {_fmt(h2, 4)} ({h2_class} heritability), "
            f"GAM = {_fmt(gam, 2)}% ({gam_class} genetic advance)."
        )

        s["heritability_interpretation"] = (
            f"Broad-sense heritability H² = {_fmt(h2, 4)} indicates {h2_class} "
            f"genetic control of {trait} within this environment. "
            "Interpret heritability jointly with GAM (see next section)."
        )

        s["joint_h²_and_gam_interpretation"] = (
            f"H² = {_fmt(h2, 4)} ({h2_class}) and GAM = {_fmt(gam, 2)}% ({gam_class}). "
            "Refer to the Genetic Parameters table and interpretation section for details."
        )

        if gcv is not None and pcv is not None:
            diff = pcv - gcv
            env_label = (
                "minimal" if diff <= 2 else
                "appreciable" if diff <= 7 else "substantial"
            )
            s["gcv_vs_pcv_interpretation"] = (
                f"GCV = {gcv:.2f}%, PCV = {pcv:.2f}%. "
                f"Environmental influence on {trait} expression appears {env_label} "
                f"in this experiment (PCV − GCV = {diff:.2f}%)."
            )
        else:
            s["gcv_vs_pcv_interpretation"] = "GCV/PCV data not available."

        return s


class _CorrFallback:
    @staticmethod
    def build(result: Dict[str, Any]) -> Dict[str, str]:
        trait_names = result.get("trait_names") or []
        r_mat = result.get("r_matrix") or []
        p_mat = result.get("p_matrix") or []
        n = len(trait_names)
        method = result.get("method", "Pearson")
        n_obs = result.get("n_observations", "—")

        sig_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                r = r_mat[i][j] if i < len(r_mat) and j < len(r_mat[i]) else None
                p = p_mat[i][j] if i < len(p_mat) and j < len(p_mat[i]) else None
                if r is not None and p is not None and p < 0.05:
                    sig_pairs.append((trait_names[i], trait_names[j], r, p))

        s: Dict[str, str] = {}
        s["data_quality_note"] = (
            f"Correlations computed using {n_obs} treatment/level means. "
            f"Method: {method.capitalize()}."
        )
        s["overall_finding"] = (
            f"A phenotypic correlation analysis was conducted across {n_obs} treatment/level means. "
            f"Out of {n*(n-1)//2} trait pair(s) tested, "
            f"{len(sig_pairs)} were significant (p < 0.05) "
            f"among the levels tested in this experiment."
        )
        if sig_pairs:
            pair_lines = [
                f"  {a} × {b}: r = {_fmt(r, 3)}, p = {_fmt_p(p)}"
                for a, b, r, p in sig_pairs[:10]
            ]
            s["pairwise_interpretation"] = "\n".join(pair_lines)
        else:
            s["pairwise_interpretation"] = \
                "No significant pair correlations (p < 0.05) were detected in this experiment."

        s["causation_caution"] = (
            "Correlation does not imply causation. The associations observed "
            "reflect co-variation among treatment/level means in this experiment and "
            "do not establish directional biological relationships between traits."
        )
        return s


def _extract_column_names_for_domain(payload: Dict[str, Any]) -> List[str]:
    """Best-effort extraction of uploaded column names from analysis payloads."""
    cols: List[str] = []

    top_cols = payload.get("column_names")
    if isinstance(top_cols, list):
        cols.extend(str(c) for c in top_cols)

    inner = payload.get("result")
    if isinstance(inner, dict):
        inner_cols = inner.get("column_names")
        if isinstance(inner_cols, list):
            cols.extend(str(c) for c in inner_cols)

    trait_results = payload.get("trait_results")
    if isinstance(trait_results, dict):
        for tr in trait_results.values():
            if isinstance(tr, dict):
                tr_cols = tr.get("column_names")
                if isinstance(tr_cols, list):
                    cols.extend(str(c) for c in tr_cols)

    # Deduplicate while preserving order
    seen = set()
    unique_cols: List[str] = []
    for c in cols:
        if c not in seen:
            unique_cols.append(c)
            seen.add(c)
    return unique_cols


def _domain_language_guide(module_type: str, domain: str) -> str:
    """Return additional domain-specific language constraints for AI prompts."""
    if module_type == "genetic_parameters":
        return ""

    if domain == "plant_breeding":
        return (
            "Language guide: This appears to be plant breeding data. "
            "You may use genotype terminology where appropriate."
        )
    if domain == "agronomy":
        return (
            "Language guide: Use agronomy language: treatment effects, "
            "management recommendations, and agronomic response. "
            "Avoid plant-breeding terms unless a source is explicitly named Genotype."
        )
    if domain == "soil_science":
        return (
            "Language guide: Use soil science language: treatment effects on soil "
            "properties. Avoid plant-breeding terms unless a source is explicitly "
            "named Genotype."
        )
    return (
        "Language guide: Use neutral language: treatment differences and "
        "statistical findings. Avoid plant-breeding terms unless a source is "
        "explicitly named Genotype."
    )


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

async def interpret_module(
    request: AcademicInterpretRequest,
) -> AcademicInterpretationResponse:
    """
    Entry point called by POST /academic/interpret.

    Flow:
      extract_result → format_prompt → AI call → validate →
      (repair if needed) → (fallback if repair fails) → build response
    """
    module_type = request.module_type
    trait       = request.trait
    crop_ctx    = request.crop_context or ""

    # ── Extract single-trait (or module-level for corr/heatmap) result ─────
    flat = _extract_trait_result(module_type, trait, request.analysis_result)

    # Domain detection informs terminology for non-genetic-parameter modules
    candidate_cols = _extract_column_names_for_domain(request.analysis_result)
    domain = detect_analysis_domain(candidate_cols, module_type)
    language_guide = _domain_language_guide(module_type, domain)

    # ── Format data for Claude prompt ─────────────────────────────────────────
    if module_type == "anova":
        data_text = _format_anova_prompt(trait or "Trait", flat)
    elif module_type == "genetic_parameters":
        data_text = _format_gp_prompt(trait or "Trait", flat)
    elif module_type == "correlation":
        data_text = _format_correlation_prompt(flat)
    else:
        data_text = _format_heatmap_prompt(flat)

    user_message = (
        (f"Crop/study context: {crop_ctx}\n\n" if crop_ctx else "") +
        (f"{language_guide}\n\n" if language_guide else "") +
        f"Please interpret these {module_type.replace('_', ' ')} results:\n\n"
        + data_text
    )

    system_prompt = _SYSTEM_PROMPTS.get(module_type, _ANOVA_SYSTEM_PROMPT)

    # ── Attempt AI generation ────────────────────────────────────────────────
    ai_text: Optional[str] = None
    validation: Optional[ValidationResult] = None
    fallback_used = False
    ai_generated = False

    if ANTHROPIC_API_KEY:
        try:
            ai_text = await _call_anthropic(system_prompt, user_message)
            # Validate only sections 1–5 (substantive content).
            # Sections 6+ (GUIDED WRITING SUPPORT, SCOPE STATEMENT, CLOSING)
            # contain deliberate mentions of forbidden phrases as "do NOT use"
            # examples — these must not trigger the validator.
            _GUIDED_SECTION_PATTERN = re.compile(
                r'(?:──|--|—)\s*\d+\.\s*GUIDED WRITING',
                re.IGNORECASE,
            )
            _split = _GUIDED_SECTION_PATTERN.split(ai_text, maxsplit=1)
            text_for_validation = _split[0]
            validation = AcademicValidator.validate(text_for_validation, module_type)

            # Section-presence checks should reflect the actual generated output,
            # not the truncated pre-guided subset used for forbidden-phrase checks.
            # This avoids false warnings like MISSING_SECTION_EXAMINER_CHECKPOINT.
            if validation.violations:
                full_lower = ai_text.lower()
                reconciled_violations = []
                for v in validation.violations:
                    if v.rule_id.startswith("MISSING_SECTION_"):
                        m = re.search(r"\(section '([^']+)' not found\)", v.excerpt, re.IGNORECASE)
                        section_keyword = m.group(1).lower() if m else ""
                        if section_keyword and section_keyword in full_lower:
                            continue
                    reconciled_violations.append(v)

                if len(reconciled_violations) != len(validation.violations):
                    block_count = sum(1 for v in reconciled_violations if v.severity == "block")
                    warning_count = sum(1 for v in reconciled_violations if v.severity == "warn")
                    validation = ValidationResult(
                        passed=block_count == 0,
                        blocked=block_count > 0,
                        violations=reconciled_violations,
                        warning_count=warning_count,
                        block_count=block_count,
                    )

            if validation.blocked:
                logger.warning(
                    "AI text blocked (%d violations) — attempting repair for %s",
                    validation.block_count, module_type,
                )
                ai_text, validation = await _repair_pass(
                    ai_text, validation.violations, module_type
                )

            if validation.blocked:
                logger.warning(
                    "Repair also blocked — switching to fallback for %s", module_type
                )
                ai_text = None
                fallback_used = True
            else:
                ai_generated = True

        except Exception as exc:
            logger.error("AI call failed for %s: %s", module_type, exc)
            fallback_used = True
    else:
        logger.info("No ANTHROPIC_API_KEY — using deterministic fallback")
        fallback_used = True

    # ── Parse or build sections ───────────────────────────────────────────────
    if ai_text and not fallback_used:
        sections = _parse_sections(ai_text)
        raw_text = ai_text
    else:
        # Deterministic fallback
        raw_text = None
        if module_type == "anova":
            sections = _AnovaFallback.build(trait or "Trait", flat)
        elif module_type == "genetic_parameters":
            sections = _GpFallback.build(trait or "Trait", flat)
        elif module_type == "correlation":
            sections = _CorrFallback.build(flat)
        else:
            sections = {"overall_pattern": flat.get("interpretation") or ""}

        if validation is None:
            # Build a clean passing result for fallback
            validation = ValidationResult(passed=True, blocked=False)

    # ── Extract fixed fields from sections ────────────────────────────────────
    def _get(*keys: str) -> str:
        for k in keys:
            v = sections.get(k, "")
            if v:
                return v
        return ""

    overall_finding     = _get("overall_finding", "overall_pattern")
    statistical_evidence = _get("statistical_evidence", "overall_finding")

    module_sections: Dict[str, str] = {}
    skip = {
        "overall_finding", "overall_pattern", "statistical_evidence",
        "scope_statement", "examiner_checkpoint", "closing",
        "research_writing_referral", "guided_writing_support",
        "data_quality_note",
    }
    for k, v in sections.items():
        if k not in skip and v:
            module_sections[k] = v

    # ── Layer C — Guided Writing ──────────────────────────────────────────────
    guided = None
    if request.include_writing_support:
        guided = build_guided_writing(module_type, trait, flat)

    # ── Fixed sections ────────────────────────────────────────────────────────
    _is_split_plot = (
        module_type == "anova"
        and flat.get("design_type") == "split_plot_rcbd"
    )
    examiner_checkpoint = [
        "F-values reported for main-plot factor, subplot factor, and interaction",
        "Correct error stratum stated for each F-test (whole-plot vs. subplot residual)",
        "Design identified as split-plot RCBD with restricted randomisation",
        "CV% reported and classified",
        "At least one scope phrase present: 'in this experiment' or 'among the levels tested'",
    ] if _is_split_plot else [
        "F-value and p-value reported for the genotype effect",
        "Tukey group letters cited for all genotypes discussed",
        "η² effect size reported alongside p-value",
        "Assumption test results (Shapiro-Wilk, Levene) referenced",
        "At least one scope phrase present in the write-up",
    ] if module_type == "anova" else [
        "H² value and classification both stated",
        "GAM% stated jointly with H² — not reported alone",
        "GCV and PCV both cited and compared",
        "Breeding implication scoped to 'this environment'",
        "Any negative variance component warning cited if present",
    ] if module_type == "genetic_parameters" else [
        "r-value and p-value reported for every pair discussed",
        "Causation language absent from the write-up",
        "Scope limited to genotypes in this experiment",
        "Strong pairs (|r| ≥ 0.70) specifically identified",
        "Statement that 'correlation does not imply causation' included",
    ]

    return AcademicInterpretationResponse(
        module_type=module_type,
        trait=trait,
        overall_finding=overall_finding or "(see module sections)",
        statistical_evidence=statistical_evidence,
        module_sections=module_sections,
        scope_statement=_SCOPE_STATEMENT,
        examiner_checkpoint=examiner_checkpoint,
        closing=_CLOSING,
        research_writing_referral=_REFERRAL,
        guided_writing=guided,
        validator_result=validation,
        fallback_used=fallback_used,
        ai_generated=ai_generated,
        raw_ai_text=raw_text,
    )
