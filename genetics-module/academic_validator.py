"""
VivaSense Academic Mentor — Layer A: Interpretation Validator
=============================================================

Pure Python rule engine.  No AI call, no R dependency, no I/O.
Takes a text string and returns a ValidationResult.

Rule categories (restored from V4.4 InterpretationValidator + V45Validator,
updated for genetics module and correlation):

  FORBIDDEN_WORDS      — specific words/phrases banned in any module (block)
  SPECULATION_PHRASES  — causal/mechanistic language (block)
  RECOMMENDATION_PHRASES — scope-exceeding advice (block)
  CAUSATION_PHRASES    — correlation-specific causal language (block)
  MISSING_SCOPE_PHRASE — at least one scope phrase required (warn)
  MODULE_SPECIFIC      — additional per-module rules (block or warn)

All rules are regex-based for precision.  Rule IDs are stable strings so
the frontend can localise error messages if needed.

Usage:
    from academic_validator import AcademicValidator
    result = AcademicValidator.validate(text, module_type="anova")
    if result.blocked:
        # use fallback
"""

from __future__ import annotations
import re
import logging
from typing import List, Tuple

from academic_schemas import ValidationResult, ValidationViolation

logger = logging.getLogger(__name__)


# ============================================================================
# RULE DEFINITIONS
# (pattern, rule_id, severity, message)
# ============================================================================

# ── Forbidden words — any module ─────────────────────────────────────────────
_FORBIDDEN_WORDS: List[Tuple[str, str, str]] = [
    # (regex_pattern, rule_id, fix_message)
    (r"\boptimal\b",              "FW_OPTIMAL",     "Replace 'optimal' with 'the highest mean among the levels tested'"),
    (r"\bnear[-\s]?optimal\b",    "FW_NEAR_OPT",    "Replace 'near-optimal' with specific comparative language"),
    (r"\bseverely\b",             "FW_SEVERELY",    "Replace 'severely' with specific quantitative language"),
    (r"\bunambiguous(ly)?\b",     "FW_UNAMBIG",     "Remove 'unambiguous' — state statistical values instead"),
    (r"\benormous(ly)?\b",        "FW_ENORMOUS",    "Replace 'enormous' with precise magnitude language"),
    (r"\bmassive(ly)?\b",         "FW_MASSIVE",     "Replace 'massive' with quantitative description"),
    (r"\bdramatic(ally)?\b",      "FW_DRAMATIC",    "Replace 'dramatic' with specific numerical comparison"),
    (r"\bstatistically decisive\b","FW_STAT_DEC",   "Remove 'statistically decisive' — use p-value instead"),
    (r"\bprove[sd]?\b",           "FW_PROVES",      "Replace 'proves/proved' — statistics do not prove, they provide evidence"),
    (r"\bproven\b",               "FW_PROVEN",      "Replace 'proven' — statistics do not prove, they provide evidence"),
    (r"\bconfirm[sed]?\b",        "FW_CONFIRMS",    "Replace 'confirms/confirmed' — use 'is consistent with' or 'showed'"),
    (r"\bexplained variance\b",   "FW_EXP_VAR",     "Replace 'explained variance' — η² is the proportion of TOTAL variance in this dataset"),
    (r"\bresults are reliable\b", "FW_RELIABLE",    "Remove 'results are reliable' — reliability is not established by a single assumption check"),
    (r"\brobust evidence\b",      "FW_ROBUST",      "Remove 'robust evidence' — describe the statistics, not the robustness"),
    (r"\boverwhelming(ly)?\b",    "FW_OVERWHELM",   "Remove 'overwhelmingly' — state the actual values"),
    (r"\bexceptional(ly)?\b",     "FW_EXCEPT",      "Remove 'exceptional' — describe specific values instead"),
    (r"\bgenetic superiority\b",  "FW_GEN_SUP",     "Remove 'genetic superiority' — describe h² and GAM values instead"),
    (r"\bp\s*=\s*0\b",            "FW_P_ZERO",      "Replace 'p = 0' — the smallest reportable value is p < 0.001"),
]

# ── Speculation / mechanistic language — any module ───────────────────────────
_SPECULATION_PHRASES: List[Tuple[str, str, str]] = [
    (r"likely due to",            "SP_LIKELY_DUE",  "Remove causal speculation — report only what the statistics show"),
    (r"because the plant",        "SP_BECAUSE_PLANT","Remove plant physiology speculation"),
    (r"nitrogen uptake",          "SP_N_UPTAKE",    "Remove physiological mechanism — not established by this analysis"),
    (r"leaf expansion",           "SP_LEAF_EXP",    "Remove physiological detail — not established by this analysis"),
    (r"genetic superiority",      "SP_GEN_SUP",     "Remove 'genetic superiority' — describe statistical estimates only"),
    (r"physiolog",                "SP_PHYSIO",      "Remove physiological explanation — not supported by statistical output alone"),
    (r"biochem",                  "SP_BIOCHEM",     "Remove biochemical explanation — not supported by statistical output alone"),
    (r"uptake efficiency",        "SP_UPTAKE_EFF",  "Remove mechanistic phrase — describe the statistical result only"),
    (r"likely caused by",         "SP_LIKELY_CAUSE","Remove causal speculation"),
    (r"can be attributed to",     "SP_ATTRIBUTED",  "Remove causal attribution — statistics describe association, not causation"),
    (r"this suggests that the",   "SP_SUGGEST_THE", "Rephrase — describe what was observed, not what it 'suggests'"),
    (r"indicates that the genotype","SP_IND_GENO",  "Rephrase to describe the statistical estimate, not a plant characteristic"),
    (r"experimental range was not extreme","SP_EXP_RANGE","Remove this phrase — do not explain why a result occurred"),
]

# ── Recommendation / scope-exceeding phrases — any module ────────────────────
_RECOMMENDATION_PHRASES: List[Tuple[str, str, str]] = [
    (r"recommend farmers",        "RC_REC_FARMERS", "Remove — single-experiment results do not support farmer recommendations"),
    (r"should be adopted widely", "RC_ADOPT_WIDE",  "Remove — adoption decisions require multi-location and multi-season evidence"),
    (r"breeding program should",  "RC_BREED_PROG",  "Remove — breeding program decisions exceed single-experiment scope"),
    (r"management protocol should","RC_MGMT_PROTO", "Remove — protocol recommendations exceed single-experiment scope"),
    (r"recommended for farmers",  "RC_REC_FOR",     "Remove — single-experiment scope cannot support farmer recommendations"),
    (r"widely recommended",       "RC_WIDELY_REC",  "Remove — scope is limited to this experiment"),
    (r"should be deployed",       "RC_DEPLOYED",    "Remove — deployment decisions exceed single-experiment scope"),
    (r"this variety should",      "RC_VAR_SHOULD",  "Remove — replace with 'this genotype showed the highest mean in this experiment'"),
    (r"select this",              "RC_SELECT_THIS", "Remove explicit selection recommendation — describe ranking only"),
]

# ── Causation language — correlation / heatmap modules only ──────────────────
_CAUSATION_PHRASES: List[Tuple[str, str, str]] = [
    (r"\bcauses\b",               "CA_CAUSES",      "Replace 'causes' — correlation does not establish causation"),
    (r"\bcaused by\b",            "CA_CAUSED_BY",   "Replace 'caused by' — describe the correlation statistic, not causation"),
    (r"\bleads to\b",             "CA_LEADS_TO",    "Replace 'leads to' — describe the r-value, not a causal pathway"),
    (r"\bresults in\b",           "CA_RESULTS_IN",  "Replace 'results in' — describe the correlation direction and magnitude"),
    (r"\bdrives\b",               "CA_DRIVES",      "Replace 'drives' — use 'is positively/negatively correlated with'"),
    (r"\binfluences\b",           "CA_INFLUENCES",  "Replace 'influences' — state the r-value and significance"),
    (r"\bdetermines\b",           "CA_DETERMINES",  "Replace 'determines' — correlation does not determine outcomes"),
]

# ── Scope phrases — at least one required ────────────────────────────────────
_SCOPE_PHRASES: List[str] = [
    "in this experiment",
    "among the levels tested",
    "among the genotypes tested",
    "within this dataset",
    "in this trial",
    "in the current study",
    "within this experiment",
    "under the conditions of this experiment",
    "within this context",
]

# ── Required section markers for ANOVA AI output ─────────────────────────────
_ANOVA_REQUIRED_SECTIONS: List[str] = [
    "data quality",
    "overall finding",
    "statistical evidence",
    "assumption",
    "examiner checkpoint",
]

# ── Required section markers for Genetic Parameters AI output ─────────────────
_GP_REQUIRED_SECTIONS: List[str] = [
    "data quality",
    "overall finding",
    "heritability",
    "guided writing",
    "examiner checkpoint",
]

# ── Required section markers for Correlation AI output ───────────────────────
_CORR_REQUIRED_SECTIONS: List[str] = [
    "data quality",
    "overall finding",
    "pairwise",
    "scope",
    "examiner checkpoint",
]


# ============================================================================
# VALIDATOR
# ============================================================================

class AcademicValidator:
    """
    Stateless validator.  All methods are class-methods; no instance needed.

    Usage:
        result = AcademicValidator.validate(text, module_type="anova")
    """

    @classmethod
    def validate(
        cls,
        text: str,
        module_type: str = "anova",
    ) -> ValidationResult:
        """
        Run all applicable rule categories against *text*.

        Parameters
        ----------
        text        : AI-generated or fallback interpretation text
        module_type : "anova" | "genetic_parameters" | "correlation" | "heatmap"

        Returns
        -------
        ValidationResult with passed=True when block_count == 0.
        """
        violations: List[ValidationViolation] = []

        lower = text.lower()

        # ── 1. Forbidden words (all modules) ─────────────────────────────────
        for pattern, rule_id, message in _FORBIDDEN_WORDS:
            match = re.search(pattern, lower)
            if match:
                excerpt = cls._excerpt(text, match.start(), match.end())
                violations.append(ValidationViolation(
                    rule_id=rule_id,
                    severity="block",
                    excerpt=excerpt,
                    message=message,
                ))

        # ── 2. Speculation phrases (all modules) ─────────────────────────────
        for pattern, rule_id, message in _SPECULATION_PHRASES:
            if re.search(pattern, lower):
                idx = lower.find(pattern.replace(r"\b", "").split(r"\s")[0])
                excerpt = cls._excerpt(text, max(0, idx), min(len(text), idx + 80))
                violations.append(ValidationViolation(
                    rule_id=rule_id,
                    severity="block",
                    excerpt=excerpt,
                    message=message,
                ))

        # ── 3. Recommendation phrases (all modules) ───────────────────────────
        for pattern, rule_id, message in _RECOMMENDATION_PHRASES:
            if re.search(pattern, lower):
                idx = lower.find(re.sub(r"\\b", "", pattern).split()[0])
                excerpt = cls._excerpt(text, max(0, idx), min(len(text), idx + 80))
                violations.append(ValidationViolation(
                    rule_id=rule_id,
                    severity="block",
                    excerpt=excerpt,
                    message=message,
                ))

        # ── 4. Causation phrases (correlation / heatmap only) ─────────────────
        if module_type in ("correlation", "heatmap"):
            for pattern, rule_id, message in _CAUSATION_PHRASES:
                if re.search(pattern, lower):
                    match = re.search(pattern, lower)
                    excerpt = cls._excerpt(text, match.start(), match.end() + 40)
                    violations.append(ValidationViolation(
                        rule_id=rule_id,
                        severity="block",
                        excerpt=excerpt,
                        message=message,
                    ))

        # ── 5. Missing scope phrase (all modules, warn) ───────────────────────
        has_scope = any(phrase in lower for phrase in _SCOPE_PHRASES)
        if not has_scope:
            violations.append(ValidationViolation(
                rule_id="MISSING_SCOPE",
                severity="warn",
                excerpt="(no scope phrase found in text)",
                message=(
                    "Add at least one scope phrase: 'in this experiment', "
                    "'among the levels tested', or 'within this dataset'."
                ),
            ))

        # ── 6. Module-specific section structure (warn) ───────────────────────
        if module_type == "anova":
            required = _ANOVA_REQUIRED_SECTIONS
        elif module_type == "genetic_parameters":
            required = _GP_REQUIRED_SECTIONS
        elif module_type in ("correlation", "heatmap"):
            required = _CORR_REQUIRED_SECTIONS
        else:
            required = []

        for section_keyword in required:
            if section_keyword.lower() not in lower:
                violations.append(ValidationViolation(
                    rule_id=f"MISSING_SECTION_{section_keyword.upper().replace(' ', '_')}",
                    severity="warn",
                    excerpt=f"(section '{section_keyword}' not found)",
                    message=f"Section '{section_keyword}' appears to be missing from the output.",
                ))

        # ── Tally ─────────────────────────────────────────────────────────────
        block_count   = sum(1 for v in violations if v.severity == "block")
        warning_count = sum(1 for v in violations if v.severity == "warn")

        result = ValidationResult(
            passed=block_count == 0,
            blocked=block_count > 0,
            violations=violations,
            warning_count=warning_count,
            block_count=block_count,
        )

        if result.blocked:
            logger.warning(
                "AcademicValidator: BLOCKED — %d violation(s) in %s output",
                block_count, module_type,
            )
        elif warning_count:
            logger.info(
                "AcademicValidator: PASSED with %d warning(s) in %s output",
                warning_count, module_type,
            )
        else:
            logger.info("AcademicValidator: PASSED clean for %s output", module_type)

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _excerpt(text: str, start: int, end: int, context: int = 30) -> str:
        """Return a short excerpt around [start:end] with surrounding context."""
        lo = max(0, start - context)
        hi = min(len(text), end + context)
        snippet = text[lo:hi].replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        return snippet
