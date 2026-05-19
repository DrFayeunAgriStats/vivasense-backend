"""
VivaSense — Shared Report Section Registry
==========================================
Ordered section definitions consumed by BOTH the Python Word export generator
and (mirrored in) the TypeScript UI layer (src/shared/reportSections.ts).

A section renders ONLY if:
  • corresponding data exists in the result object
  • the section is valid for the active domain

Both the combined-genetics export (genetics_export.py) and the
module-specific exports (export_module_routes.py) must iterate sections
in the order defined here.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from report_terms import TERMS


# ============================================================================
# SECTION DESCRIPTOR
# ============================================================================

@dataclass(frozen=True)
class ReportSection:
    id:      str
    title:   str
    level:   int                          # Word heading level (1=H1, 2=H2, 3=H3)
    domains: Optional[List[str]] = None  # None = all domains; list = restricted


# ============================================================================
# MASTER SECTION REGISTRY
# Both UI and Word export must iterate this list in order.
# ============================================================================

SECTIONS: List[ReportSection] = [
    ReportSection("executive_summary",       TERMS.executive_summary,       1),
    ReportSection("descriptive_stats",       TERMS.descriptive_stats,       2),
    ReportSection("anova",                   TERMS.anova,                   2),
    ReportSection("mean_separation",         TERMS.mean_separation,         2),
    ReportSection("treatment_variance",      TERMS.treatment_variance,      2),
    ReportSection("interpretation",          TERMS.interpretation,          2),
    ReportSection("writing_support",         TERMS.writing_support,         2),
    ReportSection("presubmission_checklist", TERMS.presubmission_checklist, 2),
]

# Sections that are rendered ONLY for the plant_breeding domain
GENETICS_ONLY_SECTION_IDS = {
    "genetic_parameters",
    "genetic_advance",
    "breeding_implication",
}

# Content that must NOT appear in agronomy or general-domain reports
AGRONOMY_FORBIDDEN_CONTENT = frozenset([
    "genetic advance",
    "selection intensity",
    "i = 1.40",
    "falconer & mackay",
    "falconer and mackay",
    "gcv",
    "pcv",
    "gam",
    "heritability",
    "breeding implication",
    "germplasm",
    "accession",
])


# ============================================================================
# SECTION LOOKUP HELPERS
# ============================================================================

def get_section(section_id: str) -> Optional[ReportSection]:
    """Return a section descriptor by id, or None if not found."""
    for sec in SECTIONS:
        if sec.id == section_id:
            return sec
    return None


def sections_for_domain(domain: Optional[str]) -> List[ReportSection]:
    """Return sections valid for the given domain (None = plant_breeding)."""
    domain = (domain or "plant_breeding").strip().lower()
    return [
        sec for sec in SECTIONS
        if sec.domains is None or domain in sec.domains
    ]
