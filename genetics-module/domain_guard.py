import re
from typing import List, Optional


def is_plant_breeding_domain(domain: Optional[str]) -> bool:
    return (domain or "plant_breeding").strip().lower() == "plant_breeding"


_FORBIDDEN_PATTERNS = (
    r"\bheritability\b",
    r"\bh[²2]\b",
    r"\bgam\b",
    r"\bgcv\b",
    r"\bpcv\b",
    r"\bgenetic advance\b",
    r"\bgenetic gain\b",
    r"\bbreeding\b",
    r"\bgenotype advancement\b",
    r"\bmultilocation advancement\b",
    r"\bselection effectiveness\b",
    r"\bselection strategy\b",
    r"\bselection efficiency\b",
    r"\bbreeding strategy\b",
    r"\belite genotype\b",
    r"\bsuperior genotype\b",
    r"\bcandidate genotype\b",
    r"\baccession\b",
    r"\bgermplasm\b",
    r"\badditive variance\b",
    r"\bphenotypic selection\b",
    r"\bdirect selection\b",
    r"\btrait inheritance\b",
    r"\bgenotype recommendation\b",
    r"\bbest genotype\b",
    r"\btop genotype\b",
    r"\bbreeding implication\b",
)
_FORBIDDEN_REGEX = [re.compile(p, flags=re.IGNORECASE) for p in _FORBIDDEN_PATTERNS]


def find_forbidden_breeding_terms(text: str) -> List[str]:
    hits: List[str] = []
    for rx in _FORBIDDEN_REGEX:
        if rx.search(text):
            hits.append(rx.pattern)
    return hits
