import re

from domain_guard import find_forbidden_breeding_terms
from genetics_interpretation import generate_genetics_interpretation


_FORBIDDEN_TEXT_RX = re.compile(
    r"(?i)heritability|h2|h²|gam|gcv|pcv|genetic advance|genetic gain|breeding|"
    r"selection strategy|selection efficiency|genotype advancement|best genotype|top genotype|"
    r"additive variance|phenotypic selection|direct selection|germplasm|accession"
)


def test_agronomy_domain_has_no_breeding_terminology():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="Yield",
        h2=0.78,
        gam=12.4,
        gcv=10.2,
        pcv=11.1,
        gxe_significant=True,
        environment_significant=True,
        domain="agronomy",
    )
    combined = f"{interpretation} {implication}"
    assert _FORBIDDEN_TEXT_RX.search(combined) is None


def test_general_domain_has_no_breeding_terminology():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="Biomass",
        h2=0.62,
        gam=8.2,
        gcv=9.0,
        pcv=10.7,
        gxe_significant=False,
        environment_significant=True,
        domain="general",
    )
    combined = f"{interpretation} {implication}"
    assert _FORBIDDEN_TEXT_RX.search(combined) is None


def test_plant_breeding_domain_retains_breeding_support():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="Plant Height",
        h2=0.72,
        gam=11.0,
        gcv=8.2,
        pcv=9.1,
        gxe_significant=False,
        environment_significant=False,
        domain="plant_breeding",
    )
    combined = f"{interpretation} {implication}".lower()
    assert "heritability" in combined
    assert "selection" in combined


def test_forbidden_term_scan_finds_breeding_language():
    text = "Breeding implication: broad-sense heritability (H2) and GAM support genotype advancement."
    hits = find_forbidden_breeding_terms(text)
    assert hits
