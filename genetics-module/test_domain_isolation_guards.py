from domain_guard import find_forbidden_breeding_terms, is_plant_breeding_domain
from genetics_interpretation import generate_genetics_interpretation
from genetics_export import _add_genetic_parameters_section
from genetics_schemas import GeneticsResult
from docx import Document


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
    assert find_forbidden_breeding_terms(combined) == []


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
    assert find_forbidden_breeding_terms(combined) == []


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


def test_domain_normalization_supports_space_and_hyphen():
    assert is_plant_breeding_domain("plant breeding")
    assert is_plant_breeding_domain("plant-breeding")


def test_non_plant_domain_adds_low_observation_caution():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="Yield",
        h2=0.51,
        gam=6.2,
        gcv=7.1,
        pcv=8.3,
        gxe_significant=False,
        environment_significant=False,
        n_observations=8,
        domain="agronomy",
    )
    assert "Only 8 observations were available" in interpretation
    assert find_forbidden_breeding_terms(f"{interpretation} {implication}") == []


def test_non_plant_domain_mentions_treatment_environment_interaction():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="Biomass",
        h2=0.45,
        gam=4.0,
        gcv=5.5,
        pcv=6.5,
        gxe_significant=True,
        environment_significant=True,
        domain="general",
    )
    assert "Treatment × environment interaction was significant" in interpretation
    assert "management recommendations" in implication
    assert find_forbidden_breeding_terms(f"{interpretation} {implication}") == []


def test_agronomy_export_guard_skips_genetic_parameters_section():
    doc = Document()
    result = GeneticsResult(
        environment_mode="single",
        n_genotypes=6,
        n_reps=3,
        n_environments=1,
        grand_mean=42.0,
        variance_components={"sigma2_genotype": 1.2, "sigma2_phenotypic": 2.1},
        heritability={"h2_broad_sense": 0.57},
        genetic_parameters={"GCV": 10.1, "PCV": 12.2, "GAM_percent": 14.4, "selection_intensity": 1.4},
    )

    _add_genetic_parameters_section(doc, result, domain="agronomy")

    rendered = " ".join(p.text for p in doc.paragraphs if p.text).lower()
    assert "genetic parameters" not in rendered
    assert "heritability" not in rendered
    assert "falconer" not in rendered
