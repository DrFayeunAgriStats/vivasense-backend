from genetics_interpretation import build_breeding_synthesis


def test_no_selection_recommendation_when_genotype_not_significant():
    text = build_breeding_synthesis(
        [
            {
                "trait_name": "Yield",
                "h2": 0.82,
                "gam_class": "High",
                "f_value": 1.234,
                "p_value": 0.321,
                "genotype_significant": False,
                "genotype_means": [
                    {"genotype": "G1", "mean": 10.0, "rank": 1, "group": "a"},
                    {"genotype": "G2", "mean": 9.8, "rank": 2, "group": "a"},
                ],
            }
        ]
    )

    assert "Genotypic variation for Yield was not statistically significant" in text
    assert "(F = 1.234, p = 0.321)" in text
    assert "making selection decisions." in text


def test_same_tukey_group_message_replaces_top_performer_claim():
    text = build_breeding_synthesis(
        [
            {
                "trait_name": "Plant Height",
                "h2": 0.76,
                "gam_class": "Medium",
                "f_value": 5.123,
                "p_value": 0.012,
                "genotype_significant": True,
                "genotype_means": [
                    {"genotype": "G1", "mean": 150.0, "rank": 1, "group": "a"},
                    {"genotype": "G2", "mean": 149.9, "rank": 2, "group": "a"},
                    {"genotype": "G3", "mean": 149.8, "rank": 3, "group": "a"},
                ],
            }
        ]
    )

    assert (
        "No genotype showed statistically superior performance. "
        "All genotypes were assigned to the same mean separation group."
    ) in text
    assert "consistently ranked among the top-performing" not in text
