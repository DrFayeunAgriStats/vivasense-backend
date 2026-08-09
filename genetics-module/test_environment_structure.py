"""
Regression suite for multi-environment structure reconstruction.

Reconstructs Sayo's trial shape — 3 locations x 3 years = 9 true environments,
replicates nested within environment — and verifies the resulting combined
ANOVA independently with statsmodels, the same path used for the CRD / RCBD /
split-plot / factorial fingerprints. The R engine is not involved: what is
under test is the experimental STRUCTURE handed to the model, not the model.

The canonical failure this guards against: Year mapped as Environment and
Location pushed into Rep, which yields a perfectly well-formed ANOVA of an
experiment that never took place.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols

from environment_structure import (
    CONSTRUCTED_ENVIRONMENT_COLUMN,
    NESTED_REP_COLUMN,
    SOURCE_CONSTRUCTED,
    SOURCE_NONE,
    SOURCE_SUPPLIED,
    resolve_environment_structure,
)

LOCATIONS = ["Ibadan", "Zaria", "Umudike"]
YEARS = [2022, 2023, 2024]
GENOTYPES = [f"G{i}" for i in range(1, 6)]
REPS = [1, 2, 3]

N_ENV = len(LOCATIONS) * len(YEARS)      # 9
N_GENO = len(GENOTYPES)                  # 5
N_REP = len(REPS)                        # 3


def make_trial(rep_labels_globally_unique: bool = False) -> pd.DataFrame:
    """3 loc x 3 yr x 5 genotypes x 3 reps = 135 observations."""
    rng = np.random.default_rng(20260809)
    rows = []
    for loc in LOCATIONS:
        for yr in YEARS:
            for rep in REPS:
                label = f"{loc[:3]}{yr}R{rep}" if rep_labels_globally_unique else rep
                for g in GENOTYPES:
                    rows.append({
                        "Location": loc,
                        "Year": yr,
                        "Rep": label,
                        "Genotype": g,
                        "Yield": (
                            2.0
                            + LOCATIONS.index(loc) * 0.4
                            + YEARS.index(yr) * 0.25
                            + GENOTYPES.index(g) * 0.3
                            + rng.normal(0, 0.15)
                        ),
                    })
    return pd.DataFrame(rows)


def combined_anova_df(df, env_col, rep_col):
    """Classical combined ANOVA: Env + Rep(Env) + Genotype + G x E."""
    d = df.rename(columns={env_col: "env", rep_col: "rep",
                           "Genotype": "geno", "Yield": "y"})
    for c in ("env", "rep", "geno"):
        d[c] = d[c].astype(str)
    model = ols("y ~ C(env) + C(env):C(rep) + C(geno) + C(geno):C(env)", data=d).fit()
    table = sm.stats.anova_lm(model, typ=1)
    return {str(i): int(r) for i, r in table["df"].items()}


# ---------------------------------------------------------------------------
# The bug being fixed
# ---------------------------------------------------------------------------


def test_year_as_environment_understates_the_experiment():
    """The original mis-mapping: Year as Environment, Location as Rep."""
    df = make_trial()
    assert df["Year"].nunique() == 3, "Year alone is only 3 levels, not 9"
    d = combined_anova_df(df, env_col="Year", rep_col="Location")
    assert d["C(env)"] == 2          # should be 8
    assert d["C(env):C(rep)"] == 6   # should be 18


# ---------------------------------------------------------------------------
# Constraint: Environment = Location x Year as the default candidate
# ---------------------------------------------------------------------------


def test_environment_constructed_from_location_and_year():
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    assert st.source == SOURCE_CONSTRUCTED
    assert st.environment_column == CONSTRUCTED_ENVIRONMENT_COLUMN
    assert st.factor_columns == ["Location", "Year"]
    assert st.n_environments == N_ENV
    assert st.reps_per_environment == N_REP
    assert st.notes, "a constructed environment must never be silent"


def test_constructed_environment_gives_correct_anova_degrees_of_freedom():
    """Independent statsmodels check against manual computation."""
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    d = combined_anova_df(df, st.environment_column, st.rep_column)

    expected = {
        "C(env)": N_ENV - 1,                              # 8
        "C(env):C(rep)": N_ENV * (N_REP - 1),             # 18  Rep(Environment)
        "C(geno)": N_GENO - 1,                            # 4
        "C(geno):C(env)": (N_GENO - 1) * (N_ENV - 1),     # 32
        "Residual": (N_GENO - 1) * (N_REP - 1) * N_ENV,   # 72
    }
    assert d == pytest.approx(expected), f"observed {d}, expected {expected}"
    # Degrees of freedom must exhaust the data exactly.
    assert sum(d.values()) + 1 == N_ENV * N_REP * N_GENO


def test_environment_labels_are_readable_and_ordered():
    df = make_trial()
    resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    labels = set(df[CONSTRUCTED_ENVIRONMENT_COLUMN])
    assert len(labels) == N_ENV
    assert "Ibadan × 2022" in labels


def test_construction_generalises_beyond_two_factors():
    """Not hardcoded to Location x Year — any ordered factor list composes."""
    df = make_trial()
    df["Irrigation"] = np.where(df.index % 2 == 0, "Rainfed", "Irrigated")
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year", "Irrigation"],
        rep_column="Rep",
    )
    assert st.source == SOURCE_CONSTRUCTED
    assert st.factor_columns == ["Location", "Year", "Irrigation"]
    assert st.n_environments == N_ENV * 2


# ---------------------------------------------------------------------------
# Constraint 1: an explicit Environment column is never overwritten
# ---------------------------------------------------------------------------


def test_supplied_environment_column_takes_precedence():
    df = make_trial()
    df["Env"] = df["Location"] + "-" + df["Year"].astype(str)
    st = resolve_environment_structure(
        df, environment_column="Env",
        environment_factor_columns=["Location", "Year"],   # both offered
        rep_column="Rep",
    )
    assert st.source == SOURCE_SUPPLIED
    assert st.environment_column == "Env"
    assert CONSTRUCTED_ENVIRONMENT_COLUMN not in df.columns
    assert any("takes precedence" in n for n in st.notes)


def test_supplied_environment_column_is_used_even_when_it_disagrees():
    """Phase 2 flags disagreement; Phase 1 must still never overwrite."""
    df = make_trial()
    df["Env"] = df["Year"].astype(str)      # only 3 levels, disagrees with 9
    st = resolve_environment_structure(
        df, environment_column="Env",
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    assert st.source == SOURCE_SUPPLIED
    assert st.n_environments == 3


# ---------------------------------------------------------------------------
# Constraint 2: Rep is nested within Environment, never globally unique
# ---------------------------------------------------------------------------


def test_shared_rep_labels_are_left_untouched():
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    assert st.rep_renumbered is False
    assert st.rep_column == "Rep"
    assert NESTED_REP_COLUMN not in df.columns


def test_globally_unique_rep_labels_are_renested():
    df = make_trial(rep_labels_globally_unique=True)
    assert df["Rep"].nunique() == N_ENV * N_REP      # 27 distinct labels

    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    assert st.rep_renumbered is True
    assert st.rep_column == NESTED_REP_COLUMN
    assert df[NESTED_REP_COLUMN].nunique() == N_REP
    assert st.reps_per_environment == N_REP
    # The researcher's own column must survive untouched.
    assert df["Rep"].nunique() == N_ENV * N_REP
    assert any("nested within environment" in n for n in st.notes)


def test_renested_reps_restore_correct_nesting_degrees_of_freedom():
    df = make_trial(rep_labels_globally_unique=True)
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    d = combined_anova_df(df, st.environment_column, st.rep_column)
    assert d["C(env):C(rep)"] == N_ENV * (N_REP - 1)   # 18


def test_globally_unique_reps_are_not_estimable_without_renesting():
    """Shows why constraint 2 matters: the raw labels oversaturate the design."""
    df = make_trial(rep_labels_globally_unique=True)
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column="Rep",
    )
    with pytest.raises(Exception):
        combined_anova_df(df, st.environment_column, "Rep")


# ---------------------------------------------------------------------------
# Single-environment paths must be untouched
# ---------------------------------------------------------------------------


def test_no_factors_means_no_construction():
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None, environment_factor_columns=[], rep_column="Rep",
    )
    assert st.source == SOURCE_NONE
    assert st.environment_column is None
    assert st.rep_column == "Rep"
    assert CONSTRUCTED_ENVIRONMENT_COLUMN not in df.columns
    assert NESTED_REP_COLUMN not in df.columns


def test_single_factor_does_not_construct_an_environment():
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location"], rep_column="Rep",
    )
    assert st.source == SOURCE_NONE
    assert st.environment_column is None
    assert st.notes, "the researcher must be told why nothing was constructed"


def test_missing_columns_are_ignored_rather_than_crashing():
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "NotAColumn"], rep_column="Rep",
    )
    assert st.source == SOURCE_NONE      # only one usable factor remains


def test_crd_style_upload_without_rep_column_is_safe():
    df = make_trial()
    st = resolve_environment_structure(
        df, environment_column=None,
        environment_factor_columns=["Location", "Year"], rep_column=None,
    )
    assert st.source == SOURCE_CONSTRUCTED
    assert st.rep_column is None
    assert st.rep_renumbered is False
