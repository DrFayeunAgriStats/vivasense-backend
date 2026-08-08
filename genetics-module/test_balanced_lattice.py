"""Regression suite for balanced-lattice block sizes.

Two things are covered, both against the production engine:

1. **Structural validity** — every advertised block size really produces a
   balanced lattice: k+1 replications, k blocks of k plots, every replication a
   complete replicate, and every treatment pair concurrent in exactly one block
   (lambda = 1). Verification reads the returned ``fieldbook``, which is a
   different data path from the ``plot_matrix`` that the generator's own
   ``_verify_balanced_lattice_pairwise`` inspects — so this is an independent
   check rather than a re-run of the engine's internal assertion.

2. **Three-way classification** — supported / not-yet-implemented / no valid
   construction, and the messages for those categories being genuinely
   different from one another rather than merely non-empty.
"""

import asyncio
from collections import defaultdict
from itertools import combinations

import pytest

from field_layout_generator import (
    LATTICE_MAX_BLOCK_SIZE,
    LATTICE_NOT_IMPLEMENTED,
    LATTICE_NO_CONSTRUCTION,
    LATTICE_SUPPORTED,
    classify_lattice_block_size,
    generate_field_layout,
    supported_lattice_block_sizes,
)
from field_layout_routes import list_lattice_block_sizes

# Structural verification is O(t^2) in pair counting; k=23 is 529 treatments
# and ~140k pairs, which is still fast but is kept out of the default sweep.
FAST_BLOCK_SIZES = [2, 3, 5, 7, 11, 13]
SLOW_BLOCK_SIZES = [17, 19, 23]


def build_lattice(block_size: int, seed: int = 42):
    treatments = [f"T{i}" for i in range(1, block_size * block_size + 1)]
    request = {
        "design_type": "lattice",
        "treatments": treatments,
        "replications": block_size + 1,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": seed,
    }
    return generate_field_layout(request), treatments


def assert_valid_balanced_lattice(block_size: int, fieldbook, treatments):
    """Assert the fieldbook is a genuine resolvable balanced lattice."""
    k = block_size
    t = k * k

    assert len(fieldbook) == t * (k + 1), "wrong total plot count"

    reps = defaultdict(list)
    blocks = defaultdict(list)
    for row in fieldbook:
        reps[row["rep"]].append(row["treatment"])
        blocks[(row["rep"], row["block"])].append(row["treatment"])

    assert len(reps) == k + 1, f"expected {k + 1} replications, got {len(reps)}"
    assert len(blocks) == k * (k + 1), "wrong total block count"

    # Resolvability: each replication contains every treatment exactly once.
    for rep, seen in reps.items():
        assert len(seen) == t, f"replication {rep} has {len(seen)} plots, expected {t}"
        assert len(set(seen)) == t, f"replication {rep} repeats a treatment"

    # Every block holds exactly k distinct treatments.
    for key, block_treatments in blocks.items():
        assert len(block_treatments) == k, f"block {key} has {len(block_treatments)} plots"
        assert len(set(block_treatments)) == k, f"block {key} repeats a treatment"

    # Pairwise balance: every treatment pair concurrent in exactly one block.
    pair_counts = defaultdict(int)
    for block_treatments in blocks.values():
        for a, b in combinations(sorted(block_treatments), 2):
            pair_counts[(a, b)] += 1

    offenders = [
        pair for pair in combinations(sorted(treatments), 2) if pair_counts.get(pair, 0) != 1
    ]
    assert not offenders, (
        f"{len(offenders)} treatment pairs are not concurrent exactly once "
        f"(lambda != 1); first offenders: {offenders[:3]}"
    )


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", FAST_BLOCK_SIZES)
def test_supported_block_size_is_a_valid_balanced_lattice(block_size):
    result, treatments = build_lattice(block_size)
    assert_valid_balanced_lattice(block_size, result["fieldbook"], treatments)


@pytest.mark.parametrize("block_size", SLOW_BLOCK_SIZES)
def test_large_supported_block_size_is_a_valid_balanced_lattice(block_size):
    result, treatments = build_lattice(block_size)
    assert_valid_balanced_lattice(block_size, result["fieldbook"], treatments)


def test_advertised_sizes_are_exactly_the_ones_verified_here():
    """Guard against advertising a block size no test structurally verifies."""
    assert supported_lattice_block_sizes() == sorted(FAST_BLOCK_SIZES + SLOW_BLOCK_SIZES)
    assert max(supported_lattice_block_sizes()) == LATTICE_MAX_BLOCK_SIZE


def test_layout_is_deterministic_for_a_fixed_seed():
    first, _ = build_lattice(5, seed=99)
    second, _ = build_lattice(5, seed=99)
    assert first["fieldbook"] == second["fieldbook"]


# ---------------------------------------------------------------------------
# Three-way classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
def test_primes_are_supported(block_size):
    classification = classify_lattice_block_size(block_size)
    assert classification["status"] == LATTICE_SUPPORTED
    assert classification["message"] == "", "supported sizes must carry no excuse"


@pytest.mark.parametrize("block_size", [4, 8, 9, 16, 25, 27, 32])
def test_prime_powers_are_not_yet_implemented(block_size):
    classification = classify_lattice_block_size(block_size)
    assert classification["status"] == LATTICE_NOT_IMPLEMENTED
    assert "not yet implement" in classification["message"]
    assert f"GF({block_size})" in classification["message"]


@pytest.mark.parametrize("block_size", [6, 10, 12, 14, 15, 18, 20, 21, 22])
def test_non_prime_powers_have_no_valid_construction(block_size):
    classification = classify_lattice_block_size(block_size)
    assert classification["status"] == LATTICE_NO_CONSTRUCTION
    assert "not yet implement" not in classification["message"]


def test_classification_reports_derived_geometry():
    classification = classify_lattice_block_size(7)
    assert classification["treatments"] == 49
    assert classification["replications"] == 8


def test_the_three_categories_produce_distinct_messages():
    not_implemented = classify_lattice_block_size(8)["message"]
    proven_impossible = classify_lattice_block_size(6)["message"]
    no_construction = classify_lattice_block_size(12)["message"]

    assert len({not_implemented, proven_impossible, no_construction}) == 3


def test_tarry_proof_is_claimed_only_for_block_size_six():
    """k=6 is a proven impossibility; k=12 existence is genuinely open."""
    assert "Tarry" in classify_lattice_block_size(6)["message"]
    assert "proven" in classify_lattice_block_size(6)["message"]

    for block_size in (10, 12, 14, 15, 18, 20, 21, 22):
        message = classify_lattice_block_size(block_size)["message"]
        assert "Tarry" not in message, f"Tarry overclaimed for k={block_size}"
        assert "proven" not in message, f"universal proof overclaimed for k={block_size}"


def test_messages_name_a_usable_alternative():
    for block_size in (4, 6, 8, 9, 10, 12, 16):
        assert "available block size" in classify_lattice_block_size(block_size)["message"]


# ---------------------------------------------------------------------------
# Block size vs treatment count labelling
# ---------------------------------------------------------------------------


def test_alternatives_state_treatment_counts_not_just_block_sizes():
    """Block size 8 sits between block sizes 7 and 11 - i.e. 49 and 121 treatments."""
    message = classify_lattice_block_size(8)["message"]
    assert "7 (49 treatments)" in message
    assert "11 (121 treatments)" in message


def test_square_block_sizes_disambiguate_against_treatment_counts():
    """Block size 25 means 625 treatments; a 25-treatment design is block size 5."""
    message = classify_lattice_block_size(25)["message"]
    assert "625 treatments" in message, "k=25 must state its real treatment count"
    assert "use block size 5" in message, "k=25 must point at the 25-treatment design"

    for block_size, root in ((4, 2), (9, 3)):
        hint = classify_lattice_block_size(block_size)["message"]
        assert f"use block size {root}" in hint


def test_no_confusion_hint_when_the_square_root_is_unusable():
    """16 = 4^2 but block size 4 is itself unsupported, so no hint is offered."""
    assert "use block size 4" not in classify_lattice_block_size(16)["message"]


# ---------------------------------------------------------------------------
# Capability endpoint
# ---------------------------------------------------------------------------


def test_endpoint_lists_every_block_size_with_a_reason():
    payload = asyncio.run(list_lattice_block_sizes())
    entries = payload["block_sizes"]

    assert [e["block_size"] for e in entries] == list(range(2, LATTICE_MAX_BLOCK_SIZE + 1))
    assert [e["block_size"] for e in entries if e["status"] == LATTICE_SUPPORTED] == (
        supported_lattice_block_sizes()
    )
    # Nothing is excluded silently: every unusable size explains itself.
    for entry in entries:
        if entry["status"] == LATTICE_SUPPORTED:
            assert entry["message"] == ""
        else:
            assert entry["message"], f"k={entry['block_size']} excluded without a reason"


def test_endpoint_respects_and_clamps_a_requested_ceiling():
    assert asyncio.run(list_lattice_block_sizes(max_block_size=9))["max_block_size"] == 9
    assert asyncio.run(list_lattice_block_sizes(max_block_size=10_000))["max_block_size"] == 100
    assert asyncio.run(list_lattice_block_sizes(max_block_size=-5))["max_block_size"] == 2


# ---------------------------------------------------------------------------
# Generation refuses unusable sizes with the classified message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [4, 6, 8, 10])
def test_generation_refuses_with_the_classified_message(block_size):
    expected = classify_lattice_block_size(block_size)["message"]
    with pytest.raises(ValueError) as excinfo:
        build_lattice(block_size)
    assert str(excinfo.value) == expected


def test_generation_no_longer_emits_the_old_generic_prime_message():
    for block_size in (4, 6, 8, 9, 10, 12):
        with pytest.raises(ValueError) as excinfo:
            build_lattice(block_size)
        assert "requires a prime block size" not in str(excinfo.value)
