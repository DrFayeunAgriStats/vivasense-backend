"""
Experimental-structure reconstruction for multi-environment trials.

Why this module exists
----------------------
The upload UI historically offered a single "Environment" slot. A trial run at
3 locations over 3 years has 9 true environments, but only one column could be
mapped, so researchers put Year in the Environment slot and Location wherever
there was room (typically Rep). The resulting model was statistically
well-formed but described an experiment that never happened.

Governing principle (Dr. Adunola):

    "Never allow a UI mapping convention to silently determine a statistical
     model. Conversely, never impose a statistical model on data whose
     experimental structure has not been established."

Two rules follow, and both are load-bearing here:

1. Environment = Location x Year is a *default candidate* for this shape of
   data, not a universal rule. An explicitly supplied Environment column always
   wins and is never silently overwritten by a constructed one.

2. Rep is nested within Environment. "Rep 1" at Location A and "Rep 1" at
   Location B are different physical units that happen to share a label. Rep
   must therefore never be treated as a globally unique identifier.

Scope
-----
Phase 1. Deliberately NOT here (tracked as Phase 2): the "experimental
structure detected" confirmation card, discrepancy flagging when a supplied
Environment column disagrees with the Location x Year candidate, and any
REML/mixed-model capability. Classical combined ANOVA remains valid once the
structure is correct — fixing structure is a separate question from choosing
the model, and is the one this module answers.

Generality
----------
The construction takes an ordered list of factor columns rather than a fixed
Location/Year pair, so season / management regime / irrigation can be combined
later without reworking the interface. No attempt is made to infer *which*
factors should combine — that stays an explicit researcher decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# Name of the column written when an environment is constructed. Prefixed so it
# cannot collide with a researcher's own column.
CONSTRUCTED_ENVIRONMENT_COLUMN = "_vivasense_environment"

# Name of the column written when rep labels must be re-expressed as
# within-environment ordinals. The researcher's own rep column is never
# overwritten — the original values stay intact and inspectable.
NESTED_REP_COLUMN = "_vivasense_rep_in_env"

# Separator used to build a readable composite environment label.
ENVIRONMENT_LABEL_SEPARATOR = " × "  # " × "

SOURCE_SUPPLIED = "supplied"
SOURCE_CONSTRUCTED = "constructed"
SOURCE_NONE = "none"


@dataclass
class EnvironmentStructure:
    """The experimental structure resolved from an uploaded dataset."""

    environment_column: Optional[str] = None
    #: "supplied" (researcher gave one), "constructed" (built from factors),
    #: or "none" (single-environment data).
    source: str = SOURCE_NONE
    #: Ordered factor columns combined to build the environment, when constructed.
    factor_columns: List[str] = field(default_factory=list)
    n_environments: Optional[int] = None

    #: Column the model should use for replication. Either the researcher's own
    #: rep column, or a derived within-environment ordinal when the original
    #: labels were globally unique.
    rep_column: Optional[str] = None
    original_rep_column: Optional[str] = None
    #: True when rep labels were re-expressed as within-environment ordinals.
    rep_renumbered: bool = False
    #: Replicates per environment when balanced; None when it varies.
    reps_per_environment: Optional[int] = None

    #: Human-readable notes for the client. Never silent.
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "environment_column": self.environment_column,
            "source": self.source,
            "factor_columns": list(self.factor_columns),
            "n_environments": self.n_environments,
            "rep_column": self.rep_column,
            "original_rep_column": self.original_rep_column,
            "rep_renumbered": self.rep_renumbered,
            "reps_per_environment": self.reps_per_environment,
            "notes": list(self.notes),
        }


def _present(df: pd.DataFrame, col: Optional[str]) -> Optional[str]:
    """Return the column name when it is a usable, present column."""
    if col is None:
        return None
    name = str(col).strip()
    if not name or name not in df.columns:
        return None
    return name


def build_environment_labels(df: pd.DataFrame, factor_columns: Sequence[str]) -> pd.Series:
    """Compose an environment label from the ordered factor columns.

    Values are stringified and joined, so Location='Ibadan', Year=2023 becomes
    'Ibadan × 2023'. Order is preserved so labels read consistently.
    """
    parts = [df[c].astype(str).str.strip() for c in factor_columns]
    combined = parts[0]
    for nxt in parts[1:]:
        combined = combined.str.cat(nxt, sep=ENVIRONMENT_LABEL_SEPARATOR)
    return combined


def _resolve_rep_nesting(
    df: pd.DataFrame,
    environment_column: str,
    rep_column: str,
    structure: EnvironmentStructure,
) -> None:
    """Ensure rep is interpreted as nested within environment.

    The combined-ANOVA model fits ``environment + environment:rep``, whose
    degrees of freedom equal e(r-1) — i.e. Rep(Environment) — only when rep
    labels repeat within each environment. If the uploaded rep labels are
    globally unique (Loc1-R1, Loc2-R1, ... all distinct), that term instead
    aliases and the nesting is lost. In that case the labels are re-expressed
    as within-environment ordinals in a NEW column; the original is untouched.
    """
    per_env_labels = {
        env: set(sub[rep_column].astype(str))
        for env, sub in df.groupby(environment_column, sort=False)
    }
    if not per_env_labels:
        return

    sizes = [len(s) for s in per_env_labels.values()]
    structure.reps_per_environment = sizes[0] if len(set(sizes)) == 1 else None

    union_size = len(set().union(*per_env_labels.values()))
    total_size = sum(sizes)

    # Labels reused across environments -> already nested-compatible.
    if union_size < total_size or len(per_env_labels) < 2:
        structure.rep_column = rep_column
        return

    # Fully disjoint labels across environments -> globally unique. Re-express
    # as within-environment ordinals so Rep(Environment) is estimable.
    ordinals = pd.Series(index=df.index, dtype="object")
    for env, sub in df.groupby(environment_column, sort=False):
        mapping = {
            label: str(i + 1)
            for i, label in enumerate(sorted(sub[rep_column].astype(str).unique()))
        }
        ordinals.loc[sub.index] = sub[rep_column].astype(str).map(mapping)

    df[NESTED_REP_COLUMN] = ordinals
    structure.rep_column = NESTED_REP_COLUMN
    structure.rep_renumbered = True
    structure.notes.append(
        f"Replication labels in '{rep_column}' were unique across environments, "
        "which would treat every plot as its own replicate. They were "
        "re-expressed as within-environment replicate numbers so replication is "
        "nested within environment. The original column is unchanged."
    )


def resolve_environment_structure(
    df: pd.DataFrame,
    environment_column: Optional[str] = None,
    environment_factor_columns: Optional[Sequence[str]] = None,
    rep_column: Optional[str] = None,
) -> EnvironmentStructure:
    """Establish the experimental structure of an uploaded dataset.

    Mutates ``df`` only by ADDING derived columns — never by altering or
    dropping a researcher's own column.

    Precedence, per constraint 1: an explicitly supplied environment column is
    always preserved and never overwritten by a constructed one. Construction
    happens only when no environment column was supplied.
    """
    structure = EnvironmentStructure()
    structure.original_rep_column = _present(df, rep_column)
    structure.rep_column = structure.original_rep_column

    supplied = _present(df, environment_column)
    factors = [c for c in (environment_factor_columns or []) if _present(df, c)]

    if supplied:
        # Constraint 1: explicit wins. Never silently overwritten.
        structure.environment_column = supplied
        structure.source = SOURCE_SUPPLIED
        structure.n_environments = int(df[supplied].nunique())
        if factors:
            # Phase 2 will compare the two and flag disagreement. Phase 1 states
            # plainly which one was used rather than deciding quietly.
            structure.notes.append(
                f"Using the supplied environment column '{supplied}' "
                f"({structure.n_environments} environments). The factor columns "
                f"{factors} were not combined, because an explicit environment "
                "column takes precedence."
            )
    elif len(factors) >= 2:
        labels = build_environment_labels(df, factors)
        df[CONSTRUCTED_ENVIRONMENT_COLUMN] = labels
        structure.environment_column = CONSTRUCTED_ENVIRONMENT_COLUMN
        structure.source = SOURCE_CONSTRUCTED
        structure.factor_columns = factors
        structure.n_environments = int(labels.nunique())
        observed = " x ".join(str(int(df[c].nunique())) for c in factors)
        structure.notes.append(
            f"Environment was constructed as the interaction of {' x '.join(factors)} "
            f"({observed} = {structure.n_environments} environments). Supply an "
            "explicit environment column to override this."
        )
    else:
        if len(factors) == 1:
            structure.notes.append(
                f"Only one environment factor ('{factors[0]}') was supplied, so no "
                "environment interaction was constructed. Provide a second factor "
                "(e.g. Year) or map an explicit environment column."
            )
        return structure

    if structure.environment_column and structure.original_rep_column:
        _resolve_rep_nesting(
            df, structure.environment_column, structure.original_rep_column, structure
        )

    return structure
