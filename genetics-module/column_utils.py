import math
import re
from typing import Any, Dict, Tuple
import pandas as pd


# Matches an identifier that is a whole number carrying only trailing zeros
# after the decimal point ("13.0", "-4.00") — the float64 artifact — while
# leaving genuinely fractional codes ("2.5") untouched.
_WHOLE_FLOAT_STR = re.compile(r"^(-?\d+)\.0+$")


def format_label(value: Any, missing: str = "—") -> str:
    """Render an identifier (genotype / treatment / block / rep / environment)
    for display, without a spurious trailing ".0".

    Numeric identifier columns — e.g. `VAR NO` holding 1..20 rather than variety
    names — reach the record builder through `DataFrame.iterrows()`, which
    collapses each row to a single dtype. When *every* column in the row is
    numeric the row Series is upcast to float64, so a plain `str(row[col])`
    yields "13.0" instead of "13". Datasets carrying a string genotype-name
    column keep object dtype and are unaffected — which is why this surfaced
    only on numeric-ID datasets.

    Whole-valued numbers render as integers; genuinely fractional values keep
    their decimals; anything non-numeric passes through as text.
    """
    if value is None:
        return missing
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return missing
        # Repair labels already stringified as "13.0" upstream (cached datasets,
        # stored analysis history, or the R engine echoing a float-derived id).
        # Deliberately narrow: only a whole number with trailing zeros after the
        # point, so a genuinely fractional code like "2.5" is left alone.
        return _WHOLE_FLOAT_STR.sub(r"\1", s)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return missing
    if math.isinf(f):
        return str(value)
    if f.is_integer():
        return str(int(f))
    # Fractional identifiers are unusual but legitimate; keep the value intact
    # rather than rounding it into a different label.
    return str(value)


def clean_and_sanitise_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Clean and sanitise DataFrame column names for safe use in R formulas and robust validation.
    - Drops all-empty columns
    - Strips spaces from column names
    - Removes columns named 'Unnamed...'
    - Raises ValueError if any column header is blank
    - Sanitises names for R safety
    Returns the modified DataFrame and a mapping of original → sanitised names.
    """
    # Drop all-empty columns
    df = df.dropna(axis=1, how='all')
    # Strip spaces
    df.columns = df.columns.str.strip()
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Check for empty headers
    empty_cols = [col for col in df.columns if str(col).strip() == ""]
    if empty_cols:
        raise ValueError(
            "Your dataset contains empty column headers. "
            "Please open your file and ensure all columns have a valid name before uploading. "
            f"Problem columns: {empty_cols}"
        )

    # Sanitise for R
    mapping: Dict[str, str] = {}
    new_columns = []
    for original in df.columns:
        sanitised = re.sub(r'[^\w]', '_', str(original))
        sanitised = re.sub(r'_+', '_', sanitised)
        sanitised = sanitised.strip('_')
        if sanitised and sanitised[0].isdigit():
            sanitised = 'col_' + sanitised
        if not sanitised:
            sanitised = f'col_{len(mapping)}'
        mapping[str(original)] = sanitised
        new_columns.append(sanitised)
    df.columns = new_columns
    return df, mapping