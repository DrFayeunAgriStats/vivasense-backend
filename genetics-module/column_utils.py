import re
from typing import Dict, Tuple
import pandas as pd

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
        raise ValueError("Your dataset contains empty column headers. Please check your file and re-upload.")

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