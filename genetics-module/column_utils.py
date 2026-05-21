import re
from typing import Dict, Tuple
import pandas as pd

def sanitise_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitise DataFrame column names for safe use in R formulas.
    Returns the modified DataFrame and a mapping of original → sanitised names.

    Rules:
      - Replace any character that is not alphanumeric or underscore with _
      - Collapse multiple consecutive underscores into one
      - Strip leading/trailing underscores
      - If result starts with a digit, prefix with 'col_'
      - Preserve original names in the mapping for display to user
    """
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