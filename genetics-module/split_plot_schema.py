from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AnovaResultRow(BaseModel):
    ms: float
    f_value: Optional[float] = Field(None, alias="f")
    p_value: Optional[float] = Field(None, alias="p")
    den_ms: float

class SplitPlotAnovaTable(BaseModel):
    factor_a: AnovaResultRow
    factor_b: AnovaResultRow
    interaction: AnovaResultRow

class CVResult(BaseModel):
    main_plot: float
    sub_plot: float

class SplitPlotOutput(BaseModel):
    design_verified: bool
    anova_table: SplitPlotAnovaTable
    cv: CVResult
    warnings: List[str]
    errors: List[str] = []