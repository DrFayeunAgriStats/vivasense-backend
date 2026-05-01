"""
VivaSense Interpretation Engine (Python)
=========================================
Single source of truth for genetic parameter classification and interpretation.
Mirrors the logic in vivasense_interpretation_engine.R.
"""

from __future__ import annotations


class InterpretationEngine:
    """Classify and interpret genetic parameters using plant breeding standards."""

    # ── Classification thresholds ────────────────────────────────────────────

    @staticmethod
    def classify_heritability(h2) -> str:
        if h2 is None:
            return "not_computed"
        try:
            h2 = float(h2)
        except (TypeError, ValueError):
            return "not_computed"
        if h2 < 0.30:
            return "low"
        if h2 < 0.60:
            return "moderate"
        return "high"

    @staticmethod
    def classify_gam(gam_percent) -> str:
        if gam_percent is None:
            return "not_computed"
        try:
            gam_percent = float(gam_percent)
        except (TypeError, ValueError):
            return "not_computed"
        if gam_percent < 5:
            return "low"
        if gam_percent <= 10:
            return "moderate"
        return "high"

    @staticmethod
    def classify_cv(cv) -> str:
        if cv is None:
            return "not_computed"
        try:
            cv = float(cv)
        except (TypeError, ValueError):
            return "not_computed"
        if cv < 10:
            return "low"
        if cv < 20:
            return "moderate"
        return "high"

    # ── Decision support ─────────────────────────────────────────────────────

    def generate_decision_support(
        self,
        trait_name: str,
        h2: float,
        gam: float,
        gcv: float,
        pcv: float,
    ) -> dict:
        """Return interpretation text and breeding recommendation for a trait.

        Parameters
        ----------
        trait_name : str
        h2         : broad-sense heritability (0–1)
        gam        : genetic advance as percent of mean (GAM%)
        gcv        : genotypic coefficient of variation (%)
        pcv        : phenotypic coefficient of variation (%)

        Returns
        -------
        dict with keys 'interpretation' and 'recommendation'
        """
        h2_class  = self.classify_heritability(h2)
        gam_class = self.classify_gam(gam)
        gcv_class = self.classify_cv(gcv)
        pcv_class = self.classify_cv(pcv)

        # ── Joint h² + GAM interpretation ───────────────────────────────────
        if h2_class == "not_computed":
            interpretation = (
                f"Heritability could not be estimated for '{trait_name}'. "
                "Data limitations prevent reliable genetic interpretation."
            )
        elif h2_class == "high" and gam_class == "high":
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) indicates HIGH genetic control "
                f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is HIGH, "
                "suggesting substantial expected response to direct selection. "
                "Additive gene effects are likely important; direct phenotypic selection should be effective."
            )
        elif h2_class == "high" and gam_class == "moderate":
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) indicates HIGH genetic control "
                f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is MODERATE, "
                "indicating a meaningful selection response. Direct phenotypic selection should yield steady "
                "genetic progress; both additive and non-additive gene effects likely contribute."
            )
        elif h2_class == "high" and gam_class == "low":
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) indicates HIGH genetic control, "
                f"yet the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW for '{trait_name}'. "
                "This dissociation suggests that while phenotypic variation is substantially genetic, the "
                "expected response to selection is limited. Non-additive gene effects or strong inbreeding "
                "depression may be responsible."
            )
        elif h2_class == "moderate" and gam_class == "high":
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) indicates MODERATE genetic control "
                f"of '{trait_name}', with the genetic advance as percent of mean (GAM = {gam:.2f}%) being HIGH. "
                "Useful selection response is achievable despite environmental influence. "
                "Both genetic and environmental management should be considered."
            )
        elif h2_class == "moderate" and gam_class == "moderate":
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) and genetic advance as percent "
                f"of mean (GAM = {gam:.2f}%) both indicate MODERATE genetic control for '{trait_name}'. "
                "Selection may be useful, though environmental factors remain important. "
                "Progress should be steady but not rapid."
            )
        elif h2_class == "moderate" and gam_class == "low":
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) suggests MODERATE genetic control "
                f"of '{trait_name}', but the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW. "
                "Direct phenotypic selection may be slow. Consider investigating additive effects more carefully "
                "or combining selection with environmental optimization."
            )
        else:  # low h2
            interpretation = (
                f"The estimated broad-sense heritability (h\u00b2 = {h2:.3f}) indicates LOW genetic control of "
                f"'{trait_name}' under the present environment. Phenotypic variation is dominated by environmental "
                "factors and/or measurement variation. Direct phenotypic selection is unlikely to be reliable; "
                "focus on improving growing conditions and management practices."
            )

        # ── GCV vs PCV addendum ──────────────────────────────────────────────
        if gcv is not None and pcv is not None:
            try:
                diff = float(pcv) - float(gcv)
                if diff <= 2:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is only slightly lower than the PCV ({pcv:.2f}%), "
                        "indicating limited environmental influence on trait expression."
                    )
                elif diff <= 7:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is moderately lower than the PCV ({pcv:.2f}%), "
                        "suggesting appreciable but not dominant environmental influence."
                    )
                else:
                    interpretation += (
                        f" The GCV ({gcv:.2f}%) is substantially lower than the PCV ({pcv:.2f}%), "
                        "indicating that environmental factors strongly affect trait expression."
                    )
            except (TypeError, ValueError):
                pass

        # ── Breeding recommendation ──────────────────────────────────────────
        if h2_class == "not_computed":
            recommendation = (
                "Heritability could not be reliably estimated. Redesign or expand the experiment "
                "to improve precision before making selection decisions."
            )
        elif h2_class == "high":
            recommendation = (
                "Strong genetic basis for the trait. Direct phenotypic selection should be effective "
                "in this environment. Prioritize identification and advancement of high-value individuals "
                "for next-generation breeding."
            )
        elif h2_class == "moderate":
            recommendation = (
                "Moderate genetic basis. Direct selection is possible but should be combined with "
                "attention to environmental standardization. Consider multi-environment evaluation "
                "to assess stability of selected genotypes."
            )
        else:  # low
            recommendation = (
                "Weak genetic basis under present conditions. Direct selection will be unreliable. "
                "Prioritize improvement of growing conditions, management practices, and measurement "
                "precision before intensifying selection."
            )

        return {"interpretation": interpretation, "recommendation": recommendation}
