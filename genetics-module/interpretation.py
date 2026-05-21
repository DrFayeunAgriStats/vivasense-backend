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
            return "Low"
        if gam_percent <= 10:
            return "Medium"
        return "High"

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

        # ── Joint H² + GAM interpretation ───────────────────────────────────
        if h2_class == "not_computed":
            interpretation = (
                f"Heritability could not be estimated for '{trait_name}'. "
                "Data limitations prevent reliable genetic interpretation."
            )
        elif h2_class == "high" and gam_class == "High":
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) indicates HIGH genetic control "
                f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is HIGH, "
                "suggesting substantial expected response to direct selection. "
                "High H² (broad-sense, entry-mean) supports reliable phenotypic selection. "
                "Additive vs non-additive gene action cannot be determined from broad-sense H² alone. "
                "Narrow-sense h² from mating designs is required."
            )
        elif h2_class == "high" and gam_class == "Medium":
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) indicates HIGH genetic control "
                f"of '{trait_name}'. The genetic advance as percent of mean (GAM = {gam:.2f}%) is MODERATE, "
                "indicating a meaningful selection response. The trait showed moderate expected response "
                "to phenotypic selection under the evaluated conditions."
            )
        elif h2_class == "high" and gam_class == "Low":
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) indicates HIGH genetic control, "
                f"yet the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW for '{trait_name}'. "
                "This dissociation suggests that while phenotypic variation is substantially genetic, the "
                "expected response to selection is limited under the evaluated conditions."
            )
        elif h2_class == "moderate" and gam_class == "High":
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) indicates MODERATE genetic control "
                f"of '{trait_name}', with the genetic advance as percent of mean (GAM = {gam:.2f}%) being HIGH. "
                "Selection response may be influenced by environmental conditions and should be interpreted cautiously."
            )
        elif h2_class == "moderate" and gam_class == "Medium":
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) and genetic advance as percent "
                f"of mean (GAM = {gam:.2f}%) both indicate MODERATE genetic control for '{trait_name}'. "
                "Selection response may be influenced by environmental conditions and should be interpreted cautiously."
            )
        elif h2_class == "moderate" and gam_class == "Low":
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) suggests MODERATE genetic control "
                f"of '{trait_name}', but the genetic advance as percent of mean (GAM = {gam:.2f}%) is LOW. "
                "Selection response may be influenced by environmental conditions and should be interpreted cautiously."
            )
        else:  # low h2
            interpretation = (
                f"The estimated broad-sense heritability (H\\u00b2 = {h2:.3f}) indicates LOW genetic control of "
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
                "Direct phenotypic selection may be effective under the conditions evaluated in this study. "
                "These results may support further evaluation of promising genotypes under additional testing environments."
            )
        elif h2_class == "moderate":
            recommendation = (
                "Selection response may be influenced by environmental conditions and should be interpreted cautiously. "
                "These results may support further evaluation of promising genotypes under additional testing environments."
            )
        else:  # low
            recommendation = (
                "Weak genetic basis under present conditions. Direct selection will be unreliable. "
                "These results may support further evaluation of promising genotypes under additional testing environments."
            )

        return {"interpretation": interpretation, "recommendation": recommendation}
