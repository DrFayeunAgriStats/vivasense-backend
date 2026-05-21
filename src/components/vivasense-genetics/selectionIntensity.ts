/**
 * selectionIntensity
 * ==================
 * Constants and helpers for the selection-intensity parameter used in
 * Genetic Advance (GA / GAM) calculations.
 *
 * The standardised selection differential (i) is taken from Falconer & Mackay
 * (1996, Table A), rounded to 3 decimal places.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface SelectionIntensityOption {
  /** Percentage selected (top p%). */
  pct: number;
  /** Standardised selection differential (i). */
  value: number;
  /** Human-readable label, e.g. "Top 5%" */
  label: string;
  /** Brief agronomic note shown beside the option. */
  note: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** The default selection intensity used when none is specified (20 %, i = 1.400). */
export const DEFAULT_SELECTION_INTENSITY = 1.400;

export const SELECTION_INTENSITIES: SelectionIntensityOption[] = [
  { label: "Top 5%  — i = 2.063", value: 2.063, pct: 0.05, note: "Stringent — elite variety development" },
  { label: "Top 10% — i = 1.755", value: 1.755, pct: 0.10, note: "Standard — recommended for most breeding programs" },
  { label: "Top 20% — i = 1.400", value: 1.400, pct: 0.20, note: "Moderate — early-generation selection" },
  { label: "Top 25% — i = 1.271", value: 1.271, pct: 0.25, note: "" },
  { label: "Top 50% — i = 0.798", value: 0.798, pct: 0.50, note: "Low intensity — exploratory analysis" },
];

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Return a one-sentence disclosure string about the assumed selection
 * intensity, suitable for display beneath analysis results.
 */
export function selectionIntensityDisclosure(intensity: number): string {
  const match = SELECTION_INTENSITIES.find((o) => Math.abs(o.value - intensity) < 0.001);
  const pctStr = match ? `${match.pct}%` : `i = ${intensity.toFixed(3)}`;
  return (
    `Genetic Advance (GA) and GAM% are computed assuming selection of the top ` +
    `${pctStr} of individuals (i = ${intensity.toFixed(3)}). ` +
    `Change this value in the column mapping step if your breeding programme ` +
    `uses a different selection fraction.`
  );
}
