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

/** The default selection intensity used when none is specified (10 %, i = 1.755). */
export const DEFAULT_SELECTION_INTENSITY = 1.755;

export const SELECTION_INTENSITIES: SelectionIntensityOption[] = [
  { pct: 1,  value: 2.665, label: "Top 1%",  note: "Very stringent — mass selection schemes" },
  { pct: 5,  value: 2.063, label: "Top 5%",  note: "Stringent — elite variety development" },
  { pct: 10, value: 1.755, label: "Top 10%", note: "Standard — recommended for most breeding programs" },
  { pct: 20, value: 1.400, label: "Top 20%", note: "Moderate — early-generation selection" },
  { pct: 30, value: 1.159, label: "Top 30%", note: "Lenient — large population screening" },
  { pct: 50, value: 0.798, label: "Top 50%", note: "Low intensity — exploratory analysis" },
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
