export type SelectionIntensityOption = {
  label: string;
  value: number;
  pct: number;
};

export const SELECTION_INTENSITY_OPTIONS = [
  { label: "Top 5%  — i = 2.063", value: 2.063, pct: 0.05 },
  { label: "Top 10% — i = 1.755", value: 1.755, pct: 0.10 },
  { label: "Top 20% — i = 1.400", value: 1.400, pct: 0.20 },
  { label: "Top 25% — i = 1.271", value: 1.271, pct: 0.25 },
  { label: "Top 50% — i = 0.798", value: 0.798, pct: 0.50 },
] as const;

export const DEFAULT_SELECTION_INTENSITY_OPTION: SelectionIntensityOption = {
  label: "Top 20% — i = 1.400",
  value: 1.400,
  pct: 0.20,
};

/** Numeric default — used wherever a plain `i` value is required. */
export const DEFAULT_SELECTION_INTENSITY = DEFAULT_SELECTION_INTENSITY_OPTION.value;

/** Backwards-compat alias. */
export const SELECTION_INTENSITIES = SELECTION_INTENSITY_OPTIONS;

export function inferSelectionPercent(selectionIntensity: number): number | null {
  if (!Number.isFinite(selectionIntensity)) {
    return null;
  }
  let best: SelectionIntensityOption | null = null;
  let bestDelta = Number.POSITIVE_INFINITY;

  for (const option of SELECTION_INTENSITY_OPTIONS) {
    const delta = Math.abs(option.value - selectionIntensity);
    if (delta < bestDelta) {
      best = option;
      bestDelta = delta;
    }
  }

  return best ? best.pct : null;
}

export function selectionIntensityDisclosure(selectionIntensity: number): string {
  const pct = inferSelectionPercent(selectionIntensity);
  const iText = selectionIntensity.toFixed(2);
  if (pct == null) {
    return `Genetic Advance estimated using i = ${iText} (Falconer & Mackay, 1996).`;
  }
  return `Genetic Advance estimated using i = ${iText} corresponding to ${pct}% selection intensity (Falconer & Mackay, 1996).`;
}
