/**
 * Shared normalization and validation utilities for VivaSense genetics modules.
 * Prevents crashes from malformed or missing server data.
 */

// ─────────────────────────────────────────────────────────────────────────────
// MATRIX UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ensure value is a proper n×n matrix of numbers/nulls.
 * Returns [] if the matrix is missing, malformed, or wrong size.
 */
export function normalizeMatrix(matrix: unknown, n: number): (number | null)[][] {
  if (!Array.isArray(matrix) || matrix.length !== n) return [];
  for (const row of matrix) {
    if (!Array.isArray(row) || row.length !== n) return [];
  }
  return matrix as (number | null)[][];
}

/**
 * True when matrix is a valid n×n grid with n > 0.
 */
export function hasValidMatrix(matrix: unknown, n: number): boolean {
  if (n === 0) return false;
  if (!Array.isArray(matrix) || matrix.length !== n) return false;
  return matrix.every((row) => Array.isArray(row) && row.length === n);
}

/**
 * Flatten an n×n matrix into a 1-D array of cells, or [] if invalid.
 */
export function flattenMatrix<T>(
  matrix: T[][],
  n: number
): T[] {
  if (!hasValidMatrix(matrix, n)) return [];
  return matrix.flat();
}

// ─────────────────────────────────────────────────────────────────────────────
// ARRAY UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/** Return value if Array, else []. */
export function safeArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

/** Return value if number, else fallback. */
export function safeNumber(value: unknown, fallback: number = 0): number {
  return typeof value === "number" && isFinite(value) ? value : fallback;
}

// ─────────────────────────────────────────────────────────────────────────────
// ANOVA NORMALIZATION
// ─────────────────────────────────────────────────────────────────────────────

export interface NormalizedAnovaTable {
  source: string[];
  df: number[];
  ss: (number | null)[];
  ms: (number | null)[];
  f_value: (number | null)[];
  p_value: (number | null)[];
  hasData: boolean;
}

export function normalizeAnovaTable(raw: unknown): NormalizedAnovaTable {
  const src = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const source = safeArray<string>(src.source);
  const hasData = source.length > 0;
  return {
    source,
    df: safeArray<number>(src.df),
    ss: safeArray<number | null>(src.ss),
    ms: safeArray<number | null>(src.ms),
    f_value: safeArray<number | null>(src.f_value),
    p_value: safeArray<number | null>(src.p_value),
    hasData,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// GENETIC PARAMETERS NORMALIZATION
// ─────────────────────────────────────────────────────────────────────────────

export interface NormalizedVarianceComponents {
  entries: { key: string; value: number }[];
  hasData: boolean;
}

export function normalizeVarianceComponents(raw: unknown): NormalizedVarianceComponents {
  if (!raw || typeof raw !== "object") return { entries: [], hasData: false };
  const entries = Object.entries(raw as Record<string, unknown>)
    .filter(([, v]) => typeof v === "number" && isFinite(v as number))
    .map(([key, value]) => ({ key, value: value as number }));
  return { entries, hasData: entries.length > 0 };
}

// ─────────────────────────────────────────────────────────────────────────────
// PREVIEW SECTION BUILDER TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface PreviewSection {
  title: string;
  rows: { label: string; value: string }[];
  note?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// HEATMAP NORMALIZATION
// ─────────────────────────────────────────────────────────────────────────────

export interface NormalizedHeatmapData {
  matrixFlat: (number | null)[];
  traitNames: string[];
  n: number;
  hasData: boolean;
  mode?: string;
}

export function normalizeHeatmapData(
  matrix: unknown,
  traitNames: unknown,
  n: number,
  mode?: string
): NormalizedHeatmapData {
  const names = safeArray<string>(traitNames);
  const isValid = hasValidMatrix(matrix, n) && names.length === n;
  const flat = isValid ? (matrix as (number | null)[][]).flat() : [];
  return {
    matrixFlat: flat,
    traitNames: names,
    n,
    hasData: isValid && flat.length === n * n,
    mode,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// DEBUG LOGGING
// ─────────────────────────────────────────────────────────────────────────────

/** Dev-only console.log — stripped in production by tree-shaking. */
export function logDebug(label: string, data: Record<string, unknown>): void {
  if (process.env.NODE_ENV !== "production") {
    console.log(`[VivaSense:${label}]`, data);
  }
}
