import React from "react";
import type { AnovaTable } from "@/services/geneticsUploadApi";

/**
 * Split-Plot ANOVA presentation. Renders sources in the canonical order with
 * stratum labels and two separate CV% values.
 *
 * Backend fields used (defensive):
 *   - anova_table (existing)
 *   - result.cv_main_plot_pct, result.cv_sub_plot_pct  (preferred)
 *   - falls back to "pending backend update" if missing
 */

interface Props {
  anovaTable: AnovaTable;
  cvMainPlot?: number | null;
  cvSubPlot?: number | null;
}

type Stratum = {
  matches: (src: string) => boolean;
  label: string;
  badge?: string;
  note?: string;
};

const SPLIT_PLOT_ORDER: Stratum[] = [
  { matches: (s) => /^rep|replication|block/i.test(s), label: "Replication" },
  {
    matches: (s) =>
      /main[\s_-]?plot/i.test(s) ||
      (!/rep/i.test(s) && /(^|[^a-z])a([^a-z]|$)/i.test(s) && !/error/i.test(s)) ||
      /factor[\s_-]?a/i.test(s),
    label: "Main Plot Factor (A)",
    badge: "F tested against Error A (Rep × Main Plot)",
  },
  {
    matches: (s) =>
      /error[\s_-]?a/i.test(s) || /rep[:×x]/i.test(s) || /(main[\s_-]?plot.*error)|(stratum[\s_-]?a)/i.test(s),
    label: "Error A (Rep × Main Plot)",
    badge: "Main Plot Error Stratum",
  },
  {
    matches: (s) =>
      /sub[\s_-]?plot/i.test(s) || /factor[\s_-]?b/i.test(s),
    label: "Sub Plot Factor (B)",
    badge: "F tested against Error B (Residual)",
  },
  {
    matches: (s) => /:|×|(?:^|\s)x(?:\s|$)/i.test(s) && !/^rep/i.test(s) && !/error/i.test(s),
    label: "A × B Interaction",
    badge: "F tested against Error B (Residual)",
  },
  {
    matches: (s) => /error[\s_-]?b|residual/i.test(s),
    label: "Error B (Residual)",
    badge: "Sub Plot Error Stratum",
  },
];

function fmtP(p: number | null) {
  if (p === null || p === undefined) return { text: "—", stars: "" };
  if (p < 0.001) return { text: p.toExponential(2), stars: "***" };
  if (p < 0.01) return { text: p.toFixed(4), stars: "**" };
  if (p < 0.05) return { text: p.toFixed(4), stars: "*" };
  return { text: p.toFixed(4), stars: "ns" };
}

export function SplitPlotResults({ anovaTable, cvMainPlot, cvSubPlot }: Props) {
  // η² computation: SS_total = sum of all SS (excluding intercept rows)
  const ssAll = anovaTable.ss as (number | null)[];
  const ssTotal = ssAll.reduce((sum, v) => (v != null ? sum + v : sum), 0);

  // Re-order rows. Each source row may map to one stratum; unmatched go at end.
  const rows = anovaTable.source.map((src, idx) => ({ src, idx }));
  const ordered: Array<{ src: string; idx: number; stratum?: Stratum }> = [];
  const used = new Set<number>();

  for (const stratum of SPLIT_PLOT_ORDER) {
    for (const r of rows) {
      if (used.has(r.idx)) continue;
      if (stratum.matches(r.src)) {
        ordered.push({ ...r, stratum });
        used.add(r.idx);
        break;
      }
    }
  }
  // Append leftovers
  for (const r of rows) if (!used.has(r.idx)) ordered.push({ ...r });

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700">
          Split-Plot Design
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Source</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">df</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">SS</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">MS</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">F-value</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">P-value</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">η²</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map(({ src, idx, stratum }) => {
              const { text: pText, stars } = fmtP(anovaTable.p_value[idx] ?? null);
              const isError = /error|residual/i.test(stratum?.label ?? src);
              const label = stratum?.label ?? src;
              const ms = anovaTable.ms[idx];
              const isErrorA = stratum?.label === "Error A (Rep × Main Plot)";
              const ss_i = ssAll[idx];
              const eta2Str = (!isError && ss_i != null && ssTotal > 0)
                ? (ss_i / ssTotal).toFixed(3)
                : "—";
              return (
                <tr key={`${src}-${idx}`} className={isError ? "bg-gray-50/70" : "bg-white"}>
                  <td className="px-3 py-1.5">
                    <div className="font-medium text-gray-800">{label}</div>
                    {stratum?.badge && (
                      <div className="mt-0.5 text-[10px] font-medium uppercase tracking-wide text-gray-500">
                        {stratum.badge}
                      </div>
                    )}
                    {isErrorA && ms != null && (
                      <div className="mt-0.5 text-[11px] font-semibold text-emerald-700">
                        MS Error A = {ms.toFixed(3)}
                      </div>
                    )}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">{anovaTable.df[idx]}</td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {anovaTable.ss[idx] != null ? (anovaTable.ss[idx] as number).toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {ms != null ? ms.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {anovaTable.f_value[idx] != null ? (anovaTable.f_value[idx] as number).toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {pText}
                    {stars && <span className="ml-1 font-semibold text-emerald-700">{stars}</span>}
                  </td>
                  <td className="px-3 py-1.5 text-gray-500">{eta2Str}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        <CvCard label="CV Main Plot (%)" value={cvMainPlot} />
        <CvCard label="CV Sub Plot (%)" value={cvSubPlot} />
      </div>

      <p className="text-[11px] text-gray-400">
        *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns p≥0.05 · Error A and Error B are reported
        separately — pooled error is not used.
      </p>
    </div>
  );
}

function CvCard({ label, value }: { label: string; value?: number | null }) {
  const has = typeof value === "number" && Number.isFinite(value);
  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50/70 px-3 py-2">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">{label}</p>
      {has ? (
        <p className="mt-0.5 text-base font-semibold text-gray-800">{(value as number).toFixed(2)}%</p>
      ) : (
        <p className="mt-0.5 text-xs italic text-amber-600">pending backend update</p>
      )}
    </div>
  );
}
