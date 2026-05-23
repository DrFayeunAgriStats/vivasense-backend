import React, { useState } from "react";

type DesignKind = "crd" | "rcbd" | "factorial" | "split_plot_rcbd";

const TABLES: Record<DesignKind, { title: string; rows: [string, string][] }> = {
  crd: {
    title: "CRD — Expected ANOVA Structure",
    rows: [
      ["Treatment", "t − 1"],
      ["Error", "t(r − 1)"],
      ["Total", "tr − 1"],
    ],
  },
  rcbd: {
    title: "RCBD — Expected ANOVA Structure",
    rows: [
      ["Replication", "r − 1"],
      ["Treatment", "t − 1"],
      ["Error", "(r − 1)(t − 1)"],
      ["Total", "rt − 1"],
    ],
  },
  factorial: {
    title: "Factorial — Expected ANOVA Structure",
    rows: [
      ["Factor A", "a − 1"],
      ["Factor B", "b − 1"],
      ["A × B", "(a − 1)(b − 1)"],
      ["Error", "ab(r − 1)"],
      ["Total", "abr − 1"],
    ],
  },
  split_plot_rcbd: {
    title: "Split-Plot — Expected ANOVA Structure",
    rows: [
      ["Replication", "r − 1"],
      ["Main Plot (A)", "a − 1"],
      ["Error A (Rep × A)", "(r − 1)(a − 1)"],
      ["Sub Plot (B)", "b − 1"],
      ["A × B", "(a − 1)(b − 1)"],
      ["Error B", "a(r − 1)(b − 1)"],
      ["Total", "rab − 1"],
    ],
  },
};

const LEGEND =
  "a = levels of Factor A (or Main Plot)\nb = levels of Factor B (or Sub Plot)\nr = number of replications\nt = number of treatments";

export function AnovaStructurePreview({ design }: { design: DesignKind }) {
  const [open, setOpen] = useState(false);
  const meta = TABLES[design];

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <span className="text-xs font-medium text-gray-600">
          {open ? "Hide" : "Show"} expected ANOVA structure
        </span>
        <span className={`text-xs text-gray-400 transition-transform ${open ? "rotate-90" : ""}`}>▶</span>
      </button>
      {open && (
        <div className="border-t border-gray-200 px-3 py-3">
          <div className="mb-2 flex items-center gap-1.5">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">
              {meta.title}
            </p>
            <span
              className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-gray-300 text-[10px] text-gray-500 cursor-help"
              title={LEGEND}
              aria-label="ANOVA notation legend"
            >
              i
            </span>
          </div>
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200 text-left text-gray-500">
                <th className="py-1.5 pr-4 font-medium">Source</th>
                <th className="py-1.5 font-medium">df</th>
              </tr>
            </thead>
            <tbody>
              {meta.rows.map(([src, df]) => (
                <tr key={src} className="border-b border-gray-100 last:border-0">
                  <td className="py-1.5 pr-4 text-gray-700">{src}</td>
                  <td className="py-1.5 font-mono text-gray-600">{df}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-gray-400">
            Informational only — actual ANOVA reflects your dataset.
          </p>
        </div>
      )}
    </div>
  );
}
