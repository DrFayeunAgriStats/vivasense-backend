import React from "react";
import type { AnovaTable } from "@/services/geneticsUploadApi";

/**
 * Factorial-specific results presentation. Renders three separate sections
 * (Factor A main effects, Factor B main effects, A × B interaction effects)
 * with header badges and an interaction-significance banner.
 *
 * Backend fields used (all optional / defensive):
 *   - traitResult.analysis_result.result.anova_table  (existing)
 *   - traitResult.analysis_result.result.factor_a_means     [ {level, mean, se, tukey_group, lsd_group} ]
 *   - traitResult.analysis_result.result.factor_b_means     [ {level, mean, se, tukey_group, lsd_group} ]
 *   - traitResult.analysis_result.result.interaction_means  [ {factor_a, factor_b, mean, se} ]
 */

interface FactorRow {
  level?: string;
  mean?: number;
  se?: number | null;
  tukey_group?: string;
  lsd_group?: string;
}
interface InteractionRow {
  factor_a?: string;
  factor_b?: string;
  mean?: number;
  se?: number | null;
}

function findInteractionP(at?: AnovaTable | null): number | null {
  if (!at) return null;
  for (let i = 0; i < at.source.length; i++) {
    const s = String(at.source[i] ?? "").toLowerCase();
    if (s.includes(":") || s.includes("×") || s.includes("x ") || s.includes(" x ")) {
      const p = at.p_value[i];
      if (typeof p === "number") return p;
    }
  }
  return null;
}

export function FactorialResults({
  anovaTable,
  factorAMeans,
  factorBMeans,
  interactionMeans,
}: {
  anovaTable?: AnovaTable | null;
  factorAMeans?: FactorRow[] | null;
  factorBMeans?: FactorRow[] | null;
  interactionMeans?: InteractionRow[] | null;
}) {
  const pInter = findInteractionP(anovaTable);

  let interactionBadge: { text: string; cls: string };
  if (pInter == null) {
    interactionBadge = { text: "Interaction p-value pending backend update", cls: "bg-gray-100 text-gray-600 border-gray-200" };
  } else if (pInter < 0.01) {
    interactionBadge = { text: `Interaction significant (p < 0.01)`, cls: "bg-red-100 text-red-700 border-red-200" };
  } else if (pInter < 0.05) {
    interactionBadge = { text: `Interaction significant (p < 0.05)`, cls: "bg-amber-100 text-amber-700 border-amber-200" };
  } else {
    interactionBadge = { text: "Interaction non-significant", cls: "bg-emerald-100 text-emerald-700 border-emerald-200" };
  }

  const showAdvisory = pInter != null && pInter < 0.05;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700">
          Factorial Design
        </span>
        <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${interactionBadge.cls}`}>
          {interactionBadge.text}
        </span>
      </div>

      {showAdvisory && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2.5 text-xs leading-relaxed text-amber-800">
          ⚠️ <strong>Significant interaction detected.</strong> Main effects should be interpreted
          cautiously — the response to Factor A depends on the level of Factor B. Interpret simple
          effects, not main effects alone.
        </div>
      )}

      <FactorSection
        title="Factor A — Main Effects"
        badge="Factor A"
        rows={factorAMeans}
        empty="Factor A means not provided by backend (pending)."
      />
      <FactorSection
        title="Factor B — Main Effects"
        badge="Factor B"
        rows={factorBMeans}
        empty="Factor B means not provided by backend (pending)."
      />
      <InteractionSection rows={interactionMeans} />
    </div>
  );
}

function FactorSection({
  title,
  badge,
  rows,
  empty,
}: {
  title: string;
  badge: string;
  rows?: FactorRow[] | null;
  empty: string;
}) {
  return (
    <section className="rounded-lg border border-gray-200 bg-white">
      <header className="flex items-center justify-between border-b border-gray-100 px-3 py-2">
        <h5 className="text-sm font-semibold text-gray-800">{title}</h5>
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
          {badge}
        </span>
      </header>
      {rows && rows.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="px-3 py-2 text-left font-semibold">Level</th>
                <th className="px-3 py-2 text-left font-semibold">Mean</th>
                <th className="px-3 py-2 text-left font-semibold">SE</th>
                <th className="px-3 py-2 text-left font-semibold">Tukey Group</th>
                <th className="px-3 py-2 text-left font-semibold">LSD Group</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.level}-${i}`} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
                  <td className="px-3 py-1.5 font-medium text-gray-700">{r.level ?? "—"}</td>
                  <td className="px-3 py-1.5 text-gray-700">{typeof r.mean === "number" ? r.mean.toFixed(2) : "—"}</td>
                  <td className="px-3 py-1.5 text-gray-600">{typeof r.se === "number" ? r.se.toFixed(2) : "—"}</td>
                  <td className="px-3 py-1.5"><GroupChip g={r.tukey_group} /></td>
                  <td className="px-3 py-1.5"><GroupChip g={r.lsd_group} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="px-3 py-3 text-xs italic text-gray-400">{empty}</p>
      )}
    </section>
  );
}

function InteractionSection({ rows }: { rows?: InteractionRow[] | null }) {
  return (
    <section className="rounded-lg border border-gray-200 bg-white">
      <header className="flex items-center justify-between border-b border-gray-100 px-3 py-2">
        <h5 className="text-sm font-semibold text-gray-800">A × B Interaction Effects</h5>
        <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-0.5 text-[11px] font-medium text-violet-700">
          A × B Interaction
        </span>
      </header>
      {rows && rows.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="px-3 py-2 text-left font-semibold">Factor A</th>
                <th className="px-3 py-2 text-left font-semibold">Factor B</th>
                <th className="px-3 py-2 text-left font-semibold">Mean</th>
                <th className="px-3 py-2 text-left font-semibold">SE</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
                  <td className="px-3 py-1.5 font-medium text-gray-700">{r.factor_a ?? "—"}</td>
                  <td className="px-3 py-1.5 font-medium text-gray-700">{r.factor_b ?? "—"}</td>
                  <td className="px-3 py-1.5 text-gray-700">{typeof r.mean === "number" ? r.mean.toFixed(2) : "—"}</td>
                  <td className="px-3 py-1.5 text-gray-600">{typeof r.se === "number" ? r.se.toFixed(2) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="px-3 py-3 text-xs italic text-gray-400">
          Interaction means not provided by backend (pending).
        </p>
      )}
    </section>
  );
}

function GroupChip({ g }: { g?: string }) {
  if (!g) return <span className="text-gray-300">—</span>;
  return (
    <span className="inline-block rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-gray-700">
      {g}
    </span>
  );
}
