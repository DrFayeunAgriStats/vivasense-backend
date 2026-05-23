/**
 * AcademicInterpretationPanel
 * ============================
 * Option A: renders only when the user explicitly requests interpretation.
 *
 * Props:
 *   traitName        — used in the section header
 *   moduleType       — "anova" | "genetic_parameters" | "correlation" | "heatmap"
 *   analysisResult   — raw result object from the analysis response
 *   cropContext      — optional crop name for context (e.g. "cowpea")
 *
 * The parent is responsible for showing/hiding this panel via a button.
 * The panel fetches on mount (once) and renders a structured output.
 */

import React, { useEffect, useRef, useState } from "react";
import { VsSpinner } from "./VsSpinner";
import {
  getAcademicInterpretation,
  AcademicInterpretationResponse,
  AcademicModuleType,
  GuidedWritingBlock,
  ValidationResult,
} from "@/services/academicApi";

const WRITING_SUPPORT_SESSION_KEY = "vivasense:academic-interpretation:writing-support:open";
const VALIDATION_REPORT_SESSION_KEY = "vivasense:academic-interpretation:validation-report:open";

// ─────────────────────────────────────────────────────────────────────────────
// MAIN PANEL
// ─────────────────────────────────────────────────────────────────────────────

interface AcademicInterpretationPanelProps {
  traitName: string;
  moduleType: AcademicModuleType;
  analysisResult: Record<string, unknown>;
  cropContext?: string;
  onClose: () => void;
}

export function AcademicInterpretationPanel({
  traitName,
  moduleType,
  analysisResult,
  cropContext,
  onClose,
}: AcademicInterpretationPanelProps) {
  const [data, setData] = useState<AcademicInterpretationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showWriting, setShowWriting] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    try {
      return window.sessionStorage.getItem(WRITING_SUPPORT_SESSION_KEY) === "open";
    } catch {
      return false;
    }
  });
  const [showValidator, setShowValidator] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    try {
      return window.sessionStorage.getItem(VALIDATION_REPORT_SESSION_KEY) === "open";
    } catch {
      return false;
    }
  });
  const fetchedRef = useRef(false);

  const toggleWriting = () => {
    setShowWriting((current) => {
      const next = !current;
      if (typeof window !== "undefined") {
        try {
          window.sessionStorage.setItem(
            WRITING_SUPPORT_SESSION_KEY,
            next ? "open" : "closed"
          );
        } catch {
          // Ignore storage failures and keep the in-memory toggle working.
        }
      }
      return next;
    });
  };

  const toggleValidator = () => {
    setShowValidator((current) => {
      const next = !current;
      if (typeof window !== "undefined") {
        try {
          window.sessionStorage.setItem(
            VALIDATION_REPORT_SESSION_KEY,
            next ? "open" : "closed"
          );
        } catch {
          // Ignore storage failures and keep the in-memory toggle working.
        }
      }
      return next;
    });
  };

  useEffect(() => {
    // Prevent double-fetch in StrictMode
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    getAcademicInterpretation({
      module_type: moduleType,
      trait: traitName,
      analysis_result: analysisResult,
      crop_context: cropContext ?? null,
      include_writing_support: true,
    })
      .then(setData)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Unexpected error")
      )
      .finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mt-4 rounded-xl border border-violet-200 bg-violet-50/40">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-violet-200">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-violet-700">
            Academic Interpretation
          </span>
          {data && (
            <span className="text-xs text-violet-500">
              — {traitName}
              {data.fallback_used && (
                <span className="ml-2 inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700 border border-amber-200">
                  deterministic fallback
                </span>
              )}
              {data.ai_generated && !data.fallback_used && (
                <span className="ml-2 inline-flex items-center rounded-full bg-violet-100 px-2 py-0.5 text-xs text-violet-700 border border-violet-200">
                  AI generated
                </span>
              )}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-xs text-violet-400 hover:text-violet-700"
          aria-label="Close interpretation panel"
        >
          ✕ Close
        </button>
      </div>

      <div className="px-4 py-4 space-y-5">
        {/* ── Loading ─────────────────────────────────────────────────────── */}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-violet-600">
            <VsSpinner size="sm" />
            Generating academic interpretation…
          </div>
        )}

        {/* ── Error ───────────────────────────────────────────────────────── */}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-700">
            <p className="font-semibold mb-1">Interpretation unavailable</p>
            <p className="text-xs">{error}</p>
          </div>
        )}

        {/* ── Content ─────────────────────────────────────────────────────── */}
        {data && (
          <>
            {/* Overall Finding */}
            <Section title="Overall Finding">
              <p className="text-sm text-gray-700 leading-relaxed">
                {data.overall_finding}
              </p>
            </Section>

            {/* Statistical Evidence */}
            <Section title="Statistical Evidence">
              <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
                {data.statistical_evidence}
              </p>
            </Section>

            {/* Module-specific sections */}
            {Object.entries(data.module_sections).map(([name, content]) => (
              <Section key={name} title={name}>
                <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
                  {content}
                </p>
              </Section>
            ))}

            {/* Examiner Checkpoint */}
            {data.examiner_checkpoint.length > 0 && (
              <Section title="Examiner Checkpoint">
                <ul className="space-y-1.5">
                  {data.examiner_checkpoint.map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="mt-0.5 shrink-0 text-gray-400">☐</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </Section>
            )}

            {/* Scope + Closing */}
            <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-3 space-y-2">
              <p className="text-xs text-gray-600 leading-relaxed italic">
                {data.scope_statement}
              </p>
              <p className="text-xs text-gray-500">{data.closing}</p>
              <p className="text-xs text-violet-600">{data.research_writing_referral}</p>
            </div>

            {/* ── Guided Writing (collapsible) ───────────────────────────── */}
            {data.guided_writing && (
              <div>
                <button
                  type="button"
                  onClick={toggleWriting}
                  aria-expanded={showWriting}
                  className="flex items-center gap-1.5 text-xs font-semibold text-violet-700 hover:text-violet-900"
                >
                  <span className={`transition-transform ${showWriting ? "rotate-90" : ""}`}>▶</span>
                  Writing Support (sentence starters + checklist)
                </button>
                {showWriting && (
                  <div className="mt-3">
                    <GuidedWritingSection gw={data.guided_writing} />
                  </div>
                )}
              </div>
            )}

            {/* ── Validator Report (collapsible) ─────────────────────────── */}
            {data.validator_result && (
              <div>
                <button
                  type="button"
                  onClick={toggleValidator}
                  aria-expanded={showValidator}
                  className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600"
                >
                  <span className={`transition-transform ${showValidator ? "rotate-90" : ""}`}>▶</span>
                  Validation report ({data.validator_result.block_count} block
                  {data.validator_result.block_count !== 1 ? "s" : ""},{" "}
                  {data.validator_result.warning_count} warning
                  {data.validator_result.warning_count !== 1 ? "s" : ""})
                </button>
                {showValidator && (
                  <div className="mt-2">
                    <ValidatorReport vr={data.validator_result} />
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GUIDED WRITING SECTION
// ─────────────────────────────────────────────────────────────────────────────

function GuidedWritingSection({ gw }: { gw: GuidedWritingBlock }) {
  return (
    <div className="space-y-4 rounded-lg border border-violet-200 bg-white px-4 py-4">
      {/* Caution note */}
      {gw.caution_note && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          <span className="font-semibold">Caution: </span>
          {gw.caution_note}
        </div>
      )}

      {/* Sentence starters */}
      {gw.sentence_starters.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Sentence Starters — fill in the blanks from your analysis
          </p>
          <div className="space-y-3">
            {gw.sentence_starters.map((s, i) => (
              <div key={i} className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5">
                <p className="text-xs font-medium text-violet-700 mb-1">{s.purpose}</p>
                <p className="text-sm font-mono text-gray-800 leading-relaxed">
                  {s.template}
                </p>
                {s.values_to_fill.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {s.values_to_fill.map((v, j) => (
                      <span
                        key={j}
                        className="inline-block rounded bg-violet-100 px-2 py-0.5 text-xs text-violet-700"
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                )}
                {s.hint && (
                  <p className="mt-1 text-xs text-gray-400 italic">Hint: {s.hint}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Examiner checklist */}
      {gw.examiner_checkpoint.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Before you submit — tick each item
          </p>
          <ul className="space-y-1.5">
            {gw.examiner_checkpoint.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="mt-0.5 shrink-0 text-gray-400">☐</span>
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-xs text-violet-600 italic">{gw.supervisor_prompt}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// VALIDATOR REPORT
// ─────────────────────────────────────────────────────────────────────────────

function ValidatorReport({ vr }: { vr: ValidationResult }) {
  if (vr.violations.length === 0) {
    return (
      <p className="text-xs text-emerald-600 rounded bg-emerald-50 px-3 py-2 border border-emerald-200">
        ✓ No rule violations detected.
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      {vr.violations.map((v, i) => (
        <div
          key={i}
          className={`rounded-lg border px-3 py-2 text-xs ${
            v.severity === "block"
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-amber-200 bg-amber-50 text-amber-700"
          }`}
        >
          <p className="font-semibold">{v.rule_id} ({v.severity})</p>
          {v.excerpt && (
            <p className="mt-0.5 font-mono text-gray-500 truncate">&ldquo;{v.excerpt}&rdquo;</p>
          )}
          <p className="mt-0.5">{v.message}</p>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TINY HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        {title}
      </p>
      {children}
    </div>
  );
}
