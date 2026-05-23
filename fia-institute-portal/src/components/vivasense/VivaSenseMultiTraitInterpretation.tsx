import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { Sparkles, Loader2, AlertCircle, CheckCircle2, ChevronDown, Download } from "lucide-react";
import { downloadResultsAsText } from "@/lib/downloadResults";
import ReactMarkdown from "react-markdown";
import { MultiTraitResultsData } from "./VivaSenseMultiTraitResults";

import { GENETICS_API_BASE as BACKEND_BASE } from "@/config/vivasense";

interface Props {
  designLabel: string;
  results: MultiTraitResultsData;
  onTraitInterpretations: (interps: Record<string, string>) => void;
  onTraitProgress: (completed: Set<string>) => void;
  onTraitGroundingChecks?: (checks: Record<string, GroundingCheck>) => void;
  onSynthesisGroundingCheck?: (check: GroundingCheck | undefined) => void;
}

function parseSSEChunk(line: string): string {
  if (!line.startsWith("data: ")) return "";
  const jsonStr = line.slice(6).trim();
  if (jsonStr === "[DONE]") return "";
  try {
    const parsed = JSON.parse(jsonStr);
    if (parsed.type === "content_block_delta" && parsed.delta?.text) return parsed.delta.text;
    if (parsed.choices?.[0]?.delta?.content) return parsed.choices[0].delta.content;
    if (parsed.type && parsed.type !== "content_block_delta") return "";
    return parsed.content || parsed.text || "";
  } catch {
    return jsonStr;
  }
}

export interface GroundingCheck {
  passed: boolean;
  warning_count: number;
  warnings: string[];
}

interface InterpretationResult {
  text: string;
  grounding_check?: GroundingCheck;
}

async function fetchInterpretation(
  analysisType: string,
  resultsPayload: Record<string, unknown>,
  signal?: AbortSignal
): Promise<InterpretationResult> {
  const resp = await fetch(`${BACKEND_BASE}/api/interpret`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_type: analysisType, results: resultsPayload }),
    signal,
  });

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `Server error (${resp.status})`);
  }

  if (!resp.body) {
    const data = await resp.json();
    return {
      text: data.interpretation || data.response || JSON.stringify(data),
      grounding_check: data.grounding_check,
    };
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";
  let isSSE: boolean | null = null;
  let groundingCheck: GroundingCheck | undefined;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    if (isSSE === null) isSSE = buffer.includes("data: ");

    if (isSSE) {
      let idx: number;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx).replace(/\r$/, "");
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        // Check for grounding_check in SSE events
        if (line.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(line.slice(6).trim());
            if (parsed.grounding_check) groundingCheck = parsed.grounding_check;
          } catch {}
        }
        const c = parseSSEChunk(line);
        if (c) full += c;
      }
    } else {
      full += buffer;
      buffer = "";
    }
  }

  if (buffer.trim()) {
    if (isSSE) {
      try {
        const parsed = JSON.parse(buffer.trim().replace(/^data: /, ""));
        if (parsed.grounding_check) groundingCheck = parsed.grounding_check;
      } catch {}
      const c = parseSSEChunk(buffer.trim());
      if (c) full += c;
    } else {
      // Try to extract grounding_check from non-SSE JSON
      try {
        const parsed = JSON.parse(full + buffer);
        if (parsed.grounding_check) groundingCheck = parsed.grounding_check;
        full = parsed.interpretation || parsed.response || full + buffer;
      } catch {
        full += buffer;
      }
    }
  }

  return { text: full || "No interpretation returned.", grounding_check: groundingCheck };
}

export function VivaSenseMultiTraitInterpretation({
  designLabel,
  results,
  onTraitInterpretations,
  onTraitProgress,
  onTraitGroundingChecks,
  onSynthesisGroundingCheck,
}: Props) {
  const traitData = results.per_trait || results.trait_results || {};
  const traitNames = Object.keys(traitData);
  const totalTraits = traitNames.length;

  const [currentTraitIndex, setCurrentTraitIndex] = useState(-1);
  const [completedTraits, setCompletedTraits] = useState<Set<string>>(new Set());
  const [traitInterps, setTraitInterps] = useState<Record<string, string>>({});
  const [synthesis, setSynthesis] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "traits" | "synthesis" | "done">("idle");
  const [synthGrounding, setSynthGrounding] = useState<GroundingCheck | undefined>();

  const runAll = useCallback(async (signal?: AbortSignal) => {
    setIsRunning(true);
    setError(null);
    setCompletedTraits(new Set());
    setTraitInterps({});
    setSynthesis("");
    setPhase("traits");

    const newInterps: Record<string, string> = {};
    const newCompleted = new Set<string>();
    const newGroundingChecks: Record<string, GroundingCheck> = {};

    try {
      // Sequential per-trait interpretation
      for (let i = 0; i < traitNames.length; i++) {
        if (signal?.aborted) return;
        const name = traitNames[i];
        setCurrentTraitIndex(i);

        const traitPayload = traitData[name] as Record<string, unknown>;

        const interpResult = await fetchInterpretation(
          `${designLabel} — ${name}`,
          traitPayload,
          signal
        );

        newInterps[name] = interpResult.text;
        if (interpResult.grounding_check) {
          newGroundingChecks[name] = interpResult.grounding_check;
        }
        newCompleted.add(name);

        setTraitInterps({ ...newInterps });
        setCompletedTraits(new Set(newCompleted));
        onTraitInterpretations({ ...newInterps });
        onTraitProgress(new Set(newCompleted));
        onTraitGroundingChecks?.({ ...newGroundingChecks });
      }

      // Synthesis call
      if (signal?.aborted) return;
      setPhase("synthesis");
      setIsSynthesizing(true);

      // Build enriched synthesis payload (no raw ANOVA / full Tukey tables)
      const summaryTable = results.summary_table;

      // Extract top correlations (|r| > 0.5) from correlation matrix in meta
      const topCorrelations: { trait1: string; trait2: string; r: number }[] = [];
      const corrMatrix = (results.meta as any)?.correlation_matrix;
      if (corrMatrix && typeof corrMatrix === "object") {
        const corrTraits = Object.keys(corrMatrix);
        for (let i = 0; i < corrTraits.length; i++) {
          for (let j = i + 1; j < corrTraits.length; j++) {
            const r = Number(corrMatrix[corrTraits[i]]?.[corrTraits[j]]);
            if (!isNaN(r) && Math.abs(r) > 0.5) {
              topCorrelations.push({ trait1: corrTraits[i], trait2: corrTraits[j], r: Math.round(r * 1000) / 1000 });
            }
          }
        }
        topCorrelations.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
      }

      // Extract PCA summary — PC1 variance and top 3 loadings
      let pcaSummary: Record<string, unknown> | null = null;
      const pcaData = (results.meta as any)?.pca;
      if (pcaData) {
        const pc1Var = pcaData.variance_explained?.[0] ?? pcaData.pc1_variance;
        const loadings = pcaData.loadings?.PC1 || pcaData.loadings?.[0];
        let topLoadings: { trait: string; loading: number }[] = [];
        if (loadings && typeof loadings === "object") {
          topLoadings = Object.entries(loadings)
            .map(([trait, val]) => ({ trait, loading: Math.round(Number(val) * 1000) / 1000 }))
            .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
            .slice(0, 3);
        }
        pcaSummary = { pc1_variance_explained: pc1Var, top_loadings_pc1: topLoadings };
      }

      // Per-trait best performer (highest mean + Tukey letter)
      const bestPerformers: Record<string, { genotype: string; mean: number; tukey_letter: string }> = {};
      for (const tName of traitNames) {
        const trait = traitData[tName];
        const meansTable = trait.tables?.means || trait.tables?.group_means || trait.tables?.treatment_means;
        if (Array.isArray(meansTable) && meansTable.length > 1 && Array.isArray(meansTable[0])) {
          const headers = (meansTable[0] as string[]).map((h: string) => String(h).toLowerCase());
          const genoIdx = headers.findIndex((h: string) => h.includes("treat") || h.includes("genotype") || h.includes("variety") || h === headers[0]);
          const meanIdx = headers.findIndex((h: string) => h.includes("mean"));
          const tukeyIdx = headers.findIndex((h: string) => h.includes("tukey") || h.includes("group") || h.includes("letter"));
          if (genoIdx >= 0 && meanIdx >= 0) {
            let best = { genotype: "", mean: -Infinity, tukey_letter: "" };
            for (let row = 1; row < meansTable.length; row++) {
              const r = meansTable[row] as unknown[];
              const m = Number(r[meanIdx]);
              if (m > best.mean) {
                best = { genotype: String(r[genoIdx]), mean: Math.round(m * 100) / 100, tukey_letter: tukeyIdx >= 0 ? String(r[tukeyIdx]) : "" };
              }
            }
            if (best.genotype) bestPerformers[tName] = best;
          }
        }
      }

      // Extract assumption diagnostics per trait
      const assumptionsSummary: { trait: string; normality_passed: boolean; normality_p?: number; homogeneity_passed: boolean; homogeneity_p?: number }[] = [];
      for (const tName of traitNames) {
        const trait = traitData[tName];
        const assumptions = trait.tables?.assumptions || trait.tables?.assumption_tests || trait.assumptions;
        let normality_passed = true;
        let normality_p: number | undefined;
        let homogeneity_passed = true;
        let homogeneity_p: number | undefined;
        if (Array.isArray(assumptions) && assumptions.length > 1 && Array.isArray(assumptions[0])) {
          const headers = (assumptions[0] as string[]).map((h: string) => String(h).toLowerCase());
          const testIdx = headers.findIndex((h: string) => h.includes("test") || h.includes("name"));
          const passIdx = headers.findIndex((h: string) => h.includes("pass") || h.includes("result") || h.includes("normal"));
          const pIdx = headers.findIndex((h: string) => h === "p" || h.includes("p-value") || h.includes("p_value"));
          for (let row = 1; row < assumptions.length; row++) {
            const r = assumptions[row] as unknown[];
            const testName = String(r[testIdx] ?? "").toLowerCase();
              const pVal = pIdx >= 0 ? Number(r[pIdx]) : undefined;
              if (passIdx >= 0) {
                const passed = String(r[passIdx]).toLowerCase();
                const isPassed = passed === "true" || passed === "pass" || passed === "yes";
                if (testName.includes("shapiro") || testName.includes("normality")) { normality_passed = isPassed; if (pVal !== undefined && !isNaN(pVal)) normality_p = pVal; }
                if (testName.includes("levene") || testName.includes("homogeneity") || testName.includes("bartlett")) { homogeneity_passed = isPassed; if (pVal !== undefined && !isNaN(pVal)) homogeneity_p = pVal; }
              } else if (pVal !== undefined && !isNaN(pVal)) {
                if (testName.includes("shapiro") || testName.includes("normality")) { normality_passed = pVal >= 0.05; normality_p = pVal; }
                if (testName.includes("levene") || testName.includes("homogeneity") || testName.includes("bartlett")) { homogeneity_passed = pVal >= 0.05; homogeneity_p = pVal; }
              }
          }
        } else if (assumptions && typeof assumptions === "object" && !Array.isArray(assumptions)) {
          const a = assumptions as Record<string, unknown>;
          if (a.normality_passed !== undefined) normality_passed = Boolean(a.normality_passed);
          if (a.homogeneity_passed !== undefined) homogeneity_passed = Boolean(a.homogeneity_passed);
          if (a.shapiro_p !== undefined) { normality_p = Number(a.shapiro_p); normality_passed = normality_p >= 0.05; }
          if (a.levene_p !== undefined) { homogeneity_p = Number(a.levene_p); homogeneity_passed = homogeneity_p >= 0.05; }
        }
        const entry: typeof assumptionsSummary[number] = { trait: tName, normality_passed, homogeneity_passed };
        if (normality_p !== undefined) entry.normality_p = Math.round(normality_p * 10000) / 10000;
        if (homogeneity_p !== undefined) entry.homogeneity_p = Math.round(homogeneity_p * 10000) / 10000;
        assumptionsSummary.push(entry);
      }

      const synthPayload: Record<string, unknown> = {
        ...results,
        trait_count: traitNames.length,
        trait_names: traitNames,
        top_correlations: topCorrelations.length > 0 ? topCorrelations : undefined,
        pca_summary: pcaSummary || undefined,
        best_combination: Object.keys(bestPerformers).length > 0 ? bestPerformers : undefined,
        assumptions_summary: assumptionsSummary,
        tukey_already_shown: true,
      };

      const synthResult = await fetchInterpretation(
        `Multi-trait synthesis — ${designLabel}`,
        synthPayload,
        signal
      );

      setSynthesis(synthResult.text);
      setSynthGrounding(synthResult.grounding_check);
      onSynthesisGroundingCheck?.(synthResult.grounding_check);
      setPhase("done");
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "Interpretation failed.");
      }
    } finally {
      setIsRunning(false);
      setIsSynthesizing(false);
    }
  }, [traitNames, traitData, results, designLabel, onTraitInterpretations, onTraitProgress]);

  // Auto-run on mount
  useEffect(() => {
    if (totalTraits === 0) return;
    const controller = new AbortController();
    runAll(controller.signal);
    return () => controller.abort();
  }, []);

  const progressPercent = totalTraits > 0
    ? Math.round(((completedTraits.size + (isSynthesizing ? 0.5 : 0)) / (totalTraits + 1)) * 100)
    : 0;

  const currentTraitName = currentTraitIndex >= 0 && currentTraitIndex < traitNames.length
    ? traitNames[currentTraitIndex]
    : "";

  return (
    <div className="space-y-6">
      {/* Synthesis / Dr. Fayeun Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-primary" />
            Dr. Fayeun's Multi-Trait Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Progress indicator */}
          {(phase === "traits" || phase === "synthesis") && (
            <div className="space-y-3 mb-6">
              <Progress value={progressPercent} className="h-2" />
              <p className="text-sm text-muted-foreground">
                {phase === "traits" && currentTraitName && (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin inline mr-1.5" />
                    Interpreting trait {currentTraitIndex + 1} of {totalTraits}:{" "}
                    <span className="font-medium text-foreground">{currentTraitName}</span>…
                  </>
                )}
                {phase === "synthesis" && (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin inline mr-1.5" />
                    Generating multi-trait synthesis…
                  </>
                )}
              </p>
              {/* Completed trait checklist */}
              {completedTraits.size > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {traitNames.map((name) => (
                    <span
                      key={name}
                      className={`inline-flex items-center gap-1 text-xs rounded-full px-2.5 py-1 ${
                        completedTraits.has(name)
                          ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {completedTraits.has(name) && <CheckCircle2 className="w-3 h-3" />}
                      {name}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-3 text-destructive text-sm bg-destructive/10 rounded-lg p-4">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{error}</span>
              </div>
              <Button variant="outline" size="sm" onClick={() => runAll()} className="self-start">
                <Sparkles className="w-4 h-4 mr-2" />
                Retry Interpretations
              </Button>
            </div>
          )}

          {synthesis && (
            <>
              {/* Synthesis grounding check warning */}
              {synthGrounding && !synthGrounding.passed && (
                <Collapsible defaultOpen className="mb-4">
                  <div className="rounded-lg border border-yellow-400 bg-yellow-50 dark:bg-yellow-950/30 dark:border-yellow-600 p-4">
                    <CollapsibleTrigger className="w-full flex items-center justify-between text-sm font-semibold text-yellow-800 dark:text-yellow-200">
                      <span className="flex items-center gap-2">
                        <AlertCircle className="w-4 h-4" />
                        Statistical Review Alerts ({synthGrounding.warning_count} warning{synthGrounding.warning_count !== 1 ? "s" : ""})
                      </span>
                      <ChevronDown className="w-4 h-4 transition-transform duration-200 [[data-state=open]>&]:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <ul className="list-disc list-inside space-y-1 mt-2">
                        {synthGrounding.warnings.map((w, i) => (
                          <li key={i} className="text-sm text-yellow-700 dark:text-yellow-300">{w}</li>
                        ))}
                      </ul>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              )}
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown>{synthesis}</ReactMarkdown>
              </div>
              <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  ⚠️ <strong>Academic Integrity Reminder:</strong> This synthesis is a starting point. Verify all numbers against the tables above. Adapt text to your own words and field context.
                </p>
              </div>
            </>
          )}

          {phase === "done" && !synthesis && !error && (
            <Button variant="outline" size="sm" onClick={() => runAll()} className="self-start">
              <Sparkles className="w-4 h-4 mr-2" />
              Regenerate Interpretations
            </Button>
          )}

          {phase === "done" && synthesis && (
            <div className="mt-6 pt-4 border-t border-border">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  try {
                    const combinedInterpretation = [
                      ...traitNames.map(n => traitInterps[n] ? `--- ${n} ---\n${traitInterps[n]}` : "").filter(Boolean),
                      synthesis ? `--- SYNTHESIS ---\n${synthesis}` : "",
                    ].filter(Boolean).join("\n\n");
                    downloadResultsAsText(
                      `Multi-trait ${designLabel}`,
                      results as any,
                      combinedInterpretation,
                    );
                  } catch (err) {
                    console.error("Download failed:", err);
                  }
                }}
              >
                <Download className="w-4 h-4 mr-2" />
                Download Results
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
