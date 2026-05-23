import { useState, useRef, useCallback, useMemo } from "react";
import Papa from "papaparse";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import {
  Upload,
  FileSpreadsheet,
  Play,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Info,
  ClipboardPaste,
  Beaker,
  ChevronDown,
  ChevronUp,
  Download,
  FileText,
  Table2,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

/* ─── Types ─── */
type DesignType = "oneway" | "oneway_rcbd" | "twoway" | "rcbd_factorial";

/** Maps UI design values to the lowercase API design key expected by the new backend */
const API_DESIGN_KEY: Record<DesignType, string> = {
  oneway: "crd",
  oneway_rcbd: "rcbd",
  twoway: "crd_factorial",
  rcbd_factorial: "rcbd_factorial",
};

interface DesignOption {
  value: DesignType;
  label: string;
  fields: string[];
}

const DESIGNS: DesignOption[] = [
  { value: "oneway", label: "CRD (One-way)", fields: ["treatment"] },
  { value: "oneway_rcbd", label: "RCBD (One-way)", fields: ["treatment", "block"] },
  { value: "twoway", label: "CRD Factorial (Two-way)", fields: ["factor_a", "factor_b"] },
  { value: "rcbd_factorial", label: "RCBD Factorial", fields: ["factor_a", "factor_b", "block"] },
];

const FIELD_LABELS: Record<string, string> = {
  treatment: "Treatment",
  block: "Block",
  factor_a: "Factor A",
  factor_b: "Factor B",
};

import { GENETICS_API_BASE } from "@/config/vivasense";

const API_BASE = GENETICS_API_BASE;

/* ─── Helpers ─── */
function sigStars(p: number | null | undefined): string {
  if (p == null || isNaN(Number(p))) return "";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

function fmtP(p: unknown): string {
  const n = Number(p);
  if (isNaN(n)) return "—";
  if (n < 0.001) return "<0.001";
  return n.toFixed(4);
}

function fmt(v: unknown, d = 3): string {
  if (v == null) return "—";
  const n = Number(v);
  if (isNaN(n)) return String(v);
  return n.toFixed(d);
}

function fmtInt(v: unknown): string {
  if (v == null) return "—";
  const n = Number(v);
  if (isNaN(n)) return String(v);
  return Math.round(n).toString();
}

/* ─── Collapsible Section ─── */
function Section({
  title,
  icon,
  defaultOpen = true,
  children,
  badge,
}: {
  title: string;
  icon?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
  badge?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            {icon}
            {title}
            {badge}
          </CardTitle>
          {open ? <ChevronUp className="w-5 h-5 text-muted-foreground" /> : <ChevronDown className="w-5 h-5 text-muted-foreground" />}
        </div>
      </CardHeader>
      {open && <CardContent>{children}</CardContent>}
    </Card>
  );
}

/* ═══════════════ MAIN PAGE ═══════════════ */
export default function VivaSenseAnova() {
  /* ─── Data state ─── */
  const [fileName, setFileName] = useState<string | null>(null);
  const [parsedData, setParsedData] = useState<Record<string, unknown>[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [pasteData, setPasteData] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  /* ─── Form state ─── */
  const [design, setDesign] = useState<DesignType | "">("");
  const [response, setResponse] = useState("");
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({});
  const [alpha] = useState(0.05);

  /* ─── Run state ─── */
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [requestId, setRequestId] = useState<string | null>(null);

  /* ─── AI Interpretation state ─── */
  const [interpretation, setInterpretation] = useState<string | null>(null);
  const [interpretLoading, setInterpretLoading] = useState(false);
  const [interpretError, setInterpretError] = useState<string | null>(null);
  const [interpretSlowMsg, setInterpretSlowMsg] = useState(false);

  const resultsRef = useRef<HTMLDivElement>(null);
  const interpretRef = useRef<HTMLDivElement>(null);

  /* ─── CSV parse ─── */
  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (res) => {
        const rows = res.data as Record<string, unknown>[];
        setParsedData(rows);
        if (rows.length > 0) setColumns(Object.keys(rows[0]));
        toast({ title: "File loaded", description: `${rows.length} rows, ${Object.keys(rows[0] ?? {}).length} columns` });
      },
      error: () => toast({ title: "Parse error", description: "Could not read CSV file", variant: "destructive" }),
    });
  }, []);

  const handlePaste = useCallback(() => {
    const trimmed = pasteData.trim();
    if (!trimmed) {
      toast({ title: "No data", description: "Please paste your data first", variant: "destructive" });
      return;
    }

    // Try JSON first
    if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
      try {
        const parsed = JSON.parse(trimmed);
        const rows = Array.isArray(parsed) ? parsed : [parsed];
        setParsedData(rows);
        if (rows.length > 0) setColumns(Object.keys(rows[0]));
        setFileName("pasted-data.json");
        toast({ title: "Data loaded", description: `${rows.length} rows parsed from JSON` });
        return;
      } catch {
        // Fall through to CSV parsing
      }
    }

    // Try CSV/TSV parsing
    const result = Papa.parse(trimmed, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
    });
    const rows = result.data as Record<string, unknown>[];
    if (rows.length > 0 && Object.keys(rows[0]).length > 1) {
      setParsedData(rows);
      setColumns(Object.keys(rows[0]));
      setFileName("pasted-data.csv");
      toast({ title: "Data loaded", description: `${rows.length} rows parsed from CSV` });
    } else {
      toast({ title: "Could not parse data", description: "Paste valid CSV (with headers) or JSON array", variant: "destructive" });
    }
  }, [pasteData]);

  /* ─── Current design config ─── */
  const designConfig = useMemo(() => DESIGNS.find((d) => d.value === design), [design]);

  /* ─── Validation ─── */
  const canSubmit = useMemo(() => {
    if (!design || !response || parsedData.length === 0) return false;
    if (!designConfig) return false;
    return designConfig.fields.every((f) => !!fieldValues[f]);
  }, [design, response, parsedData.length, designConfig, fieldValues]);

  /* ─── Submit ─── */
  const handleSubmit = useCallback(async () => {
    if (!design || !designConfig) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setRequestId(null);

    if (parsedData.length > 500) {
      toast({
        title: "Large dataset",
        description: `Dataset has ${parsedData.length} rows. Large datasets may be slow.`,
      });
    }

    // Only include columns needed for this analysis
    const neededCols = new Set<string>();
    neededCols.add(response);
    designConfig.fields.forEach((f) => {
      const val = fieldValues[f];
      if (val) neededCols.add(val);
    });
    const filteredData = parsedData.map((row) => {
      const slim: Record<string, unknown> = {};
      for (const col of neededCols) {
        if (col in row) slim[col] = row[col];
      }
      return slim;
    });

    const payload: Record<string, unknown> = {
      design: API_DESIGN_KEY[design],
      response,
      data: filteredData,
    };
    designConfig.fields.forEach((f) => {
      payload[f] = fieldValues[f];
    });

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 120_000);

      console.log(`[COMPONENT] VivaSenseAnova -> ${API_BASE}/genetics/analyze`);
      const res = await fetch(`${API_BASE}/genetics/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);

      const rid = res.headers.get("X-Request-ID");
      if (rid) setRequestId(rid);

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body?.detail;
        if (Array.isArray(detail)) {
          setError(detail.map((d: { type?: string; msg?: string }) => `[${d.type ?? "ERROR"}] ${d.msg ?? JSON.stringify(d)}`).join("\n"));
        } else if (typeof detail === "string") {
          setError(detail);
        } else {
          setError(`Server returned ${res.status}`);
        }
        return;
      }

      const data = await res.json();
      if (data.success === false || data.status === "error") {
        const errs = data.errors ?? data.error;
        if (Array.isArray(errs)) {
          setError(errs.map((e: { code?: string; message?: string }) => `[${e.code ?? "ERROR"}] ${e.message ?? JSON.stringify(e)}`).join("\n"));
        } else if (typeof errs === "string") {
          setError(errs);
        } else {
          setError("Analysis failed. Check your data and variables.");
        }
        return;
      }

      setResults(data);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 200);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request timed out after 120 seconds. Try a smaller dataset.");
      } else {
        setError(String((err as Error)?.message ?? err));
      }
    } finally {
      setLoading(false);
    }
  }, [design, designConfig, response, fieldValues, parsedData, alpha]);

  /* ─── Extract result sections ─── */
  // The API returns: anova_table, assumption_checks, descriptive_statistics, post_hoc, model_formula, model_statistics, df_summary, design_summary, n_obs, notes
  const anova = results?.anova_table ?? results?.anova ?? (results?.tables as Record<string, unknown>)?.anova ?? null;
  const assumptions = results?.assumption_checks ?? results?.assumptions ?? (results?.tables as Record<string, unknown>)?.assumptions ?? null;
  const descriptive = results?.descriptive_statistics ?? results?.descriptive_stats ?? (results?.tables as Record<string, unknown>)?.descriptive_stats ?? null;
  const notes = results?.notes ?? results?.warnings ?? null;

  // Post-hoc / Tukey data
  const tukey = results?.post_hoc ?? results?.tukey ?? results?.posthoc ?? (results?.tables as Record<string, unknown>)?.tukey
    ?? (results?.means && results?.letters ? { means: results.means, letters: results.letters, cld_available: true } : null);

  return (
    <Layout>
      <div className="min-h-screen bg-background">
        {/* ─── Hero ─── */}
        <div className="bg-primary text-primary-foreground py-12">
          <div className="container mx-auto max-w-4xl px-4">
            <div className="flex items-center gap-3 mb-3">
              <Beaker className="w-8 h-8" />
              <h1 className="font-serif text-3xl lg:text-4xl font-bold">VivaSense ANOVA</h1>
            </div>
            <p className="text-primary-foreground/80 text-lg max-w-2xl">
              Publication-ready Analysis of Variance for agricultural research. Upload your data, select variables, and get journal-quality results.
            </p>
          </div>
        </div>

        <div className="container mx-auto max-w-4xl px-4 py-10 space-y-8">
          {/* ═══ SECTION 1: DATA INPUT ═══ */}
          <Section title="Data Input" icon={<Upload className="w-5 h-5 text-primary" />}>
            <Tabs defaultValue="upload" className="w-full">
              <TabsList className="mb-4">
                <TabsTrigger value="upload" className="gap-2">
                  <FileSpreadsheet className="w-4 h-4" /> Upload CSV
                </TabsTrigger>
                <TabsTrigger value="paste" className="gap-2">
                  <ClipboardPaste className="w-4 h-4" /> Paste Data
                </TabsTrigger>
              </TabsList>

              <TabsContent value="upload">
                <div
                  className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors"
                  onClick={() => fileRef.current?.click()}
                >
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                  <FileSpreadsheet className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
                  {fileName ? (
                    <div>
                      <p className="font-medium text-foreground">{fileName}</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {parsedData.length} rows · {columns.length} columns
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="font-medium text-foreground">Click to upload CSV</p>
                      <p className="text-sm text-muted-foreground mt-1">Accepts .csv files only</p>
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="paste">
                <Textarea
                  placeholder={'Paste CSV or JSON data:\n\nCSV example:\nGenotype,Yield,Block\nA,120,B1\nB,135,B1\n\nJSON example:\n[{"Genotype": "A", "Yield": 120}, ...]'}
                  className="font-mono text-sm min-h-[120px]"
                  value={pasteData}
                  onChange={(e) => setPasteData(e.target.value)}
                />
                <Button className="mt-3" size="sm" onClick={handlePaste} disabled={!pasteData.trim()}>
                  Parse Data
                </Button>
              </TabsContent>
            </Tabs>

            {columns.length > 0 && (
              <div className="mt-4 p-3 rounded-lg bg-muted/40 border border-border">
                <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-2">Detected Columns</p>
                <div className="flex flex-wrap gap-1.5">
                  {columns.map((c) => (
                    <Badge key={c} variant="secondary" className="text-xs font-mono">{c}</Badge>
                  ))}
                </div>
              </div>
            )}
          </Section>

          {/* ═══ SECTION 2: VARIABLE SELECTION ═══ */}
          {columns.length > 0 && (
            <Section title="Variable Selection" icon={<Beaker className="w-5 h-5 text-primary" />}>
              <div className="grid gap-5">
                {/* Design */}
                <div>
                  <label className="text-sm font-medium text-foreground block mb-1.5">Design Type</label>
                  <Select value={design} onValueChange={(v) => { setDesign(v as DesignType); setFieldValues({}); }}>
                    <SelectTrigger><SelectValue placeholder="Select experimental design" /></SelectTrigger>
                    <SelectContent>
                      {DESIGNS.map((d) => (
                        <SelectItem key={d.value} value={d.value}>{d.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Response */}
                <div>
                  <label className="text-sm font-medium text-foreground block mb-1.5">Response Variable</label>
                  <Select value={response} onValueChange={setResponse}>
                    <SelectTrigger><SelectValue placeholder="Select response (numeric)" /></SelectTrigger>
                    <SelectContent>
                      {columns.map((c) => (
                        <SelectItem key={c} value={c}>{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Conditional fields */}
                {designConfig?.fields.map((f) => (
                  <div key={f}>
                    <label className="text-sm font-medium text-foreground block mb-1.5">{FIELD_LABELS[f]}</label>
                    <Select
                      value={fieldValues[f] ?? ""}
                      onValueChange={(v) => setFieldValues((prev) => ({ ...prev, [f]: v }))}
                    >
                      <SelectTrigger><SelectValue placeholder={`Select ${FIELD_LABELS[f].toLowerCase()}`} /></SelectTrigger>
                      <SelectContent>
                        {columns.map((c) => (
                          <SelectItem key={c} value={c}>{c}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* ═══ SECTION 3: RUN BUTTON ═══ */}
          {columns.length > 0 && (
            <div className="flex items-center gap-4">
              <Button
                size="lg"
                disabled={!canSubmit || loading}
                onClick={handleSubmit}
                className="gap-2"
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                {loading ? "Running Analysis…" : "Run ANOVA"}
              </Button>
              {parsedData.length > 500 && (
                <p className="text-sm text-amber-600 flex items-center gap-1">
                  <AlertTriangle className="w-4 h-4" />
                  {parsedData.length} rows — large datasets may be slow.
                </p>
              )}
            </div>
          )}

          {/* ═══ ERROR ═══ */}
          {error && (
            <Card className="border-destructive/50">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <XCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
                  <pre className="text-sm text-destructive whitespace-pre-wrap font-mono">{error}</pre>
                </div>
              </CardContent>
            </Card>
          )}

          {/* ═══ RESULTS ═══ */}
          <div ref={resultsRef}>
            {results && (
              <div className="space-y-6">
                {/* Request ID */}
                {requestId && (
                  <p className="text-xs text-muted-foreground font-mono">Request ID: {requestId}</p>
                )}

                {/* SECTION 1: Model Summary */}
                <ModelSummaryCard
                  data={results}
                  designLabel={designConfig?.label ?? String(design)}
                  responseName={response}
                  nObs={parsedData.length}
                />

                {/* SECTION 2: ANOVA Table */}
                {anova && <AnovaTableCard data={anova} />}

                {/* SECTION 3: Assumption Checks */}
                {assumptions && <AssumptionChecksCard data={assumptions as Record<string, unknown>} />}

                {/* SECTION 4: Table of Means with Tukey Letters */}
                {tukey && <TukeyMeansCard data={tukey as Record<string, unknown>} designType={design as DesignType} />}

                {/* SECTION 5: Post-hoc Pairwise Comparisons */}
                {tukey && <PairwiseCard data={tukey as Record<string, unknown>} designType={design as DesignType} />}

                {/* SECTION 6: Descriptive Statistics */}
                {descriptive && <DescriptiveCard data={descriptive} designType={design as DesignType} />}

                {/* SECTION 7: Notes and Warnings */}
                {notes && <NotesCard data={notes} />}

                {/* SECTION 8: Charts */}
                <ChartsSection
                  results={results}
                  designType={design as DesignType}
                  responseName={response}
                  fieldValues={fieldValues}
                />

                {/* BUTTON ROW */}
                <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
                  <Button
                    variant="outline"
                    className="gap-2 border-primary text-primary hover:bg-primary hover:text-primary-foreground"
                    disabled={interpretLoading}
                    onClick={async () => {
                      if (!results) return;
                      setInterpretLoading(true);
                      setInterpretError(null);
                      setInterpretation(null);
                      setInterpretSlowMsg(false);

                      const slowTimer = setTimeout(() => setInterpretSlowMsg(true), 8000);

                      try {
                        const controller = new AbortController();
                        const timeout = setTimeout(() => controller.abort(), 45000);

                        console.log(`[COMPONENT] VivaSenseAnova -> ${API_BASE}/genetics/analyze`);
                        const resp = await fetch(`${API_BASE}/genetics/analyze`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ result: results }),
                          signal: controller.signal,
                        });
                        clearTimeout(timeout);

                        if (!resp.ok) {
                          const errBody = await resp.json().catch(() => ({}));
                          throw new Error(errBody.detail || errBody.error || `Server error (${resp.status})`);
                        }

                        const data = await resp.json();
                        setInterpretation(data.interpretation || JSON.stringify(data));

                        // Scroll to interpretation card after render
                        setTimeout(() => {
                          interpretRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                        }, 100);
                      } catch (err: unknown) {
                        const msg = err instanceof Error ? err.message : String(err);
                        setInterpretError(msg.includes("aborted") ? "Request timed out after 45 seconds. Please try again." : msg);
                      } finally {
                        clearTimeout(slowTimer);
                        setInterpretLoading(false);
                        setInterpretSlowMsg(false);
                      }
                    }}
                  >
                    {interpretLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Beaker className="w-4 h-4" />}
                    {interpretLoading ? "Generating…" : "Generate AI Interpretation"}
                  </Button>
                  <ExportWordButton results={results} design={design as DesignType} response={response} />
                  <Button
                    variant="outline"
                    className="gap-2"
                    onClick={() => {
                      setResults(null);
                      setError(null);
                      setInterpretation(null);
                      setInterpretError(null);
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                  >
                    New Analysis
                  </Button>
                </div>

                {interpretSlowMsg && interpretLoading && (
                  <p className="text-center text-sm text-muted-foreground animate-pulse">
                    Still generating… this may take up to 45 seconds.
                  </p>
                )}

                {/* AI Interpretation Error */}
                {interpretError && (
                  <Card className="border-destructive/50 bg-destructive/5">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <XCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
                        <div>
                          <p className="font-semibold text-destructive">AI interpretation unavailable. Please try again.</p>
                          {interpretError && <p className="text-sm text-destructive/80 mt-1">{interpretError}</p>}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* AI Interpretation Card */}
                {interpretation && (
                  <div ref={interpretRef}>
                    <Card className="border-l-4 border-l-primary overflow-hidden">
                      <CardHeader className="pb-2">
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <Beaker className="w-5 h-5 text-primary" />
                          AI Interpretation
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="prose prose-sm max-w-none dark:prose-invert">
                          {interpretation.split("\n\n").map((para, i) => (
                            <p key={i}>{para}</p>
                          ))}
                        </div>
                        <div className="border-t border-border pt-3 space-y-1">
                          <p className="text-sm italic text-muted-foreground">
                            AI-generated academic guidance. Verify all statements and discuss with your supervisor before submitting.
                          </p>
                          <p className="text-sm italic text-muted-foreground">
                            — Dr. Fayeun, VivaSense Academic Mentor
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}

/* ═══════════════ SECTION WRAPPER WITH GREEN ACCENT ═══════════════ */

function ResultSection({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card className="overflow-hidden">
      <div
        className="flex items-center gap-0 cursor-pointer select-none"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="w-1.5 self-stretch bg-primary rounded-l-lg shrink-0" />
        <CardHeader className="flex-1 py-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-bold">{title}</CardTitle>
            {open ? <ChevronUp className="w-5 h-5 text-muted-foreground" /> : <ChevronDown className="w-5 h-5 text-muted-foreground" />}
          </div>
        </CardHeader>
      </div>
      {open && (
        <>
          <div className="border-t border-border" />
          <CardContent className="pt-5">{children}</CardContent>
        </>
      )}
    </Card>
  );
}

/* ═══════════════ EXPORT WORD BUTTON ═══════════════ */

function ExportWordButton({ results, design, response }: { results: Record<string, unknown>; design: DesignType; response: string }) {
  const [exporting, setExporting] = useState(false);
  const ts = new Date().toISOString().slice(0, 10).replace(/-/g, "");

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      console.log(`[COMPONENT] ExportWordButton -> ${API_BASE}/genetics/analyze`);
      const res = await fetch(`${API_BASE}/genetics/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ result: results }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `anova_${design}_${response}_${ts}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      toast({ title: "Report exported", description: "Word report downloaded" });
    } catch {
      toast({ title: "Export failed", description: "Could not generate Word report.", variant: "destructive" });
    } finally {
      setExporting(false);
    }
  }, [results, design, response, ts]);

  return (
    <Button className="gap-2" onClick={handleExport} disabled={exporting}>
      {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
      Download Word Report
    </Button>
  );
}

/* ═══════════════ SECTION 1: MODEL SUMMARY ═══════════════ */

function isValidEtaSquared(eta: number | undefined | null): boolean {
  if (eta === null || eta === undefined) return false;
  if (isNaN(eta)) return false;
  if (eta < 0 || eta > 1) return false;
  return true;
}

function ModelSummaryCard({ data, designLabel, responseName, nObs }: {
  data: Record<string, unknown>;
  designLabel: string;
  responseName: string;
  nObs: number;
}) {
  const formula = data.model_formula ? String(data.model_formula) : (data.formula ? String(data.formula) : null);
  const modelStats = data.model_statistics as Record<string, unknown> | undefined;
  const designSummary = data.design_summary as Record<string, unknown> | undefined;
  const dfSummaryRaw = data.df_summary as Record<string, unknown> | undefined;
  const grandMean = modelStats?.grand_mean ?? data.grand_mean ?? (data.meta as Record<string, unknown>)?.grand_mean ?? null;
  const mseVal = modelStats?.mse ?? extractMSEFromAnova(data.anova_table ?? data.anova ?? (data.tables as Record<string, unknown>)?.anova);
  const cvPercent = modelStats?.cv_percent ?? data.cv_percent ?? null;
  const effectSizes = data.effect_sizes as Record<string, Record<string, unknown>> | undefined;
  const actualNObs = data.n_obs != null ? Number(data.n_obs) : nObs;

  // Extract DF summary from df_summary key or from ANOVA rows
  const anovaRaw = data.anova_table ?? data.anova ?? (data.tables as Record<string, unknown>)?.anova;
  const anovaRows = normalizeTable(anovaRaw);
  const dfSummary: { label: string; df: number }[] = [];
  if (dfSummaryRaw) {
    Object.entries(dfSummaryRaw).forEach(([label, df]) => {
      const n = Number(df);
      if (!isNaN(n)) dfSummary.push({ label, df: n });
    });
  } else if (anovaRows) {
    anovaRows.forEach((r) => {
      const src = String(r.source ?? r.Source ?? "");
      const df = Number(r.df ?? r.DF ?? r.Df ?? 0);
      if (!isNaN(df) && src) dfSummary.push({ label: src, df });
    });
  }

  return (
    <ResultSection title="Model Summary">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left column */}
        <div className="space-y-3">
          <SummaryItem label="Design" value={designLabel} />
          {formula && <SummaryItem label="Model Formula" value={formula} mono />}
          <SummaryItem label="Observations (n)" value={String(actualNObs)} />
          {grandMean != null && <SummaryItem label="Grand Mean" value={fmt(grandMean, 4)} />}
        </div>
        {/* Right column */}
        <div className="space-y-3">
          {mseVal != null && <SummaryItem label="MSE" value={fmt(mseVal, 4)} />}
          {cvPercent != null && <SummaryItem label="CV (%)" value={`${Number(cvPercent).toFixed(1)}%`} />}
          {effectSizes && Object.entries(effectSizes).map(([factor, sizes]) => {
            const eta = typeof sizes.eta_squared === "number" ? sizes.eta_squared : Number(sizes.eta_squared);
            return (
              <div key={factor}>
                {isValidEtaSquared(eta) ? (
                  <SummaryItem
                    label={`η² (${factor})`}
                    value={`${fmt(eta, 4)} — ${etaLabel(eta)}`}
                  />
                ) : (
                  <SummaryItem
                    label={`η² (${factor})`}
                    value={"η² unavailable — awaiting backend validation"}
                  />
                )}
                {sizes.omega_squared != null && (
                  <SummaryItem label={`ω² (${factor})`} value={fmt(sizes.omega_squared, 4)} />
                )}
              </div>
            );
          })}
        </div>
      </div>
      {/* DF Summary Row */}
      {dfSummary.length > 0 && (
        <div className="mt-5 p-3 rounded-lg bg-muted/40 border border-border">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">DF Summary</p>
          <div className="flex flex-wrap gap-x-6 gap-y-1">
            {dfSummary.map((d) => (
              <span key={d.label} className="text-sm font-mono">
                <span className="text-muted-foreground">{d.label}:</span>{" "}
                <span className="font-semibold text-foreground">{d.df}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </ResultSection>
  );
}

function SummaryItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-muted-foreground font-medium uppercase tracking-wide">{label}</span>
      <span className={`text-base font-semibold text-foreground ${mono ? "font-mono text-sm" : ""}`}>{value}</span>
    </div>
  );
}

function etaLabel(eta: number): string {
  if (eta >= 0.14) return "large";
  if (eta >= 0.06) return "medium";
  return "small";
}

function extractMSEFromAnova(anovaData: unknown): number | null {
  const rows = normalizeTable(anovaData);
  if (!rows) return null;
  for (const r of rows) {
    const src = String(r.source ?? r.Source ?? "").toLowerCase();
    if (/error|residual|within/.test(src)) {
      const ms = r.mean_sq ?? r.MS ?? r["Mean Sq"] ?? r.ms;
      if (ms != null) return Number(ms);
      const ss = Number(r.sum_sq ?? r.SS ?? r["Sum Sq"] ?? 0);
      const df = Number(r.df ?? r.DF ?? r.Df ?? 0);
      if (df > 0) return ss / df;
    }
  }
  return null;
}

/* ═══════════════ SECTION 2: ANOVA TABLE ═══════════════ */

function AnovaTableCard({ data }: { data: unknown }) {
  const rawRows = normalizeTable(data);
  if (!rawRows || rawRows.length === 0) return null;

  const rows = rawRows.map((r) => {
    const source = String(r.source ?? r.Source ?? "");
    const df = r.df ?? r.DF ?? r.Df;
    const ss = r.sum_sq ?? r.SS ?? r["Sum Sq"] ?? r["Sum of Squares"];
    const msRaw = r.mean_sq ?? r.MS ?? r["Mean Sq"] ?? r["Mean Square"];
    const fRaw = r.f_value ?? r.F ?? r["F value"] ?? r["F-value"];
    const pRaw = r.p_value ?? r["PR(>F)"] ?? r["Pr(>F)"] ?? r["p-value"] ?? r["p_value"] ?? r["pvalue"];
    const dfNum = df != null ? Number(df) : null;
    const ssNum = ss != null ? Number(ss) : null;
    const ms = msRaw != null ? Number(msRaw) : (ssNum != null && dfNum != null && dfNum > 0 ? ssNum / dfNum : null);
    return { source, df: dfNum, ss: ssNum, ms, f: fRaw != null ? Number(fRaw) : null, p: pRaw != null ? Number(pRaw) : null };
  });

  return (
    <ResultSection title="ANOVA Table">
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse font-mono border border-border rounded-lg overflow-hidden">
          <thead>
            <tr className="bg-muted/50 border-b-2 border-foreground/20">
              {["Source", "DF", "SS", "MS", "F value", "p-value", "Sig."].map((h) => (
                <th key={h} className={`px-4 py-3 font-semibold text-foreground ${h === "Source" ? "text-left" : "text-right"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const p = r.p;
              const stars = sigStars(p);
              const isErr = /error|residual|total/i.test(r.source);
              const pClass = getPValueClass(p);
              const sigClass = getSigClass(stars);
              return (
                <tr key={i} className={`border-b border-border/50 ${i % 2 === 0 ? "bg-background" : "bg-muted/20"} hover:bg-muted/30`}>
                  <td className="px-4 py-3 text-left font-medium text-foreground">{r.source}</td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{fmtInt(r.df)}</td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{fmt(r.ss, 4)}</td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{fmt(r.ms, 4)}</td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{isErr ? "—" : fmt(r.f, 4)}</td>
                  <td className={`px-4 py-3 text-right tabular-nums ${isErr ? "text-muted-foreground" : pClass}`}>
                    {isErr ? "—" : fmtP(r.p)}
                  </td>
                  <td className={`px-4 py-3 text-right font-bold whitespace-nowrap ${isErr ? "" : sigClass}`}>
                    {isErr ? "" : stars}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground mt-3 italic">
        Significance: '***' p &lt; 0.001, '**' p &lt; 0.01, '*' p &lt; 0.05, 'ns' not significant (α = 0.05)
      </p>
    </ResultSection>
  );
}

function getPValueClass(p: number | null): string {
  if (p == null || isNaN(p)) return "text-muted-foreground";
  if (p < 0.001) return "font-bold text-green-700 dark:text-green-400";
  if (p < 0.01) return "font-bold text-foreground";
  if (p < 0.05) return "font-bold text-foreground";
  return "text-muted-foreground";
}

function getSigClass(stars: string): string {
  if (stars === "***") return "text-green-700 dark:text-green-400";
  if (stars === "**") return "text-green-600 dark:text-green-500";
  if (stars === "*") return "text-yellow-600 dark:text-yellow-400";
  return "text-muted-foreground";
}

/* ═══════════════ SECTION 3: ASSUMPTION CHECKS ═══════════════ */

function AssumptionChecksCard({ data }: { data: Record<string, unknown> }) {
  const normality = data.normality as Record<string, unknown> | undefined;
  const homogeneity = (data.homogeneity ?? data.levene ?? data.homoscedasticity) as Record<string, unknown> | undefined;

  const tests: { name: string; target: string; test: string; stat: string; p: string; result: "PASS" | "FAIL" | "N/A" }[] = [];

  if (normality && typeof normality === "object") {
    const p = Number(normality.p_value ?? normality.p ?? NaN);
    const passed = (normality.passed as boolean) ?? (!isNaN(p) && p > 0.05);
    tests.push({
      name: "Normality of Residuals",
      target: "p > 0.05",
      test: String(normality.test ?? "Shapiro-Wilk"),
      stat: fmt(normality.statistic ?? normality.stat, 4),
      p: fmtP(normality.p_value ?? normality.p),
      result: isNaN(p) ? "N/A" : passed ? "PASS" : "FAIL",
    });
  }

  if (homogeneity && typeof homogeneity === "object") {
    const h = homogeneity;
    const p = Number(h.p_value ?? h.p ?? NaN);
    const passed = (h.passed as boolean) ?? (!isNaN(p) && p > 0.05);
    tests.push({
      name: "Homogeneity of Variances",
      target: "p > 0.05",
      test: String(h.test ?? "Levene"),
      stat: fmt(h.statistic ?? h.stat, 4),
      p: fmtP(h.p_value ?? h.p),
      result: isNaN(p) ? "N/A" : passed ? "PASS" : "FAIL",
    });
  }

  if (tests.length === 0) return null;
  const hasFail = tests.some((t) => t.result === "FAIL");

  return (
    <ResultSection title="Assumption Checks">
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse font-mono border border-border rounded-lg overflow-hidden">
          <thead>
            <tr className="bg-muted/50 border-b-2 border-foreground/20">
              {["Test", "Target", "Statistic", "p-value", "Result"].map((h) => (
                <th key={h} className={`px-4 py-3 font-semibold text-foreground ${h === "Test" ? "text-left" : "text-right"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tests.map((t, i) => (
              <tr key={t.name} className={`border-b border-border/50 ${i % 2 === 0 ? "bg-background" : "bg-muted/20"}`}>
                <td className="px-4 py-3 text-left font-medium">{t.name}</td>
                <td className="px-4 py-3 text-right text-muted-foreground">{t.target}</td>
                <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{t.stat}</td>
                <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{t.p}</td>
                <td className="px-4 py-3 text-right">
                  <Badge variant={t.result === "PASS" ? "default" : t.result === "FAIL" ? "destructive" : "secondary"}
                    className={t.result === "PASS" ? "bg-green-600 hover:bg-green-700" : ""}>
                    {t.result}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hasFail && (
        <div className="mt-4 flex items-start gap-2 p-3 rounded-lg bg-yellow-50 border border-yellow-300 dark:bg-yellow-950/20 dark:border-yellow-800">
          <AlertTriangle className="w-5 h-5 text-yellow-600 shrink-0 mt-0.5" />
          <p className="text-sm text-yellow-800 dark:text-yellow-200 font-medium">
            One or more assumptions violated. Interpret results with caution.
          </p>
        </div>
      )}
    </ResultSection>
  );
}

/* ═══════════════ SECTION 4: TABLE OF MEANS WITH TUKEY LETTERS ═══════════════ */

function TukeyMeansCard({ data, designType }: { data: Record<string, unknown>; designType: DesignType }) {
  const isFactorial = designType === "twoway" || designType === "rcbd_factorial";
  const skipped = data.skipped === true;
  const skipReason = data.reason ?? data.skip_reason;

  if (skipped) {
    return (
      <ResultSection title="Table of Means">
        <div className="p-4 rounded-lg bg-muted/40 border border-border flex items-start gap-2">
          <Info className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground">{String(skipReason ?? "Mean separation was skipped (treatment not significant).")}</p>
        </div>
      </ResultSection>
    );
  }

  const meansObj = data.means as Record<string, unknown> | undefined;
  const lettersObj = data.letters as Record<string, unknown> | undefined;

  if (!meansObj) return null;

  if (isFactorial) {
    // For factorial: render separate tables per factor + interaction
    const factorKeys = Object.keys(meansObj);
    return (
      <ResultSection title="Table of Means (Tukey HSD)">
        {factorKeys.map((fk) => {
          const mMap = meansObj[fk] as Record<string, number> | undefined;
          const lMap = lettersObj?.[fk] as Record<string, string> | undefined;
          if (!mMap || typeof mMap !== "object") return null;
          return (
            <div key={fk} className="mb-6 last:mb-0">
              <h4 className="font-semibold text-foreground mb-3">{fk} Means</h4>
              <MeansTable meansMap={mMap} lettersMap={lMap} />
            </div>
          );
        })}
        <TukeyNote />
      </ResultSection>
    );
  }

  // Single factor
  const factorKey = Object.keys(meansObj)[0];
  const meansMap = (factorKey ? meansObj[factorKey] : meansObj) as Record<string, number> | undefined;
  const lettersMap = lettersObj ? (factorKey && lettersObj[factorKey] ? lettersObj[factorKey] : lettersObj) as Record<string, string> | undefined : undefined;

  if (!meansMap || typeof meansMap !== "object") return null;

  return (
    <ResultSection title="Table of Means (Tukey HSD)">
      <MeansTable meansMap={meansMap} lettersMap={lettersMap} />
      <TukeyNote />
    </ResultSection>
  );
}

function MeansTable({ meansMap, lettersMap }: { meansMap: Record<string, number>; lettersMap?: Record<string, string> }) {
  const rows = Object.entries(meansMap)
    .map(([name, mean]) => ({ name, mean: Number(mean), letter: lettersMap?.[name] ?? "" }))
    .sort((a, b) => b.mean - a.mean);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse font-mono border border-border rounded-lg overflow-hidden">
        <thead>
          <tr className="bg-muted/50 border-b-2 border-foreground/20">
            {["Treatment/Group", "Mean", "Tukey Letter"].map((h) => (
              <th key={h} className={`px-4 py-3 font-semibold text-foreground ${h === "Treatment/Group" ? "text-left" : "text-right"}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name} className={`border-b border-border/50 ${i % 2 === 0 ? "bg-background" : "bg-muted/20"} hover:bg-muted/30`}>
              <td className="px-4 py-3 text-left font-medium text-foreground">{r.name}</td>
              <td className="px-4 py-3 text-right tabular-nums">{fmt(r.mean, 4)}</td>
              <td className="px-4 py-3 text-right">
                {r.letter ? (
                  <span className="font-bold text-green-700 dark:text-green-400 text-base">{r.letter}</span>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TukeyNote() {
  return (
    <p className="text-xs text-muted-foreground mt-3 italic">
      Means followed by the same letter are not significantly different (Tukey HSD, α = 0.05).
    </p>
  );
}

/* ═══════════════ SECTION 5: POST-HOC PAIRWISE COMPARISONS ═══════════════ */

function PairwiseCard({ data, designType }: { data: Record<string, unknown>; designType: DesignType }) {
  const pairwise = (data.pairwise ?? data.comparisons ?? data.pairs) as unknown;
  const pRows = normalizeTable(pairwise);

  if (!pRows || pRows.length === 0) return null;

  return (
    <ResultSection title="Post-hoc Pairwise Comparisons" defaultOpen={false}>
      <PairwiseTable rows={pRows} />
    </ResultSection>
  );
}

function PairwiseTable({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse font-mono border border-border rounded-lg overflow-hidden">
        <thead>
          <tr className="bg-muted/50 border-b-2 border-foreground/20">
            {["Pair", "Difference", "Lower CI", "Upper CI", "p-adj", "Significant"].map((h) => (
              <th key={h} className={`px-4 py-3 font-semibold text-foreground ${h === "Pair" ? "text-left" : "text-right"}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const p = Number(r.p_adj ?? r["p adj"] ?? r.p_value ?? NaN);
            const sig = !isNaN(p) && p < 0.05;
            return (
              <tr key={i} className={`border-b border-border/50 ${i % 2 === 0 ? "bg-background" : "bg-muted/20"} hover:bg-muted/30`}>
                <td className="px-4 py-3 text-left font-medium">{String(r.pair ?? r.comparison ?? r.Pair ?? `${r.group1 ?? ""} - ${r.group2 ?? ""}`)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.diff ?? r.difference ?? r.meandiff, 4)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.lower ?? r.ci_lower ?? r["lower CI"], 4)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.upper ?? r.ci_upper ?? r["upper CI"], 4)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmtP(r.p_adj ?? r["p adj"] ?? r.p_value)}</td>
                <td className="px-4 py-3 text-right">
                  {sig ? (
                    <span className="font-bold text-green-700 dark:text-green-400">Yes ✓</span>
                  ) : (
                    <span className="text-muted-foreground">No</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════ SECTION 6: DESCRIPTIVE STATISTICS ═══════════════ */

function DescriptiveCard({ data, designType }: { data: unknown; designType: DesignType }) {
  let rows: { group: string; n: number; mean: number; sd: number; se: number; cv: number }[] = [];

  if (data && typeof data === "object" && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    for (const [key, val] of Object.entries(obj)) {
      if (key === "overall" || typeof val !== "object" || val === null) continue;
      const levelMap = val as Record<string, unknown>;
      const firstVal = Object.values(levelMap)[0];
      if (firstVal && typeof firstVal === "object" && firstVal !== null && "mean" in (firstVal as Record<string, unknown>)) {
        rows = Object.entries(levelMap).map(([name, stats]) => {
          const s = stats as Record<string, number>;
          return { group: name, n: s.count ?? 0, mean: s.mean ?? 0, sd: s.std ?? 0, se: s.sem ?? 0, cv: s.cv ?? 0 };
        }).sort((a, b) => b.mean - a.mean);
        break;
      }
    }
    if (obj.overall && typeof obj.overall === "object") {
      const ov = obj.overall as Record<string, number>;
      rows.push({ group: "Overall", n: ov.n ?? ov.count ?? 0, mean: ov.mean ?? 0, sd: ov.std ?? 0, se: ov.sem ?? 0, cv: ov.cv ?? 0 });
    }
  }

  if (rows.length === 0) {
    const normalized = normalizeTable(data);
    if (!normalized || normalized.length === 0) return null;
    rows = normalized.map((r) => ({
      group: String(r.treatment ?? r.Treatment ?? r.group ?? Object.values(r)[0]),
      n: Number(r.count ?? r.n ?? r.N ?? 0),
      mean: Number(r.mean ?? r.Mean ?? 0),
      sd: Number(r.sd ?? r.SD ?? r.std ?? 0),
      se: Number(r.se ?? r.SE ?? r.sem ?? 0),
      cv: Number(r.cv ?? r.CV ?? 0),
    }));
  }

  if (rows.length === 0) return null;

  return (
    <ResultSection title="Descriptive Statistics" defaultOpen={false}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse font-mono border border-border rounded-lg overflow-hidden">
          <thead>
            <tr className="bg-muted/50 border-b-2 border-foreground/20">
              {["Group", "N", "Mean", "SD", "SE", "CV%"].map((h) => (
                <th key={h} className={`px-4 py-3 font-semibold text-foreground ${h === "Group" ? "text-left" : "text-right"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={`border-b border-border/50 ${r.group === "Overall" ? "font-semibold bg-muted/40" : i % 2 === 0 ? "bg-background" : "bg-muted/20"} hover:bg-muted/30`}>
                <td className="px-4 py-3 text-left font-medium">{r.group}</td>
                <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{fmtInt(r.n)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.mean, 4)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.sd, 4)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.se, 4)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{fmt(r.cv, 1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ResultSection>
  );
}

/* ═══════════════ SECTION 7: NOTES ═══════════════ */

function NotesCard({ data }: { data: unknown }) {
  const items = Array.isArray(data) ? data : typeof data === "string" ? [data] : [];
  if (items.length === 0) return null;

  return (
    <ResultSection title="Notes & Warnings">
      <div className="space-y-2">
        {items.map((n, i) => (
          <div key={i} className="flex items-start gap-2 p-3 rounded-lg bg-yellow-50 border border-yellow-300 dark:bg-yellow-950/20 dark:border-yellow-800">
            <AlertTriangle className="w-4 h-4 text-yellow-600 shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">{typeof n === "string" ? n : JSON.stringify(n)}</p>
          </div>
        ))}
      </div>
    </ResultSection>
  );
}

/* ═══════════════ SECTION 8: CHARTS ═══════════════ */

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ErrorBar, LineChart, Line, Cell } from "recharts";

function ChartsSection({ results, designType, responseName, fieldValues }: {
  results: Record<string, unknown>;
  designType: DesignType;
  responseName: string;
  fieldValues: Record<string, string>;
}) {
  const isFactorial = designType === "twoway" || designType === "rcbd_factorial";
  const meansObj = results.means as Record<string, unknown> | undefined;
  const descStats = results.descriptive_stats as Record<string, unknown> | undefined;

  if (!meansObj) return null;

  const factorKeys = Object.keys(meansObj);

  // Build chart data for each factor
  const factorCharts = factorKeys.map((fk) => {
    const mMap = meansObj[fk] as Record<string, number>;
    if (!mMap || typeof mMap !== "object") return null;

    // Try to get SE from descriptive stats
    let seMap: Record<string, number> = {};
    if (descStats && descStats[fk] && typeof descStats[fk] === "object") {
      const fStats = descStats[fk] as Record<string, Record<string, number>>;
      seMap = Object.fromEntries(
        Object.entries(fStats).map(([k, v]) => [k, v.sem ?? v.se ?? 0])
      );
    }

    // Get letters
    const lettersObj = results.letters as Record<string, unknown> | undefined;
    const lMap = lettersObj?.[fk] as Record<string, string> | undefined ?? (
      factorKeys.length === 1 && lettersObj
        ? (lettersObj[Object.keys(lettersObj)[0]] as Record<string, string> | undefined)
        : undefined
    );

    const chartData = Object.entries(mMap).map(([name, mean]) => ({
      name,
      mean: Number(mean),
      se: seMap[name] ?? 0,
      letter: lMap?.[name] ?? "",
    })).sort((a, b) => b.mean - a.mean);

    return { factorName: fk, data: chartData };
  }).filter(Boolean) as { factorName: string; data: { name: string; mean: number; se: number; letter: string }[] }[];

  if (factorCharts.length === 0) return null;

  return (
    <ResultSection title="Charts">
      <div className="space-y-8">
        {factorCharts.map((chart) => (
          <div key={chart.factorName}>
            <h4 className="font-semibold text-foreground mb-4 text-center">
              Mean {responseName} by {chart.factorName}
            </h4>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={chart.data} margin={{ top: 30, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                  tickLine={false}
                  label={{ value: responseName, angle: -90, position: "insideLeft", style: { fontSize: 12, fill: "hsl(var(--muted-foreground))" } }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px" }}
                  formatter={(value: number) => [fmt(value, 4), "Mean"]}
                />
                <Bar dataKey="mean" fill="#0A7F5A" radius={[4, 4, 0, 0]} maxBarSize={60}
                  label={({ x, y, width, index }: { x: number; y: number; width: number; index: number }) => {
                    const d = chart.data[index];
                    if (!d?.letter) return null;
                    return (
                      <text x={x + width / 2} y={y - 8} textAnchor="middle" fill="#0A7F5A" fontWeight="bold" fontSize={14}>
                        {d.letter}
                      </text>
                    );
                  }}
                >
                  <ErrorBar dataKey="se" width={4} strokeWidth={2} stroke="#333" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}

        {/* Interaction plot for factorial designs */}
        {isFactorial && factorCharts.length >= 2 && (
          <InteractionPlot results={results} responseName={responseName} fieldValues={fieldValues} />
        )}
      </div>
    </ResultSection>
  );
}

function InteractionPlot({ results, responseName, fieldValues }: {
  results: Record<string, unknown>;
  responseName: string;
  fieldValues: Record<string, string>;
}) {
  const descStats = results.descriptive_stats as Record<string, unknown> | undefined;
  if (!descStats) return null;

  // Try to find cell-level data (A×B combinations)
  // The API may return descriptive_stats with nested structure
  const factorA = fieldValues.factor_a;
  const factorB = fieldValues.factor_b;

  // Build interaction data from means
  const meansObj = results.means as Record<string, unknown> | undefined;
  if (!meansObj) return null;

  // Look for interaction key (e.g., "A:B" or combined)
  const interKey = Object.keys(meansObj).find((k) => k.includes(":") || k.includes("×"));
  if (!interKey) return null;

  const cellMeans = meansObj[interKey] as Record<string, number>;
  if (!cellMeans) return null;

  // Parse "A:B" formatted keys
  const COLORS = ["#0A7F5A", "#2563EB", "#D97706", "#DC2626", "#7C3AED", "#0891B2"];
  const dataMap: Record<string, Record<string, number>> = {};
  const bLevels = new Set<string>();

  Object.entries(cellMeans).forEach(([key, mean]) => {
    const parts = key.split(":");
    if (parts.length === 2) {
      const [a, b] = parts;
      if (!dataMap[a]) dataMap[a] = {};
      dataMap[a][b] = Number(mean);
      bLevels.add(b);
    }
  });

  const bArr = Array.from(bLevels);
  const chartData = Object.entries(dataMap).map(([aLevel, bMeans]) => ({
    name: aLevel,
    ...bMeans,
  }));

  if (chartData.length === 0) return null;

  return (
    <div>
      <h4 className="font-semibold text-foreground mb-4 text-center">
        Interaction Plot: {factorA ?? "Factor A"} × {factorB ?? "Factor B"}
      </h4>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
            axisLine={{ stroke: "hsl(var(--border))" }}
            tickLine={false}
            label={{ value: factorA ?? "Factor A", position: "insideBottom", offset: -10, style: { fontSize: 12 } }}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
            axisLine={{ stroke: "hsl(var(--border))" }}
            tickLine={false}
            label={{ value: responseName, angle: -90, position: "insideLeft", style: { fontSize: 12 } }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px" }}
          />
          <Legend />
          {bArr.map((b, i) => (
            <Line key={b} type="monotone" dataKey={b} name={b} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 7 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ─── Table normalizer ─── */
function normalizeTable(data: unknown): Record<string, unknown>[] | null {
  if (Array.isArray(data) && data.length > 0) {
    if (typeof data[0] === "object" && data[0] !== null && !Array.isArray(data[0])) {
      return data as Record<string, unknown>[];
    }
    if (Array.isArray(data[0])) {
      const headers = (data[0] as unknown[]).map(String);
      return data.slice(1).map((row: unknown) => {
        const r: Record<string, unknown> = {};
        (row as unknown[]).forEach((cell, i) => { r[headers[i]] = cell; });
        return r;
      });
    }
  }
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    const keys = Object.keys(obj);
    if (keys.length > 0 && typeof obj[keys[0]] === "object" && obj[keys[0]] !== null) {
      const innerKeys = Object.keys(obj[keys[0]] as Record<string, unknown>);
      return innerKeys.map((ik) => {
        const row: Record<string, unknown> = { source: ik };
        keys.forEach((k) => { row[k] = (obj[k] as Record<string, unknown>)[ik]; });
        return row;
      });
    }
  }
  return null;
}
