import { useEffect, useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  BarChart3,
  Sigma,
  GitCompare,
  Network,
  Compass,
  GitBranch,
  Award,
  FileDown,
  Database,
  Lock,
  Sparkles,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { VivaSenseProvider } from "@/contexts/VivaSenseContext";
import { VivaSenseForm, AnalysisType, type DatasetStatus } from "@/components/vivasense/VivaSenseForm";
import { VivaSenseGeneticsForm, GeneticsAnalysisType } from "@/components/vivasense/VivaSenseGeneticsForm";
import { VivaSenseLoading } from "@/components/vivasense/VivaSenseLoading";
import { VivaSenseError } from "@/components/vivasense/VivaSenseError";
import { VivaSenseFastAPIResults, FastAPIResultsData } from "@/components/vivasense/VivaSenseFastAPIResults";
import { VivaSenseMultiTraitResults, MultiTraitResultsData } from "@/components/vivasense/VivaSenseMultiTraitResults";
import VivaSenseResultsDisplay from "@/components/vivasense/VivaSenseResultsDisplay";
import { VivaSenseInterpretation } from "@/components/vivasense/VivaSenseInterpretation";
import { VivaSenseMultiTraitInterpretation, type GroundingCheck } from "@/components/vivasense/VivaSenseMultiTraitInterpretation";
import { VivaSenseGeneticsResultsTabs, GeneticsMultiTraitResults } from "@/components/vivasense/VivaSenseGeneticsResultsTabs";
import { ProFeatureModal } from "@/components/vivasense/ProFeatureModal";
import { VivaSensePlanBadge } from "@/components/vivasense/VivaSensePlanBadge";
import { InstallVivaSenseButton } from "@/components/vivasense/InstallVivaSenseButton";
import { computeDescriptiveStats } from "@/lib/descriptiveStatsApi";
import { analyzeUpload, computeAnova, fileToBase64 } from "@/lib/geneticsUploadApi";
import {
  classifyAnovaRequest,
  classifyGeneticsRequest,
  isProMode,
  getProGuardInfo,
  subscribeVivaSenseMode,
  type ProGuardInfo,
} from "@/lib/vivasenseGating";

type ModuleId =
  | "descriptive"
  | "anova"
  | "genetics_correlations"
  | "gxe"
  | "genetic_parameters"
  | "pca"
  | "clustering"
  | "path_analysis"
  | "selection_index"
  | "export";

interface ModuleEntry {
  id: ModuleId;
  label: string;
  icon: React.ElementType;
  pro: boolean;
  group: "BASIC" | "ADVANCED" | "UTILITIES";
  proKey?: Parameters<typeof getProGuardInfo>[0];
}

const MODULES: ModuleEntry[] = [
  { id: "descriptive", label: "Descriptive Statistics", icon: Sigma, pro: false, group: "BASIC" },
  { id: "anova", label: "ANOVA (Single Environment)", icon: BarChart3, pro: false, group: "BASIC" },
  { id: "genetics_correlations", label: "Trait Correlations", icon: GitCompare, pro: false, group: "BASIC" },
  { id: "gxe", label: "G×E Analysis", icon: Network, pro: true, group: "ADVANCED", proKey: "anova_gxe" },
  { id: "genetic_parameters", label: "Genetic Parameters", icon: Sigma, pro: true, group: "ADVANCED", proKey: "genetic_parameters" },
  { id: "pca", label: "PCA", icon: Compass, pro: true, group: "ADVANCED", proKey: "pca" },
  { id: "clustering", label: "Clustering", icon: Network, pro: true, group: "ADVANCED", proKey: "clustering" },
  { id: "path_analysis", label: "Path Analysis", icon: GitBranch, pro: true, group: "ADVANCED", proKey: "path_analysis" },
  { id: "selection_index", label: "Selection Index / MGIDI", icon: Award, pro: true, group: "ADVANCED", proKey: "selection_index" },
  { id: "export", label: "Export Results", icon: FileDown, pro: true, group: "UTILITIES", proKey: "export_word" },
];

function VivaSenseWorkspaceInner() {
  const navigate = useNavigate();
  const [active, setActive] = useState<ModuleId>("anova");
  const [proGuard, setProGuard] = useState<ProGuardInfo | null>(null);
  const [, force] = useState(0);
  useEffect(() => subscribeVivaSenseMode(() => force((n) => n + 1)), []);

  // ANOVA state
  const [anovaLoading, setAnovaLoading] = useState(false);
  const [anovaError, setAnovaError] = useState<string | null>(null);
  const clearError = () => setAnovaError(null);
  const [results, setResults] = useState<FastAPIResultsData | null>(null);
  const [multiTraitResults, setMultiTraitResults] = useState<MultiTraitResultsData | null>(null);
  const [lastAnalysisType, setLastAnalysisType] = useState<string>("");
  const [lastDesign, setLastDesign] = useState<string>("");
  const [traitInterpretations, setTraitInterpretations] = useState<Record<string, string>>({});
  const [completedTraits, setCompletedTraits] = useState<Set<string>>(new Set());
  const [traitGroundingChecks, setTraitGroundingChecks] = useState<Record<string, GroundingCheck>>({});
  const [datasetMeta, setDatasetMeta] = useState<{ name: string; rows: number } | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>({ filename: null, rows: 0, cols: 0, error: null });

  // Genetics state
  const [geneticsLoading, setGeneticsLoading] = useState(false);
  const [geneticsError, setGeneticsError] = useState<string | null>(null);
  const [geneticsResults, setGeneticsResults] = useState<GeneticsMultiTraitResults | null>(null);
  const [lastGeneticsType, setLastGeneticsType] = useState<string>("");
  const [lastGeneticsTraits, setLastGeneticsTraits] = useState<string[]>([]);

  const handleModuleClick = (m: ModuleEntry) => {
    if (m.pro && !isProMode()) {
      setProGuard(getProGuardInfo(m.proKey ?? "advanced_interpretation"));
      return;
    }
    setActive(m.id);
  };

  const getFileType = (file: File) => (file.name.split(".").pop()?.toLowerCase() || "csv") as "csv" | "xlsx" | "xls";
  const toSingleTraitAnovaResult = (response: Record<string, unknown>, trait: string): Record<string, unknown> => {
    const traitResults = response.trait_results as Record<string, unknown> | undefined;
    const traitResult = (traitResults?.[trait] as Record<string, unknown> | undefined) ?? response;

    // Extract nested analysis_result fields if present (from /genetics/analyze-upload flow)
    const analysisResult = (traitResult as any).analysis_result;
    const resultFields = analysisResult?.result || {};

    return {
      ...traitResult,
      ...resultFields,
      response: trait,
      anova: (traitResult as any).anova_table
    };
  };

  const handleAnovaSubmit = async (analysisType: AnalysisType, formData: FormData) => {
    if (!isProMode() && analysisType !== "descriptive") {
      const guard = classifyAnovaRequest(analysisType, formData);
      if (guard) { setProGuard(guard); return; }
    }
    clearError();
    setResults(null);
    setMultiTraitResults(null);
    setAnovaLoading(true);
    try {
      const file = formData.get("file") as File;
      const datasetToken = (formData.get("dataset_token") as string) || datasetStatus.datasetToken;
      if (!datasetToken) throw new Error("Upload a dataset to begin.");
      setDatasetMeta({ name: file.name, rows: datasetStatus.rows });

      if (analysisType === "multitrait") {
        const selectedDesign = (formData.get("design") as string) || "oneway";
        const traits = ((formData.get("traits") as string) || "").split(",").filter(Boolean);
        const data = await analyzeUpload({
          base64_content: await fileToBase64(file),
          file_type: getFileType(file),
          genotype_column: (formData.get("factor") as string) || (formData.get("treatment") as string) || (formData.get("factor_a") as string),
          rep_column: (formData.get("block") as string) || null,
          environment_column: null,
          trait_columns: traits,
          mode: "single",
          random_environment: false,
          selection_intensity: 1.4,
          module: "anova",
          design_type: selectedDesign === "oneway" ? "crd" : selectedDesign === "oneway_rcbd" ? "rcbd" : selectedDesign === "splitplot" ? "split_plot_rcbd" : "factorial",
          treatment_column: (formData.get("factor") as string) || (formData.get("treatment") as string) || undefined,
          factor_a_column: (formData.get("factor_a") as string) || undefined,
          factor_b_column: (formData.get("factor_b") as string) || undefined,
          main_plot_column: (formData.get("main_plot") as string) || undefined,
          sub_plot_column: (formData.get("sub_plot") as string) || undefined,
        });
        setMultiTraitResults({ per_trait: data.trait_results as any, meta: { design: selectedDesign, traits, n_traits: traits.length } } as any);
        setLastAnalysisType(analysisType);
        setLastDesign(selectedDesign);
      } else if (analysisType === "descriptive") {
        const columns = formData.getAll("columns") as string[];
        const data = await computeDescriptiveStats({ dataset_token: datasetToken, trait_columns: columns, genotype_column: (formData.get("by") as string) || null });
        setResults(data as unknown as FastAPIResultsData);
        setLastAnalysisType(analysisType);
        setLastDesign("");
      } else {
        const trait = formData.get("trait") as string;
        const data = await computeAnova({ dataset_token: datasetToken, trait_columns: [trait] });
        setResults(toSingleTraitAnovaResult(data, trait) as FastAPIResultsData);
        setLastAnalysisType(analysisType);
        setLastDesign("");
      }
    } catch (err: any) {
      setAnovaError(err?.message || "Analysis failed.");
    } finally {
      setAnovaLoading(false);
    }
  };

  const handleGeneticsSubmit = async (analysisType: GeneticsAnalysisType, formData: FormData) => {
    if (!isProMode()) {
      const guard = classifyGeneticsRequest(analysisType);
      if (guard) { setProGuard(guard); return; }
    }
    setGeneticsLoading(true);
    setGeneticsError(null);
    setGeneticsResults(null);
    try {
      const file = formData.get("file") as File;
      const traits = analysisType === "regression"
        ? [(formData.get("response_col") as string)].filter(Boolean)
        : ((formData.get("traits") as string) || "").split(",").filter(Boolean);
      if (traits.length === 0) throw new Error("Please select at least one trait.");
      const data = await analyzeUpload({
        base64_content: await fileToBase64(file),
        file_type: getFileType(file),
        genotype_column: (formData.get("genotype") as string) || traits[0],
        rep_column: (formData.get("rep") as string) || null,
        environment_column: (formData.get("location") as string) || null,
        trait_columns: traits,
        mode: (formData.get("location") as string) ? "multi" : "single",
        random_environment: false,
        selection_intensity: 1.4,
        module: analysisType === "correlations" ? "correlation" : "genetic_parameters",
      });
      setGeneticsResults(data as unknown as GeneticsMultiTraitResults);
      setLastGeneticsTraits(traits);
      setLastGeneticsType(analysisType);
    } catch (err: any) {
      setGeneticsError(err.message || "An error occurred during analysis.");
    } finally {
      setGeneticsLoading(false);
    }
  };

  // What form to show in main panel
  const renderConfig = () => {
    if (active === "descriptive" || active === "anova") {
      return (
        <VivaSenseForm
          onSubmit={handleAnovaSubmit}
          isLoading={anovaLoading}
          onDatasetChange={(status) => {
            setDatasetStatus(status);
            setDatasetMeta(status.filename && !status.error ? { name: status.filename, rows: status.rows } : null);
          }}
        />
      );
    }
    if (active === "genetics_correlations") {
      return (
        <VivaSenseGeneticsForm onSubmit={handleGeneticsSubmit} isLoading={geneticsLoading} />
      );
    }
    return null;
  };

  const groups: Array<{ label: string; tone: string; items: ModuleEntry[] }> = [
    { label: "BASIC", tone: "Free", items: MODULES.filter((m) => m.group === "BASIC") },
    { label: "ADVANCED", tone: "Pro", items: MODULES.filter((m) => m.group === "ADVANCED") },
    { label: "UTILITIES", tone: "", items: MODULES.filter((m) => m.group === "UTILITIES") },
  ];

  const pro = isProMode();

  return (
    <Layout>
      {/* Top bar */}
      <div className="border-b bg-background/80 backdrop-blur sticky top-0 z-10">
        <div className="container-wide flex items-center justify-between gap-4 py-3">
          <div className="flex items-center gap-3 min-w-0">
            <Database className="h-5 w-5 text-muted-foreground shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-semibold truncate">
                {datasetMeta ? `Dataset: ${datasetMeta.name}` : "No dataset loaded"}
              </p>
              <p className="text-xs text-muted-foreground">
                {datasetMeta ? `${datasetMeta.rows} rows` : "Upload a CSV/XLSX to begin"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <VivaSensePlanBadge />
            <InstallVivaSenseButton />
            {!pro && (
              <Button
                size="sm"
                onClick={() => setProGuard(getProGuardInfo("genetic_parameters"))}
                style={{ backgroundColor: "#1B5E20" }}
                className="text-white hover:opacity-90 gap-1.5"
              >
                <Sparkles className="h-3.5 w-3.5" />
                Upgrade
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => navigate("/vivasense")}>
              ← Landing
            </Button>
          </div>
        </div>
      </div>

      <div className="container-wide py-6 grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-6">
        {/* Sidebar */}
        <aside className="space-y-5">
          {groups.map((g) => (
            <div key={g.label}>
              <div className="flex items-center justify-between mb-2 px-1">
                <p className="text-[11px] font-semibold tracking-wider text-muted-foreground">
                  {g.label}
                </p>
                {g.tone && (
                  <Badge variant="secondary" className="text-[9px] uppercase">{g.tone}</Badge>
                )}
              </div>
              <div className="space-y-1">
                {g.items.map((m) => {
                  const Icon = m.icon;
                  const locked = m.pro && !pro;
                  const isActive = active === m.id;
                  return (
                    <button
                      key={m.id}
                      onClick={() => handleModuleClick(m)}
                      className={`w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors text-left ${
                        isActive
                          ? "text-white"
                          : locked
                          ? "text-muted-foreground hover:bg-muted/60"
                          : "text-foreground hover:bg-muted"
                      }`}
                      style={isActive ? { backgroundColor: "#1B5E20" } : undefined}
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      <span className="truncate">{m.label}</span>
                      {locked && <Lock className="h-3 w-3 ml-auto shrink-0 opacity-60" />}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </aside>

        {/* Workspace */}
        <main className="min-w-0 space-y-6">
          <Card className="p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="font-serif text-xl font-semibold text-foreground">
                {MODULES.find((m) => m.id === active)?.label ?? "Analysis"}
              </h2>
              <p className="text-xs text-muted-foreground">
                Configure your dataset and run a publication-ready analysis.
              </p>
            </div>
            {renderConfig()}
            <VivaSenseLoading isLoading={anovaLoading || geneticsLoading} />
            <VivaSenseError error={anovaError || geneticsError} onDismiss={() => { clearError(); setGeneticsError(null); }} />
          </Card>

          {/* Results */}
          {(results || multiTraitResults || geneticsResults) && (
            <Card className="p-5 shadow-sm">
              <h3 className="font-serif text-lg font-semibold mb-3">Results</h3>

              {results && (
                <>
                  <VivaSenseResultsDisplay result={results as unknown as Record<string, unknown>} />
                  <div className="mt-6">
                    <VivaSenseInterpretation analysisType={lastAnalysisType} results={results} />
                  </div>
                </>
              )}

              {multiTraitResults && (
                <>
                  <VivaSenseMultiTraitResults
                    results={multiTraitResults}
                    onClear={() => setMultiTraitResults(null)}
                    traitInterpretations={traitInterpretations}
                    completedTraits={completedTraits}
                    traitGroundingChecks={traitGroundingChecks}
                  />
                  <div className="mt-6">
                    <VivaSenseMultiTraitInterpretation
                      designLabel={lastDesign.replace(/_/g, " ")}
                      results={multiTraitResults}
                      onTraitInterpretations={setTraitInterpretations}
                      onTraitProgress={setCompletedTraits}
                      onTraitGroundingChecks={setTraitGroundingChecks}
                    />
                  </div>
                </>
              )}

              {geneticsResults && (
                <VivaSenseGeneticsResultsTabs
                  results={geneticsResults}
                  analysisType={lastGeneticsType}
                  selectedTraits={lastGeneticsTraits}
                  onClear={() => setGeneticsResults(null)}
                />
              )}
            </Card>
          )}
        </main>
      </div>

      <ProFeatureModal
        open={!!proGuard}
        onOpenChange={(o) => !o && setProGuard(null)}
        guard={proGuard}
      />
    </Layout>
  );
}

export default function VivaSenseWorkspace() {
  return (
    <VivaSenseProvider>
      <VivaSenseWorkspaceInner />
    </VivaSenseProvider>
  );
}
