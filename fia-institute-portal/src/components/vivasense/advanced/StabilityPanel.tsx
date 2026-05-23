import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Play, AlertTriangle, TrendingUp } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runStability, buildStabilityPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type {
  StabilityResponse, StabilityMethod, GgeBiplotType,
} from "@/types/advancedAnalysis";
import { DatasetTokenWarning } from "./shared";
import { ClassicTab } from "./stability/ClassicTab";
import { AmmiTab } from "./stability/AmmiTab";
import { GgeTab } from "./stability/GgeTab";
import { ComparisonTab } from "./stability/ComparisonTab";

interface Props {
  datasetContext: DatasetContext | null;
}

const ALL_METHODS: { id: StabilityMethod; label: string }[] = [
  { id: "eberhart-russell", label: "Eberhart–Russell" },
  { id: "shukla", label: "Shukla" },
  { id: "ammi", label: "AMMI" },
  { id: "gge-biplot", label: "GGE Biplot" },
];

export function StabilityPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const traits = datasetContext?.availableTraitColumns ?? [];
  const [trait, setTrait] = useState<string>(traits[0] ?? "");
  const [methods, setMethods] = useState<StabilityMethod[]>(ALL_METHODS.map((m) => m.id));
  const [biplotType, setBiplotType] = useState<GgeBiplotType>("which-won-where");
  const [ammiComponents, setAmmiComponents] = useState<number>(2);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StabilityResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"classic" | "ammi" | "gge" | "comparison">("classic");

  const datasetToken = datasetContext?.datasetToken ?? null;
  const ggeEnabled = methods.includes("gge-biplot");
  const ammiEnabled = methods.includes("ammi");

  const toggleMethod = (id: StabilityMethod) => {
    setMethods((prev) => (prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]));
  };

  const handleRun = async () => {
    if (!datasetToken) {
      toast({ title: "Dataset required", description: "Please upload and confirm a dataset first." });
      return;
    }
    if (!trait) {
      toast({ title: "Select a trait", description: "Pick a trait to analyze." });
      return;
    }
    if (methods.length === 0) {
      toast({ title: "Select a method", description: "Choose at least one stability method." });
      return;
    }
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await runStability(
        buildStabilityPayload({
          datasetToken,
          trait,
          methods,
          biplotType: ggeEnabled ? biplotType : undefined,
          ammiComponents: ammiEnabled ? ammiComponents : undefined,
        })
      );
      if (res.status !== "success") {
        throw new Error("Stability analysis failed on the server.");
      }
      const response = { data: res };
      console.log("FULL STABILITY RESPONSE", response);
      console.log("AMMI DATA", response.data);
      console.log("AVAILABLE KEYS", Object.keys(response.data || {}));
      console.log("[stability response]", res);

      console.log("AMMI RESULTS KEYS", Object.keys((res.ammi_results as unknown as Record<string, unknown>) || {}));
      console.log(
        "AMMI BIPLOT KEYS",
        Object.keys((res.ammi_results?.biplot_data as unknown as Record<string, unknown>) || {})
      );

      setResult(res);
      // Auto-pick the most informative tab available
      const hasAmmi = !!res.ammi_results && Array.isArray(res.ammi_results.anova_table?.source);
      const hasGge = !!res.gge_results && !!res.gge_results.biplot_data;
      if (hasAmmi) setActiveTab("ammi");
      else if (hasGge) setActiveTab("gge");
      else setActiveTab("classic");
      toast({
        title: "Stability analysis complete",
        description: `${res.n_genotypes} genotypes × ${res.n_environments} environments`,
      });
    } catch (e) {
      const raw = (e as Error).message ?? "Unexpected error";
      // Friendlier messaging for common backend cases
      let msg = raw;
      if (/at least 2 environments|n_environments/i.test(raw))
        msg = "Stability analysis requires at least 2 environments. Your dataset has only 1.";
      else if (/missing|nan/i.test(raw))
        msg = `Trait "${trait}" has too many missing values. Please select a different trait or clean your data.`;
      else if (/timeout|aborted/i.test(raw))
        msg = "Analysis is taking longer than expected. Please wait or refresh and try again.";
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally {
      setIsRunning(false);
    }
  };

  const tabsAvailable = useMemo(() => {
    if (!result) return { classic: false, ammi: false, gge: false, comparison: false };
    const hasGenoStab = Array.isArray(result.genotype_stability) && result.genotype_stability.length > 0;
    const hasAmmi = !!result.ammi_results && Array.isArray(result.ammi_results.anova_table?.source);
    const hasGge = !!result.gge_results && !!result.gge_results.biplot_data;
    return {
      classic: hasGenoStab,
      ammi: hasAmmi,
      gge: hasGge,
      comparison: hasGenoStab,
    };
  }, [result]);

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Stability Analysis
            <Badge variant="secondary" className="text-[10px] uppercase">G×E</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Compute Eberhart–Russell, Shukla, AMMI and GGE biplot analyses to identify stable, adapted
            genotypes across environments. Interpretation appears before raw statistics.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <Label className="text-xs uppercase tracking-wide">Trait</Label>
              <Select value={trait} onValueChange={setTrait}>
                <SelectTrigger><SelectValue placeholder="Select a trait" /></SelectTrigger>
                <SelectContent>
                  {traits.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-xs uppercase tracking-wide">Methods</Label>
              <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-2">
                {ALL_METHODS.map((m) => (
                  <label key={m.id} className="flex items-center gap-2 text-sm cursor-pointer select-none">
                    <Checkbox
                      checked={methods.includes(m.id)}
                      onCheckedChange={() => toggleMethod(m.id)}
                    />
                    {m.label}
                  </label>
                ))}
              </div>
            </div>

            {ggeEnabled && (
              <div>
                <Label className="text-xs uppercase tracking-wide">GGE biplot view</Label>
                <div className="mt-2 flex flex-wrap gap-2">
                  {[
                    { value: "mean-stability", label: "Mean vs Stability" },
                    { value: "which-won-where", label: "Which-Won-Where" },
                    { value: "discriminativeness", label: "Environment Evaluation" },
                  ].map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setBiplotType(option.value as GgeBiplotType)}
                      className={[
                        "rounded-full border px-3 py-1.5 text-sm transition-colors",
                        biplotType === option.value
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground",
                      ].join(" ")}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {ammiEnabled && (
              <div>
                <Label className="text-xs uppercase tracking-wide">AMMI components (IPCA)</Label>
                <Select value={String(ammiComponents)} onValueChange={(v) => setAmmiComponents(Number(v))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {[1, 2, 3, 4].map((n) => <SelectItem key={n} value={String(n)}>{n}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          <Button onClick={handleRun} disabled={isRunning || !trait || methods.length === 0} className="gap-2">
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Stability Analysis
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="p-4 flex items-start gap-3 text-sm">
            <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
            <div className="text-destructive">{error}</div>
          </CardContent>
        </Card>
      )}

      {isRunning && (
        <div className="space-y-4">
          <Skeleton className="h-20" />
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}
          </div>
          <Skeleton className="h-72" />
          <p className="text-xs text-muted-foreground">
            Computing requested methods (AMMI decomposition + GGE biplot can take 15–30s on large datasets)…
          </p>
        </div>
      )}

      {result && (
        <Card>
          <CardContent className="p-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <SummaryStat label="Trait" value={result.trait ?? "—"} />
            <SummaryStat label="Genotypes" value={String(result.n_genotypes ?? result.genotype_stability?.length ?? "—")} />
            <SummaryStat label="Environments" value={String(result.n_environments ?? Object.keys(result.environment_means ?? {}).length ?? "—")} />
            <SummaryStat label="Grand mean" value={typeof result.grand_mean === "number" ? result.grand_mean.toFixed(2) : "—"} />
          </CardContent>
        </Card>
      )}

      {result && (
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
          <TabsList className="grid grid-cols-2 md:grid-cols-4 w-full md:w-auto">
            <TabsTrigger value="classic" disabled={!tabsAvailable.classic}>Classic</TabsTrigger>
            <TabsTrigger value="ammi" disabled={!tabsAvailable.ammi}>AMMI</TabsTrigger>
            <TabsTrigger value="gge" disabled={!tabsAvailable.gge}>GGE Biplot</TabsTrigger>
            <TabsTrigger value="comparison" disabled={!tabsAvailable.comparison}>Comparison</TabsTrigger>
          </TabsList>
          <TabsContent value="classic" className="mt-6">
            <ClassicTab result={result} />
          </TabsContent>
          <TabsContent value="ammi" className="mt-6">
            {result.ammi_results
              ? <AmmiTab ammi={result.ammi_results} trait={result.trait} />
              : <EmptyTab message="AMMI results not returned for this run. Enable the AMMI method and re-run." />}
          </TabsContent>
          <TabsContent value="gge" className="mt-6">
            {result.gge_results
              ? <div className="mx-auto max-w-[1280px]"><GgeTab gge={result.gge_results} trait={result.trait} biplotType={biplotType} /></div>
              : <EmptyTab message="GGE biplot results not returned for this run. Enable the GGE Biplot method and re-run." />}
          </TabsContent>
          <TabsContent value="comparison" className="mt-6">
            <ComparisonTab result={result} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}

function EmptyTab({ message }: { message: string }) {
  return (
    <Card>
      <CardContent className="p-6 text-sm text-muted-foreground text-center">{message}</CardContent>
    </Card>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-base font-semibold tabular-nums">{value}</div>
    </div>
  );
}
