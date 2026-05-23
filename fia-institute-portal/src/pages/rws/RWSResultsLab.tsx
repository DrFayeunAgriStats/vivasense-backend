import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FlaskConical, Upload, ClipboardPaste, FileText, Loader2, ArrowRight } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { RWSChatPanel } from "@/components/rws/RWSChatPanel";
import { supabase } from "@/integrations/supabase/client";

const ANALYSIS_TYPES = [
  { value: "anova", label: "ANOVA" },
  { value: "correlation", label: "Correlation Matrix" },
  { value: "regression", label: "Regression" },
  { value: "pca", label: "Principal Component Analysis (PCA)" },
  { value: "cluster", label: "Cluster Analysis" },
  { value: "ammi", label: "AMMI" },
  { value: "gge", label: "GGE Biplot" },
  { value: "ssr", label: "Molecular Marker Analysis (SSR)" },
];

export default function RWSResultsLab() {
  const { user, profile, session } = useAuth();
  const { toast } = useToast();
  const [analysisType, setAnalysisType] = useState("");
  const [inputMethod, setInputMethod] = useState<"paste" | "csv">("paste");
  const [pastedData, setPastedData] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [dataSummary, setDataSummary] = useState("");

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      toast({ title: "Error", description: "Please upload a CSV file.", variant: "destructive" });
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setPastedData(text);
      setInputMethod("csv");
    };
    reader.readAsText(file);
  };

  const handleSubmit = async () => {
    if (!analysisType) {
      toast({ title: "Select analysis type", description: "Please choose the type of statistical output.", variant: "destructive" });
      return;
    }
    if (!pastedData.trim()) {
      toast({ title: "No data", description: "Please paste your table or upload a CSV file.", variant: "destructive" });
      return;
    }

    setIsSubmitting(true);

    try {
      // Save upload to DB if authenticated
      if (user) {
        await supabase.from("analysis_uploads").insert({
          user_id: user.id,
          analysis_type: analysisType,
          input_data: pastedData.substring(0, 50000),
          input_method: inputMethod,
        });
      }

      // Create a short summary for context
      const lines = pastedData.trim().split("\n");
      const summary = `Analysis type: ${analysisType}. Data has ${lines.length} rows. First few lines:\n${lines.slice(0, 5).join("\n")}`;
      setDataSummary(summary);
      setSubmitted(true);
    } catch (err) {
      toast({ title: "Error", description: "Failed to process data.", variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setSubmitted(false);
    setPastedData("");
    setAnalysisType("");
    setDataSummary("");
  };

  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-12 md:py-16">
        <div className="container max-w-5xl">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-full bg-primary-foreground/10 flex items-center justify-center">
              <FlaskConical className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-serif text-3xl md:text-4xl font-bold">Guided Results Interpretation Lab</h1>
              <p className="text-primary-foreground/70 text-sm mt-1">
                Learn to interpret statistical output using the FIA 4-Level Thinking Framework
              </p>
            </div>
          </div>
          <p className="text-primary-foreground/80 max-w-2xl text-sm">
            Upload or paste your statistical output, and the AI will generate guided interpretation questions — not answers.
            You develop the interpretation skills; the AI guides your thinking.
          </p>
        </div>
      </section>

      <section className="container max-w-5xl py-8">
        {!submitted ? (
          <div className="space-y-6 max-w-3xl mx-auto">
            {/* Framework explanation */}
            <Card className="border-primary/20">
              <CardContent className="pt-6">
                <h3 className="font-serif font-semibold text-foreground mb-3">FIA 4-Level Thinking Framework</h3>
                <div className="grid sm:grid-cols-2 gap-3">
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs font-semibold text-primary mb-1">Level 1 — Observation</p>
                    <p className="text-xs text-muted-foreground">"What does the table show?"</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs font-semibold text-primary mb-1">Level 2 — Quantification</p>
                    <p className="text-xs text-muted-foreground">"What are the key numerical values?"</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs font-semibold text-primary mb-1">Level 3 — Interpretation</p>
                    <p className="text-xs text-muted-foreground">"What biological or practical meaning does this suggest?"</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs font-semibold text-primary mb-1">Level 4 — Scientific Caution</p>
                    <p className="text-xs text-muted-foreground">"What limitations should be acknowledged?"</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Analysis type */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Step 1: Select Analysis Type</CardTitle>
              </CardHeader>
              <CardContent>
                <Select value={analysisType} onValueChange={setAnalysisType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose the type of statistical output" />
                  </SelectTrigger>
                  <SelectContent>
                    {ANALYSIS_TYPES.map((t) => (
                      <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            {/* Data input */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Step 2: Input Your Data</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Tabs value={inputMethod} onValueChange={(v) => setInputMethod(v as "paste" | "csv")}>
                  <TabsList className="grid grid-cols-2 w-full max-w-xs">
                    <TabsTrigger value="paste" className="gap-1.5">
                      <ClipboardPaste className="w-3.5 h-3.5" /> Paste Table
                    </TabsTrigger>
                    <TabsTrigger value="csv" className="gap-1.5">
                      <Upload className="w-3.5 h-3.5" /> Upload CSV
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="paste" className="mt-4">
                    <Textarea
                      value={pastedData}
                      onChange={(e) => setPastedData(e.target.value)}
                      placeholder="Paste your statistical output table here (e.g., ANOVA table, correlation matrix, PCA loadings)..."
                      rows={10}
                      className="font-mono text-xs"
                    />
                  </TabsContent>

                  <TabsContent value="csv" className="mt-4">
                    <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                      <FileText className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                      <p className="text-sm text-muted-foreground mb-3">Upload a CSV file with your statistical output</p>
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        className="text-sm"
                      />
                    </div>
                    {pastedData && inputMethod === "csv" && (
                      <div className="mt-3 p-3 bg-muted rounded-lg">
                        <p className="text-xs font-medium text-foreground mb-1">Preview:</p>
                        <pre className="text-[10px] text-muted-foreground overflow-auto max-h-32">
                          {pastedData.split("\n").slice(0, 5).join("\n")}
                        </pre>
                      </div>
                    )}
                  </TabsContent>
                </Tabs>

                <Button onClick={handleSubmit} disabled={isSubmitting || !analysisType || !pastedData.trim()} className="w-full gap-2">
                  {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
                  Generate Interpretation Questions
                </Button>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-serif text-xl font-bold text-foreground">
                  Interpreting: {ANALYSIS_TYPES.find((t) => t.value === analysisType)?.label}
                </h2>
                <p className="text-sm text-muted-foreground">
                  The AI will guide you through the 4-level interpretation framework. Answer each question in your own words.
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={handleReset}>
                New Analysis
              </Button>
            </div>

            {/* Show data preview */}
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs font-medium text-muted-foreground mb-1">Your data:</p>
                <pre className="text-[10px] text-muted-foreground overflow-auto max-h-24 bg-muted p-2 rounded font-mono">
                  {pastedData.split("\n").slice(0, 8).join("\n")}
                  {pastedData.split("\n").length > 8 && "\n..."}
                </pre>
              </CardContent>
            </Card>

            {/* AI Chat Panel in interpret mode */}
            <div className="h-[600px]">
              <RWSChatPanel
                defaultMode="interpret"
                showModeSelector={false}
                context={{
                  track: profile?.academic_track || undefined,
                  discipline: profile?.discipline || undefined,
                  stage: profile?.current_research_stage || undefined,
                  analysis_type: analysisType,
                  data_summary: dataSummary,
                }}
                placeholder="Type your interpretation or answer the guided questions..."
              />
            </div>
          </div>
        )}
      </section>
    </Layout>
  );
}
