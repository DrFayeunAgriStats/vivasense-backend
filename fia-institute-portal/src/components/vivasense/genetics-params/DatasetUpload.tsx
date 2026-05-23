import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Upload, Loader2, FileSpreadsheet, AlertTriangle, CheckCircle2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { uploadPreview, fileToBase64 } from "@/lib/geneticsUploadApi";
import type { DatasetContext, UploadPreviewResponse } from "@/types/geneticsUpload";

interface Props {
  onDatasetReady: (ctx: DatasetContext) => void;
  datasetContext: DatasetContext | null;
}

export function DatasetUpload({ onDatasetReady, datasetContext }: Props) {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<UploadPreviewResponse | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);

  const [genotypeCol, setGenotypeCol] = useState("");
  const [repCol, setRepCol] = useState("");
  const [envCol, setEnvCol] = useState("");
  const [mode, setMode] = useState<"single" | "multi">("single");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setPreview(null);
    }
  };

  const handlePreview = async () => {
    if (!file) return;
    setIsPreviewing(true);
    setPreview(null);
    try {
      const res = await uploadPreview(file);
      setPreview(res);
      setGenotypeCol(res.detected_columns.genotype?.column ?? "");
      setRepCol(res.detected_columns.rep?.column ?? "");
      setEnvCol(res.detected_columns.environment?.column ?? "");
      setMode(res.mode_suggestion);
    } catch (err: any) {
      toast({ title: "Preview failed", description: err.message, variant: "destructive" });
    } finally {
      setIsPreviewing(false);
    }
  };

  const handleConfirmMapping = async () => {
    if (!file || !preview || !genotypeCol || !repCol) return;
    try {
      const base64 = await fileToBase64(file);
      const ext = file.name.split(".").pop()?.toLowerCase() ?? "csv";
      const ctx: DatasetContext = {
        file,
        base64Content: base64,
        fileType: ext as "csv" | "xlsx" | "xls",
        genotypeColumn: genotypeCol,
        repColumn: repCol,
        environmentColumn: envCol || null,
        availableTraitColumns: preview.detected_columns.traits,
        mode,
        datasetToken: preview.dataset_token ?? null,
      };
      onDatasetReady(ctx);
      toast({ title: "Dataset ready", description: "You can now run analysis in any module." });
    } catch (err: any) {
      toast({ title: "Error preparing dataset", description: err.message, variant: "destructive" });
    }
  };

  // If dataset already loaded, show compact summary
  if (datasetContext) {
    return (
      <Card className="border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-900/10">
        <CardContent className="py-4 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-600" />
            <div>
              <p className="text-sm font-medium text-foreground">
                Dataset loaded: <span className="text-emerald-700 dark:text-emerald-400">{datasetContext.file.name}</span>
              </p>
              <p className="text-xs text-muted-foreground">
                {datasetContext.availableTraitColumns.length} traits available · {datasetContext.mode} mode
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              onDatasetReady(null as any);
              setFile(null);
              setPreview(null);
            }}
          >
            Upload New Dataset
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            Upload Dataset
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileChange} className="flex-1" />
            <Button onClick={handlePreview} disabled={!file || isPreviewing} className="gap-2">
              {isPreviewing ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileSpreadsheet className="h-4 w-4" />}
              Preview
            </Button>
          </div>
          {file && <p className="text-xs text-muted-foreground">Selected: <span className="font-medium">{file.name}</span></p>}
        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Column Mapping</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex flex-wrap gap-2 text-sm">
              <Badge variant="secondary">{preview.n_rows} rows</Badge>
              <Badge variant="secondary">{preview.n_columns} columns</Badge>
              <Badge variant="outline">Suggested: {preview.mode_suggestion}</Badge>
            </div>

            {preview.warnings.length > 0 && (
              <div className="rounded-md border border-yellow-300 bg-yellow-50 p-3 text-sm text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200 dark:border-yellow-700">
                <div className="flex items-center gap-2 mb-1 font-medium">
                  <AlertTriangle className="h-4 w-4" /> Warnings
                </div>
                <ul className="list-disc pl-5 space-y-0.5">
                  {preview.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Genotype Column</Label>
                <Select value={genotypeCol} onValueChange={setGenotypeCol}>
                  <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                  <SelectContent>
                    {preview.column_names.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Rep Column</Label>
                <Select value={repCol} onValueChange={setRepCol}>
                  <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                  <SelectContent>
                    {preview.column_names.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Environment Column</Label>
                <Select value={envCol || "__none__"} onValueChange={(v) => setEnvCol(v === "__none__" ? "" : v)}>
                  <SelectTrigger><SelectValue placeholder="None" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None</SelectItem>
                    {preview.column_names.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Mode</Label>
                <Select value={mode} onValueChange={(v) => setMode(v as "single" | "multi")}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="single">Single Environment</SelectItem>
                    <SelectItem value="multi">Multi-Environment</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Data preview */}
            {preview.data_preview.length > 0 && (
              <div className="overflow-x-auto">
                <p className="text-sm font-medium mb-2">Data Preview (first rows)</p>
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr>
                      {preview.column_names.map((c) => (
                        <th key={c} className="border border-border px-2 py-1.5 bg-muted font-medium text-left">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.data_preview.slice(0, 5).map((row, i) => (
                      <tr key={i} className="even:bg-muted/30">
                        {preview.column_names.map((c) => (
                          <td key={c} className="border border-border px-2 py-1">{String(row[c] ?? "")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <Button
              onClick={handleConfirmMapping}
              disabled={!genotypeCol || !repCol}
              className="gap-2"
            >
              <CheckCircle2 className="h-4 w-4" />
              Confirm Mapping & Prepare Dataset
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
