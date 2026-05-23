import { useState, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Upload, Loader2, FileSpreadsheet, X, Dna } from "lucide-react";
import * as XLSX from "xlsx";

export type GeneticsAnalysisType =
  | "variance_components"
  | "stability"
  | "ammi"
  | "gge"
  | "correlations"
  | "multivariate"
  | "molecular"
  | "regression";

interface Props {
  onSubmit: (analysisType: GeneticsAnalysisType, formData: FormData) => Promise<void>;
  isLoading: boolean;
  retryMessage?: string | null;
}

interface ColumnInfo {
  name: string;
  isNumeric: boolean;
}

const GENETICS_ANALYSIS_OPTIONS: { value: GeneticsAnalysisType; label: string; description: string }[] = [
  { value: "variance_components", label: "Variance Components & Heritability", description: "σ²g, H², GA, GCV, PCV" },
  { value: "stability", label: "Stability Analysis", description: "Eberhart-Russell, bi, S²di, ASV" },
  { value: "ammi", label: "AMMI Analysis", description: "AMMI model + biplot" },
  { value: "gge", label: "GGE Biplot", description: "Which-won-where analysis" },
  { value: "correlations", label: "Trait Correlations & Path Analysis", description: "Correlations, path coefficients, selection index" },
  { value: "multivariate", label: "Multivariate Analysis", description: "PCA, clustering, dendrogram" },
  { value: "molecular", label: "Molecular Marker Analysis", description: "Jaccard, Dice, diversity indices" },
  { value: "regression", label: "Multiple Regression", description: "Stepwise regression, VIF, R²" },
];

const isNumericLike = (value: unknown) => {
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value !== "string") return false;
  const cleaned = value.trim().replace(/,/g, "");
  if (!cleaned) return false;
  return /^[+-]?(\d+(\.\d+)?|\.\d+)(e[+-]?\d+)?$/i.test(cleaned);
};

export function VivaSenseGeneticsForm({ onSubmit, isLoading, retryMessage }: Props) {
  const [analysisType, setAnalysisType] = useState<GeneticsAnalysisType | "">("");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);

  // Column assignments
  const [genotype, setGenotype] = useState("");
  const [environment, setEnvironment] = useState("");
  const [block, setBlock] = useState("");
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [alpha, setAlpha] = useState("0.05");

  // Regression-specific
  const [responseVar, setResponseVar] = useState("");
  const [selectedPredictors, setSelectedPredictors] = useState<string[]>([]);

  const allColumnNames = useMemo(() => columns.map((c) => c.name), [columns]);
  const numericColumns = useMemo(() => columns.filter((c) => c.isNumeric), [columns]);

  const designColumns = useMemo(() => {
    const set = new Set<string>();
    if (genotype) set.add(genotype);
    if (environment) set.add(environment);
    if (block) set.add(block);
    return set;
  }, [genotype, environment, block]);

  const availableTraitColumns = useMemo(
    () => numericColumns.filter((c) => !designColumns.has(c.name)),
    [numericColumns, designColumns]
  );

  // For regression: available predictors exclude the response variable
  const availablePredictorColumns = useMemo(
    () => availableTraitColumns.filter((c) => c.name !== responseVar),
    [availableTraitColumns, responseVar]
  );

  const isSupportedFile = (f: File) => {
    const name = f.name.toLowerCase();
    return name.endsWith(".xlsx") || name.endsWith(".csv");
  };

  const parseFile = useCallback(async (nextFile: File) => {
    setParseError(null);
    try {
      const lowerName = nextFile.name.toLowerCase();
      let workbook: XLSX.WorkBook;
      if (lowerName.endsWith(".csv")) {
        const text = await nextFile.text();
        workbook = XLSX.read(text, { type: "string" });
      } else {
        const buffer = await nextFile.arrayBuffer();
        workbook = XLSX.read(buffer, { type: "array" });
      }
      const firstSheetName = workbook.SheetNames[0];
      if (!firstSheetName) throw new Error("No sheets found in the uploaded file.");
      const firstSheet = workbook.Sheets[firstSheetName];
      const jsonData = XLSX.utils.sheet_to_json(firstSheet, { header: 1, defval: "" }) as unknown[][];
      if (jsonData.length < 2) throw new Error("File must have at least a header row and one data row.");

      const rawHeaders = Array.isArray(jsonData[0]) ? jsonData[0] : [];
      const headers = rawHeaders.map((h, idx) => {
        const label = String(h ?? "").trim();
        return label.length ? label : `Column ${idx + 1}`;
      });
      const dataRows = jsonData.slice(1);
      const columnInfo: ColumnInfo[] = headers.map((header, index) => {
        const values = dataRows
          .map((row) => (Array.isArray(row) ? row[index] : undefined))
          .filter((v) => {
            if (v === undefined || v === null) return false;
            if (typeof v === "string") {
              const t = v.trim();
              if (!t) return false;
              if (["na", "n/a", "nan", "null", "-", "--"].includes(t.toLowerCase())) return false;
            }
            return v !== "";
          });
        const isNum = values.length > 0 && values.every(isNumericLike);
        return { name: header, isNumeric: isNum };
      });

      setColumns(columnInfo);
      resetFieldSelections();
    } catch (error) {
      console.error("Error parsing file:", error);
      setColumns([]);
      resetFieldSelections();
      setParseError(error instanceof Error ? error.message : "Could not read that file.");
    }
  }, []);

  const resetFieldSelections = () => {
    setGenotype("");
    setEnvironment("");
    setBlock("");
    setSelectedTraits([]);
    setResponseVar("");
    setSelectedPredictors([]);
  };

  const toggleTrait = useCallback((name: string) => {
    setSelectedTraits((prev) =>
      prev.includes(name) ? prev.filter((v) => v !== name) : prev.length < 15 ? [...prev, name] : prev
    );
  }, []);

  const togglePredictor = useCallback((name: string) => {
    setSelectedPredictors((prev) =>
      prev.includes(name) ? prev.filter((v) => v !== name) : prev.length < 15 ? [...prev, name] : prev
    );
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (!selectedFile) return;
      if (!isSupportedFile(selectedFile)) {
        setFile(null);
        setColumns([]);
        resetFieldSelections();
        setParseError("Unsupported file type. Please upload a .xlsx or .csv file.");
        setFileInputKey((k) => k + 1);
        return;
      }
      setFile(selectedFile);
      parseFile(selectedFile);
    },
    [parseFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      if (!droppedFile) return;
      if (!isSupportedFile(droppedFile)) {
        setParseError("Unsupported file type. Please upload a .xlsx or .csv file.");
        return;
      }
      setFile(droppedFile);
      parseFile(droppedFile);
    },
    [parseFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const removeFile = useCallback(() => {
    setFile(null);
    setColumns([]);
    resetFieldSelections();
    setParseError(null);
    setFileInputKey((k) => k + 1);
  }, []);

  const isRegression = analysisType === "regression";

  // Which fields are needed per analysis type
  const needsEnvironment = ["variance_components", "stability", "ammi", "gge"].includes(analysisType as string);
  const needsBlock = ["variance_components", "stability", "ammi", "gge"].includes(analysisType as string);
  const needsTraits = !isRegression && ["variance_components", "stability", "ammi", "gge", "correlations", "multivariate"].includes(analysisType as string);
  const needsGenotype = analysisType !== "" && !isRegression;

  const isFormValid = useMemo(() => {
    if (!file || !analysisType) return false;
    if (isRegression) {
      return responseVar !== "" && selectedPredictors.length >= 1;
    }
    if (needsGenotype && !genotype) return false;
    if (needsEnvironment && !environment) return false;
    if (needsBlock && !block) return false;
    if (needsTraits && selectedTraits.length < 1) return false;
    if (analysisType === "correlations" && selectedTraits.length < 2) return false;
    if (analysisType === "multivariate" && selectedTraits.length < 2) return false;
    if (analysisType === "molecular" && !genotype) return false;
    return true;
  }, [file, analysisType, genotype, environment, block, selectedTraits, needsGenotype, needsEnvironment, needsTraits, isRegression, responseVar, selectedPredictors]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !analysisType || !isFormValid) return;

    const fd = new FormData();
    fd.append("file", file);
    fd.append("alpha", alpha);

    if (isRegression) {
      // Regression-specific payload
      fd.append("response_col", responseVar);
      fd.append("predictor_cols", selectedPredictors.join(","));
      console.log("[Genetics] Submitting regression:", {
        analysisType,
        response_col: responseVar,
        predictor_cols: selectedPredictors.join(","),
        alpha,
      });
    } else {
      fd.append("genotype", genotype);
      if (environment) fd.append("location", environment);
      if (block) fd.append("rep", block);
      if (selectedTraits.length > 0) fd.append("traits", selectedTraits.join(","));
      console.log("[Genetics] Submitting:", {
        analysisType,
        genotype,
        location: environment,
        rep: block,
        traits: selectedTraits.join(","),
        alpha,
      });
    }

    await onSubmit(analysisType, fd);
  };

  const renderColumnSelect = (
    label: string,
    value: string,
    onChange: (v: string) => void,
    id: string,
    required = true
  ) => (
    <div className="space-y-2">
      <Label htmlFor={id} className="text-base font-medium">
        {label} {required && <span className="text-destructive">*</span>}
      </Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id={id} className="h-12">
          <SelectValue placeholder={`Select ${label.toLowerCase()}`} />
        </SelectTrigger>
        <SelectContent>
          {allColumnNames.map((name) => (
            <SelectItem key={name} value={name}>
              {name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );

  return (
    <section className="py-20" id="genetics-form">
      <div className="container-wide">
        <div className="max-w-3xl mx-auto">
          <Card className="border-2 border-primary/20 shadow-lg">
            <CardHeader className="text-center pb-2">
              <CardTitle className="font-serif text-2xl lg:text-3xl flex items-center justify-center gap-3">
                <Dna className="w-7 h-7 text-primary" />
                Plant Breeding Genetics Analysis
              </CardTitle>
              <CardDescription className="text-base">
                Upload multilocational trial data for advanced genetic analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6 lg:p-8">
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Analysis Type */}
                <div className="space-y-2">
                  <Label htmlFor="genetics_type" className="text-base font-medium">
                    Analysis type <span className="text-destructive">*</span>
                  </Label>
                  <Select
                    value={analysisType}
                    onValueChange={(v) => {
                      setAnalysisType(v as GeneticsAnalysisType);
                      resetFieldSelections();
                    }}
                  >
                    <SelectTrigger id="genetics_type" className="h-12">
                      <SelectValue placeholder="Select analysis type" />
                    </SelectTrigger>
                    <SelectContent>
                      {GENETICS_ANALYSIS_OPTIONS.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>
                          <div className="flex flex-col">
                            <span>{opt.label}</span>
                            <span className="text-xs text-muted-foreground">{opt.description}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* File Upload */}
                {analysisType && (
                  <div className="space-y-2">
                    <Label className="text-base font-medium">
                      Upload CSV / Excel <span className="text-destructive">*</span>
                    </Label>
                    {!file ? (
                      <div
                        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                          isDragOver
                            ? "border-primary bg-primary/5"
                            : "border-border hover:border-primary/50"
                        }`}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                      >
                        <Upload className="w-10 h-10 text-muted-foreground mx-auto mb-4" />
                        <p className="text-muted-foreground mb-2">
                          Drag and drop your {isRegression ? "data file" : "multilocational trial data"}
                        </p>
                        <p className="text-sm text-muted-foreground/70">
                          Accepted formats: .xlsx, .csv
                        </p>
                        <Input
                          key={fileInputKey}
                          type="file"
                          accept=".xlsx,.csv"
                          onChange={handleFileChange}
                          className="absolute inset-0 opacity-0 cursor-pointer"
                        />
                      </div>
                    ) : (
                      <div className="flex items-center gap-4 p-4 bg-primary/5 rounded-lg border border-primary/20">
                        <FileSpreadsheet className="w-10 h-10 text-primary" />
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-foreground truncate">{file.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {columns.length} columns detected • {numericColumns.length} numeric
                          </p>
                        </div>
                        <Button type="button" variant="ghost" size="icon" onClick={removeFile} className="flex-shrink-0">
                          <X className="w-5 h-5" />
                        </Button>
                      </div>
                    )}
                    {parseError && (
                      <p className="text-sm text-destructive" role="alert">{parseError}</p>
                    )}
                  </div>
                )}

                {/* Column Mapping */}
                {columns.length > 0 && (
                  <>
                    {/* Standard genetics fields */}
                    {needsGenotype && renderColumnSelect("Genotype column", genotype, setGenotype, "gen_col")}
                    {needsEnvironment && renderColumnSelect("Environment / Location column", environment, setEnvironment, "env_col")}
                    {needsBlock && renderColumnSelect("Block / Rep column", block, setBlock, "block_col", true)}

                    {/* Regression: Response Variable + Predictor Variables */}
                    {isRegression && (
                      <>
                        {/* Response Variable (Y) — single dropdown */}
                        <div className="space-y-2">
                          <Label htmlFor="response_var" className="text-base font-medium">
                            Response Variable (Dependent) <span className="text-destructive">*</span>
                          </Label>
                          <Select
                            value={responseVar}
                            onValueChange={(v) => {
                              setResponseVar(v);
                              // Remove from predictors if selected
                              setSelectedPredictors((prev) => prev.filter((p) => p !== v));
                            }}
                          >
                            <SelectTrigger id="response_var" className="h-12">
                              <SelectValue placeholder="Select the trait you want to predict" />
                            </SelectTrigger>
                            <SelectContent>
                              {availableTraitColumns.map((col) => (
                                <SelectItem key={col.name} value={col.name}>
                                  {col.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Predictor Variables (X) — multi-select checkboxes */}
                        {responseVar && (
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <Label className="text-base font-medium">
                                  Predictor Variables (Independent) <span className="text-destructive">*</span>
                                </Label>
                                <p className="text-sm text-muted-foreground mt-1">
                                  Select one or more traits to use as predictors ({selectedPredictors.length} selected)
                                </p>
                              </div>
                              {availablePredictorColumns.length > 0 && (
                                <div className="flex gap-2">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setSelectedPredictors(availablePredictorColumns.map((c) => c.name).slice(0, 15))}
                                  >
                                    Select All
                                  </Button>
                                  <Button
                                    type="button"
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setSelectedPredictors([])}
                                  >
                                    Clear All
                                  </Button>
                                </div>
                              )}
                            </div>
                            {availablePredictorColumns.length < 1 ? (
                              <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
                                <p className="text-sm text-destructive font-medium">
                                  No numeric predictor columns available.
                                </p>
                              </div>
                            ) : (
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {availablePredictorColumns.map((col) => (
                                  <label
                                    key={col.name}
                                    className={`flex items-center gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
                                      selectedPredictors.includes(col.name)
                                        ? "bg-primary/10 border-primary"
                                        : "bg-background border-border hover:border-primary/40"
                                    }`}
                                  >
                                    <Checkbox
                                      checked={selectedPredictors.includes(col.name)}
                                      onCheckedChange={() => togglePredictor(col.name)}
                                    />
                                    <span className="text-sm font-medium truncate">{col.name}</span>
                                  </label>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </>
                    )}

                    {/* Standard trait selection with checkboxes (non-regression) */}
                    {needsTraits && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <Label className="text-base font-medium">
                              Select Traits to Analyze <span className="text-destructive">*</span>
                            </Label>
                            <p className="text-sm text-muted-foreground mt-1">
                              Select one or more traits to analyze ({selectedTraits.length} selected)
                            </p>
                          </div>
                          {availableTraitColumns.length > 0 && (
                            <div className="flex gap-2">
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedTraits(availableTraitColumns.map((c) => c.name).slice(0, 15))}
                              >
                                Select All
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                onClick={() => setSelectedTraits([])}
                              >
                                Clear All
                              </Button>
                            </div>
                          )}
                        </div>
                        {availableTraitColumns.length < 1 ? (
                          <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
                            <p className="text-sm text-destructive font-medium">
                              No numeric trait columns available after assigning design columns.
                            </p>
                          </div>
                        ) : (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {availableTraitColumns.map((col) => (
                              <label
                                key={col.name}
                                className={`flex items-center gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
                                  selectedTraits.includes(col.name)
                                    ? "bg-primary/10 border-primary"
                                    : "bg-background border-border hover:border-primary/40"
                                }`}
                              >
                                <Checkbox
                                  checked={selectedTraits.includes(col.name)}
                                  onCheckedChange={() => toggleTrait(col.name)}
                                />
                                <span className="text-sm font-medium truncate">{col.name}</span>
                              </label>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Molecular: just needs genotype column, markers are auto-detected */}
                    {analysisType === "molecular" && (
                      <div className="p-4 bg-muted rounded-lg">
                        <p className="text-sm text-muted-foreground">
                          <strong>Note:</strong> Marker columns (binary 0/1) will be auto-detected from your data.
                          Ensure your file has a genotype column and marker loci as separate columns.
                        </p>
                      </div>
                    )}

                    {/* Alpha */}
                    <div className="space-y-2">
                      <Label htmlFor="gen_alpha" className="text-base font-medium">
                        Significance level (α)
                      </Label>
                      <Select value={alpha} onValueChange={setAlpha}>
                        <SelectTrigger id="gen_alpha" className="h-12">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0.01">0.01</SelectItem>
                          <SelectItem value="0.05">0.05</SelectItem>
                          <SelectItem value="0.10">0.10</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </>
                )}

                {retryMessage && (
                  <div className="p-4 bg-accent border border-border rounded-lg text-center">
                    <p className="text-accent-foreground font-medium">{retryMessage}</p>
                  </div>
                )}

                <Button
                  type="submit"
                  size="lg"
                  className="w-full text-white hover:opacity-90"
                  style={{ backgroundColor: "#1B5E20" }}
                  disabled={!isFormValid || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      {retryMessage ? "Retrying..." : "Running Genetics Analysis..."}
                    </>
                  ) : (
                    <>
                      <Dna className="w-5 h-5" />
                      Run {isRegression ? "Multiple Regression" : "Genetics Analysis"}
                    </>
                  )}
                </Button>
                {!isFormValid && !isLoading && (
                  <p className="text-xs text-muted-foreground text-center mt-2">
                    {!file
                      ? "Upload a dataset to begin."
                      : "Complete required fields to run analysis."}
                  </p>
                )}
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
