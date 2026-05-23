import { useState, useCallback, useMemo, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Upload, Loader2, FileSpreadsheet, X, Check } from "lucide-react";
import * as XLSX from "xlsx";
import { uploadPreview } from "@/lib/geneticsUploadApi";

export type AnalysisType =
  | "descriptive"
  | "oneway"
  | "twoway"
  | "rcbd_factorial"
  | "splitplot"
  | "multitrait";

export type MultiTraitDesign = "oneway" | "oneway_rcbd" | "twoway" | "rcbd_factorial" | "splitplot";

export interface DatasetStatus {
  filename: string | null;
  rows: number;
  cols: number;
  error: string | null;
  datasetToken?: string | null;
}

interface VivaSenseFormProps {
  onSubmit: (analysisType: AnalysisType, formData: FormData) => Promise<void>;
  isLoading: boolean;
  retryMessage?: string | null;
  onDatasetChange?: (status: DatasetStatus) => void;
}

interface ColumnInfo {
  name: string;
  isNumeric: boolean;
}

const ANALYSIS_OPTIONS: { value: AnalysisType; label: string }[] = [
  { value: "descriptive", label: "Descriptive Statistics" },
  { value: "oneway", label: "One-way ANOVA" },
  { value: "twoway", label: "Two-way ANOVA" },
  { value: "rcbd_factorial", label: "RCBD Factorial" },
  { value: "splitplot", label: "Split-plot ANOVA" },
  { value: "multitrait", label: "Multi-trait Analysis" },
];

const MULTITRAIT_DESIGN_OPTIONS: { value: MultiTraitDesign; label: string }[] = [
  { value: "oneway", label: "One-way ANOVA" },
  { value: "oneway_rcbd", label: "RCBD One-way ANOVA" },
  { value: "twoway", label: "Two-way ANOVA" },
  { value: "rcbd_factorial", label: "RCBD Factorial" },
  { value: "splitplot", label: "Split-plot ANOVA" },
];

export function VivaSenseForm({ onSubmit, isLoading, retryMessage, onDatasetChange }: VivaSenseFormProps) {
  const [analysisType, setAnalysisType] = useState<AnalysisType | "">("");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [datasetToken, setDatasetToken] = useState<string | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);

  // Dynamic field selections
  const [trait, setTrait] = useState("");
  const [factor, setFactor] = useState("");
  const [factorA, setFactorA] = useState("");
  const [factorB, setFactorB] = useState("");
  const [factorC, setFactorC] = useState("");
  const [block, setBlock] = useState("");
  const [mainPlot, setMainPlot] = useState("");
  const [subPlot, setSubPlot] = useState("");
  const [alpha, setAlpha] = useState("0.05");
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [byColumn, setByColumn] = useState("");
  const [multiTraitDesign, setMultiTraitDesign] = useState<MultiTraitDesign | "">("");
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);

  const allColumnNames = useMemo(() => columns.map((c) => c.name), [columns]);
  const numericColumns = useMemo(() => columns.filter((c) => c.isNumeric), [columns]);

  // Columns assigned to design roles — exclude from trait selection
  const designColumns = useMemo(() => {
    const set = new Set<string>();
    if (factor) set.add(factor);
    if (factorA) set.add(factorA);
    if (factorB) set.add(factorB);
    if (factorC) set.add(factorC);
    if (block) set.add(block);
    if (mainPlot) set.add(mainPlot);
    if (subPlot) set.add(subPlot);
    return set;
  }, [factor, factorA, factorB, factorC, block, mainPlot, subPlot]);

  // Remove selected traits that have been assigned to a design role
  useEffect(() => {
    setSelectedTraits((prev) => prev.filter((t) => !designColumns.has(t)));
  }, [designColumns]);

  const availableTraitColumns = useMemo(
    () => numericColumns.filter((c) => !designColumns.has(c.name)),
    [numericColumns, designColumns]
  );

  const isSupportedFile = (f: File) => {
    const name = f.name.toLowerCase();
    return name.endsWith(".xlsx") || name.endsWith(".csv");
  };

  const isNumericLike = (value: unknown) => {
    if (typeof value === "number") return Number.isFinite(value);
    if (typeof value !== "string") return false;
    const cleaned = value.trim().replace(/,/g, "");
    if (!cleaned) return false;
    const numericPattern = /^[+-]?(\d+(\.\d+)?|\.\d+)(e[+-]?\d+)?$/i;
    return numericPattern.test(cleaned);
  };

  const parseFile = useCallback(async (nextFile: File) => {
    setParseError(null);
    setDatasetToken(null);
    setIsPreviewing(true);
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
        const isNumeric = values.length > 0 && values.every(isNumericLike);
        return { name: header, isNumeric };
      });

      setColumns(columnInfo);
      resetFieldSelections();
      const preview = await uploadPreview(nextFile);
      if (!preview.dataset_token) {
        throw new Error("Upload preview did not return a dataset token. Please try uploading the file again.");
      }
      setDatasetToken(preview.dataset_token);
      onDatasetChange?.({
        filename: nextFile.name,
        rows: preview.n_rows ?? dataRows.length,
        cols: preview.n_columns ?? headers.length,
        error: null,
        datasetToken: preview.dataset_token,
      });
    } catch (error) {
      console.error("Error parsing file:", error);
      setColumns([]);
      setDatasetToken(null);
      resetFieldSelections();
      const msg = error instanceof Error ? error.message : "Could not read that file.";
      setParseError(msg);
      onDatasetChange?.({ filename: nextFile.name, rows: 0, cols: 0, error: msg });
    } finally {
      setIsPreviewing(false);
    }
  }, [onDatasetChange]);

  const resetFieldSelections = () => {
    setTrait("");
    setFactor("");
    setFactorA("");
    setFactorB("");
    setFactorC("");
    setBlock("");
    setMainPlot("");
    setSubPlot("");
    setSelectedColumns([]);
    setByColumn("");
    setMultiTraitDesign("");
    setSelectedTraits([]);
  };

  const toggleTrait = useCallback((name: string) => {
    setSelectedTraits((prev) =>
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
        const msg = "Unsupported file type. Please upload a .xlsx or .csv file.";
        setParseError(msg);
        setFileInputKey((k) => k + 1);
        onDatasetChange?.({ filename: selectedFile.name, rows: 0, cols: 0, error: msg });
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
    setDatasetToken(null);
    resetFieldSelections();
    setParseError(null);
    setFileInputKey((k) => k + 1);
    onDatasetChange?.({ filename: null, rows: 0, cols: 0, error: null });
  }, [onDatasetChange]);

  const toggleColumn = useCallback((name: string) => {
    setSelectedColumns((prev) =>
      prev.includes(name) ? prev.filter((v) => v !== name) : [...prev, name]
    );
  }, []);

  const isMultitraitDesignValid = useMemo(() => {
    if (!multiTraitDesign) return false;
    if (multiTraitDesign === "oneway") return !!factor;
    if (multiTraitDesign === "oneway_rcbd") return !!block && !!factor; // Factor C is optional
    if (multiTraitDesign === "twoway") return !!factorA && !!factorB; // Factor C is optional
    if (multiTraitDesign === "rcbd_factorial") return !!block && !!factorA && !!factorB; // Factor C is optional
    if (multiTraitDesign === "splitplot") return !!block && !!mainPlot && !!subPlot;
    return false;
  }, [multiTraitDesign, factor, factorA, factorB, factorC, block, mainPlot, subPlot]);

  const isFormValid = useMemo(() => {
    if (!file || !datasetToken || !analysisType) return false;
    if (analysisType === "multitrait") {
      return isMultitraitDesignValid && selectedTraits.length >= 2;
    }
    if (analysisType === "descriptive") return selectedColumns.length > 0;
    if (analysisType === "oneway") return !!factor && !!trait;
    if (analysisType === "twoway") return !!factorA && !!factorB && !!trait; // Factor C is optional
    if (analysisType === "rcbd_factorial") return !!block && !!factorA && !!factorB && !!trait; // Factor C is optional
    if (analysisType === "splitplot") return !!block && !!mainPlot && !!subPlot && !!trait;
    return false;
  }, [file, datasetToken, analysisType, selectedColumns, factor, trait, factorA, factorB, factorC, block, mainPlot, subPlot, isMultitraitDesignValid, selectedTraits]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !analysisType || !isFormValid) return;
    if (!datasetToken) return;

    const fd = new FormData();
    fd.append("file", file);
    fd.append("dataset_token", datasetToken);

    if (analysisType === "multitrait") {
      if (!multiTraitDesign) return;
      fd.append("traits", selectedTraits.join(","));
      fd.append("alpha", alpha);
      fd.append("design", multiTraitDesign);
      if (multiTraitDesign === "oneway") {
        fd.append("factor", factor);
      } else if (multiTraitDesign === "oneway_rcbd") {
        fd.append("block", block);
        fd.append("treatment", factor);
      } else if (multiTraitDesign === "twoway") {
        fd.append("factor_a", factorA);
        fd.append("factor_b", factorB);
        if (factorC && factorC !== "__none__") fd.append("factor_c", factorC);
      } else if (multiTraitDesign === "rcbd_factorial") {
        fd.append("block", block);
        fd.append("factor_a", factorA);
        fd.append("factor_b", factorB);
        if (factorC && factorC !== "__none__") fd.append("factor_c", factorC);
      } else if (multiTraitDesign === "splitplot") {
        fd.append("block", block);
        fd.append("main_plot", mainPlot);
        fd.append("sub_plot", subPlot);
      }
      await onSubmit(analysisType, fd);
      return;
    }

    if (analysisType === "descriptive") {
      selectedColumns.forEach((c) => fd.append("columns", c));
      if (byColumn) fd.append("by", byColumn);
    } else if (analysisType === "oneway") {
      fd.append("factor", factor);
      fd.append("trait", trait);
      fd.append("alpha", alpha);
    } else if (analysisType === "twoway") {
      fd.append("factor_a", factorA);
      fd.append("factor_b", factorB);
      if (factorC && factorC !== "__none__") fd.append("factor_c", factorC);
      fd.append("trait", trait);
      fd.append("alpha", alpha);
    } else if (analysisType === "rcbd_factorial") {
      fd.append("block", block);
      fd.append("factor_a", factorA);
      fd.append("factor_b", factorB);
      if (factorC && factorC !== "__none__") fd.append("factor_c", factorC);
      fd.append("trait", trait);
      fd.append("alpha", alpha);
    } else if (analysisType === "splitplot") {
      fd.append("block", block);
      fd.append("main_plot", mainPlot);
      fd.append("sub_plot", subPlot);
      fd.append("trait", trait);
      fd.append("alpha", alpha);
    }

    await onSubmit(analysisType, fd);
  };

  const renderColumnSelect = (
    label: string,
    value: string,
    onChange: (v: string) => void,
    id: string,
    filterNumeric = false,
    required = true
  ) => {
    const options = filterNumeric ? numericColumns.map((c) => c.name) : allColumnNames;
    return (
      <div className="space-y-2">
        <Label htmlFor={id} className="text-base font-medium">
          {label} {required && <span className="text-destructive">*</span>}
        </Label>
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger id={id} className="h-12">
            <SelectValue placeholder={`Select ${label.toLowerCase()}`} />
          </SelectTrigger>
          <SelectContent>
            {!required && <SelectItem value="__none__">None / Optional</SelectItem>}
            {options.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    );
  };

  return (
    <section className="py-20" id="analysis-form">
      <div className="container-wide">
        <div className="max-w-3xl mx-auto">
          <Card className="border-2 border-primary/20 shadow-lg">
            <CardContent className="p-6 lg:p-8">
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* File Upload — always visible (upload-first UX) */}
                {true && (
                  <div className="space-y-2">
                    <Label className="text-base font-medium">
                      Upload Excel or CSV file <span className="text-destructive">*</span>
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
                          Drag and drop your file here, or click to browse
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
                            {isPreviewing ? "Preparing dataset preview…" : `${columns.length} columns detected • ${numericColumns.length} numeric`}
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

                {/* Analysis Type selector — appears once dataset is loaded */}
                {file && (
                  <div className="space-y-2">
                    <Label htmlFor="analysis_type" className="text-base font-medium">
                      Analysis Type <span className="text-destructive">*</span>
                    </Label>
                    <Select
                      value={analysisType}
                      onValueChange={(v) => setAnalysisType(v as AnalysisType)}
                    >
                      <SelectTrigger id="analysis_type" className="h-12">
                        <SelectValue placeholder="Select an analysis" />
                      </SelectTrigger>
                      <SelectContent>
                        {ANALYSIS_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {/* Multi-trait design selector */}
                {analysisType === "multitrait" && (
                  <div className="space-y-2">
                    <Label htmlFor="multitrait_design" className="text-base font-medium">
                      ANOVA Design <span className="text-destructive">*</span>
                    </Label>
                    <Select
                      value={multiTraitDesign}
                      onValueChange={(v) => {
                        setMultiTraitDesign(v as MultiTraitDesign);
                        setFactor("");
                        setFactorA("");
                        setFactorB("");
                        setBlock("");
                        setMainPlot("");
                        setSubPlot("");
                        setSelectedTraits([]);
                      }}
                    >
                      <SelectTrigger id="multitrait_design" className="h-12">
                        <SelectValue placeholder="Select ANOVA design" />
                      </SelectTrigger>
                      <SelectContent>
                        {MULTITRAIT_DESIGN_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {/* Legacy duplicate file upload block — disabled (kept inline below) */}
                {false && (
                  <div className="space-y-2">
                    <Label className="text-base font-medium">
                      Upload Excel or CSV file <span className="text-destructive">*</span>
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
                          Drag and drop your file here, or click to browse
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

                {/* Dynamic fields based on analysis type */}
                {columns.length > 0 && analysisType !== "multitrait" && (
                  <>
                    {analysisType === "descriptive" && (
                      <>
                        <div className="space-y-2">
                          <Label className="text-base font-medium">
                            Columns to describe <span className="text-destructive">*</span>
                          </Label>
                          <p className="text-sm text-muted-foreground mb-3">Select numeric columns</p>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                            {numericColumns.map((col) => (
                              <button
                                key={col.name}
                                type="button"
                                onClick={() => toggleColumn(col.name)}
                                className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm text-left transition-colors ${
                                  selectedColumns.includes(col.name)
                                    ? "bg-primary text-primary-foreground border-primary"
                                    : "bg-background border-border hover:border-primary/50"
                                }`}
                              >
                                {selectedColumns.includes(col.name) && (
                                  <Check className="w-4 h-4 flex-shrink-0" />
                                )}
                                <span className="truncate">{col.name}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                        {renderColumnSelect("Group by (optional)", byColumn, setByColumn, "by_col")}
                      </>
                    )}

                    {analysisType === "oneway" && (
                      <>
                        {renderColumnSelect("Factor column", factor, setFactor, "factor_col")}
                        {renderColumnSelect("Trait (numeric)", trait, setTrait, "trait_col", true)}
                      </>
                    )}

                    {analysisType === "twoway" && (
                      <>
                        {renderColumnSelect("Factor A", factorA, setFactorA, "factor_a")}
                        {renderColumnSelect("Factor B", factorB, setFactorB, "factor_b")}
                        {renderColumnSelect("Factor C", factorC, setFactorC, "factor_c", false, false)}
                        {renderColumnSelect("Trait (numeric)", trait, setTrait, "trait_col", true)}
                      </>
                    )}

                    {analysisType === "rcbd_factorial" && (
                      <>
                        {renderColumnSelect("Block column", block, setBlock, "block_col")}
                        {renderColumnSelect("Factor A", factorA, setFactorA, "factor_a")}
                        {renderColumnSelect("Factor B", factorB, setFactorB, "factor_b")}
                        {renderColumnSelect("Factor C", factorC, setFactorC, "factor_c", false, false)}
                        {renderColumnSelect("Trait (numeric)", trait, setTrait, "trait_col", true)}
                      </>
                    )}

                    {analysisType === "splitplot" && (
                      <>
                        {renderColumnSelect("Block column", block, setBlock, "block_col")}
                        {renderColumnSelect("Main plot factor", mainPlot, setMainPlot, "main_plot")}
                        {renderColumnSelect("Sub-plot factor", subPlot, setSubPlot, "sub_plot")}
                        {renderColumnSelect("Trait (numeric)", trait, setTrait, "trait_col", true)}
                      </>
                    )}
                  </>
                )}

                {/* Multi-trait: design-specific factor fields + trait selection */}
                {columns.length > 0 && analysisType === "multitrait" && multiTraitDesign && (
                  <>
                    {multiTraitDesign === "oneway" && (
                      renderColumnSelect("Factor column", factor, setFactor, "mt_factor")
                    )}

                    {multiTraitDesign === "oneway_rcbd" && (
                      <>
                        {renderColumnSelect("Block column", block, setBlock, "mt_block")}
                        {renderColumnSelect("Factor column", factor, setFactor, "mt_factor")}
                      </>
                    )}

                    {multiTraitDesign === "twoway" && (
                      <>
                        {renderColumnSelect("Factor A", factorA, setFactorA, "mt_factor_a")}
                        {renderColumnSelect("Factor B", factorB, setFactorB, "mt_factor_b")}
                        {renderColumnSelect("Factor C", factorC, setFactorC, "mt_factor_c", false, false)}
                      </>
                    )}

                    {multiTraitDesign === "rcbd_factorial" && (
                      <>
                        {renderColumnSelect("Block column", block, setBlock, "mt_block")}
                        {renderColumnSelect("Factor A", factorA, setFactorA, "mt_factor_a")}
                        {renderColumnSelect("Factor B", factorB, setFactorB, "mt_factor_b")}
                        {renderColumnSelect("Factor C", factorC, setFactorC, "mt_factor_c", false, false)}
                      </>
                    )}

                    {multiTraitDesign === "splitplot" && (
                      <>
                        {renderColumnSelect("Block column", block, setBlock, "mt_block")}
                        {renderColumnSelect("Main plot factor", mainPlot, setMainPlot, "mt_main_plot")}
                        {renderColumnSelect("Sub-plot factor", subPlot, setSubPlot, "mt_sub_plot")}
                      </>
                    )}

                    {/* Trait multi-select */}
                    <div className="space-y-2">
                      <Label className="text-base font-medium">
                        Traits to analyse <span className="text-destructive">*</span>
                      </Label>
                      <p className="text-sm text-muted-foreground mb-3">
                        Select 2–15 numeric columns ({selectedTraits.length} selected
                        {availableTraitColumns.length < numericColumns.length &&
                          ` · ${numericColumns.length - availableTraitColumns.length} column${numericColumns.length - availableTraitColumns.length > 1 ? 's' : ''} assigned to design`}
                        )
                      </p>
                      {availableTraitColumns.length < 2 ? (
                        <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
                          <p className="text-sm text-destructive font-medium">
                            Not enough numeric columns remaining for multi-trait analysis.
                          </p>
                          <p className="text-sm text-destructive/80 mt-1">
                            You have {availableTraitColumns.length} numeric column{availableTraitColumns.length !== 1 ? 's' : ''} available after assigning design columns.
                            Multi-trait analysis requires at least 2. Please use a dataset with more numeric columns, or choose a simpler design that uses fewer design columns.
                          </p>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {availableTraitColumns.map((col) => (
                            <button
                              key={col.name}
                              type="button"
                              onClick={() => toggleTrait(col.name)}
                              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm text-left transition-colors ${
                                selectedTraits.includes(col.name)
                                  ? "bg-primary text-primary-foreground border-primary"
                                  : "bg-background border-border hover:border-primary/50"
                              }`}
                            >
                              {selectedTraits.includes(col.name) && (
                                <Check className="w-4 h-4 flex-shrink-0" />
                              )}
                              <span className="truncate">{col.name}</span>
                            </button>
                          ))}
                        </div>
                      )}
                      {selectedTraits.length > 0 && selectedTraits.length < 2 && (
                        <p className="text-sm text-destructive">Select at least 2 traits</p>
                      )}
                    </div>
                  </>
                )}

                {/* Alpha (for ANOVA types) */}
                {columns.length > 0 && analysisType && analysisType !== "descriptive" && (analysisType !== "multitrait" || multiTraitDesign) && (
                      <div className="space-y-2">
                        <Label htmlFor="alpha" className="text-base font-medium">
                          Significance level (α)
                        </Label>
                        <Select value={alpha} onValueChange={setAlpha}>
                          <SelectTrigger id="alpha" className="h-12">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="0.01">0.01</SelectItem>
                            <SelectItem value="0.05">0.05</SelectItem>
                            <SelectItem value="0.10">0.10</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                )}

                {/* Retry Message */}
                {retryMessage && (
                  <div className="p-4 bg-accent border border-border rounded-lg text-center">
                    <p className="text-accent-foreground font-medium">{retryMessage}</p>
                  </div>
                )}

                {/* Submit Button */}
                <Button
                  type="submit"
                  size="lg"
                  className="w-full text-white hover:opacity-90"
                  style={{ backgroundColor: "#1B5E20" }}
                  disabled={!isFormValid || isLoading}
                >
                  {isLoading || isPreviewing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      {isPreviewing ? "Preparing Dataset..." : retryMessage ? "Retrying..." : "Running Analysis..."}
                    </>
                  ) : (
                    "Run Analysis"
                  )}
                </Button>
                {!isFormValid && !isLoading && (
                  <p className="text-xs text-muted-foreground text-center mt-2">
                    {!file
                      ? "Upload a dataset to begin."
                      : !datasetToken
                      ? "Preparing dataset preview. Please wait."
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
