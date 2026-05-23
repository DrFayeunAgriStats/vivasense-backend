import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Download, Shuffle, MapPin, Info } from "lucide-react";
import { API_BASE } from "@/services/apiConfig";
import { getVivaSenseMode, subscribeVivaSenseMode, type VivaSenseMode } from "@/lib/vivasenseGating";

type Design =
  | "crd"
  | "rcbd"
  | "latin_square"
  | "split_plot"
  | "split_split"
  | "factorial_rcbd"
  | "lattice"
  | "alpha_lattice";

const DESIGNS: Array<{ value: Design; label: string; desc: string; requiresPro: boolean }> = [
  { value: "crd", label: "CRD", desc: "Completely Randomised Design", requiresPro: false },
  { value: "rcbd", label: "RCBD", desc: "Randomised Complete Block Design", requiresPro: false },
  { value: "latin_square", label: "Latin Square", desc: "n x n Latin Square Design", requiresPro: true },
  { value: "split_plot", label: "Split Plot", desc: "Two-level nested design", requiresPro: true },
  { value: "split_split", label: "Split-Split Plot", desc: "Three-level nested design", requiresPro: true },
  { value: "factorial_rcbd", label: "Factorial RCBD", desc: "All factor combinations in blocks", requiresPro: true },
  { value: "lattice", label: "Balanced Lattice", desc: "Incomplete block - perfect square", requiresPro: true },
  { value: "alpha_lattice", label: "Alpha Lattice", desc: "Flexible incomplete block design", requiresPro: true },
];

interface FormFieldProps {
  label: string;
  helper?: string;
  required?: boolean;
  children: React.ReactNode;
}

function FormField({ label, helper, required, children }: FormFieldProps) {
  return (
    <div className="space-y-1.5">
      <Label className="block">
        {label}
        {required ? " *" : ""}
      </Label>
      {children}
      {helper ? <p className="text-xs text-muted-foreground">{helper}</p> : null}
    </div>
  );
}

interface PlotCell {
  plotId: number;
  treatment: string;
  rep: number;
  row: number;
  col: number;
  blockLabel?: string;
}

interface LayoutSummary {
  title?: string;
  n_plots?: number;
  n_treatments?: number;
  replications?: number;
  plot_width_m?: number;
  plot_length_m?: number;
  aisle_width_m?: number;
  seed?: number;
  block_size?: number;
  n_blocks_per_rep?: number;
  alpha_value?: number;
}

interface FieldLayoutResult {
  design?: Design;
  seed?: number;
  treatments?: string[];
  replications?: number;
  total_plots?: number;
  layout_matrix?: unknown[];
  plot_labels?: string[];
  timestamp?: string;
  export_data?: {
    printable_plot_numbering?: Array<Record<string, unknown>>;
    row_column_matrix?: Array<Array<Record<string, unknown>>>;
    field_map_labels?: Array<Record<string, unknown>>;
  };

  // Legacy compatibility fields
  design_type: Design;
  plot_matrix: unknown[];
  fieldbook: Array<Record<string, unknown>>;
  layout_summary: LayoutSummary;
  alpha_value: number | null;
}

export function FieldLayoutGenerator() {
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());
  const [design, setDesign] = useState<Design>("rcbd");
  const [treatmentsRaw, setTreatmentsRaw] = useState("T1, T2, T3, T4");
  const [replications, setReplications] = useState(3);
  const [plotWidth, setPlotWidth] = useState(2);
  const [plotLength, setPlotLength] = useState(3);
  const [aisleWidth, setAisleWidth] = useState(0.5);
  const [seed, setSeed] = useState(42);

  const [nTreatments, setNTreatments] = useState(4);
  const [mainTreatments, setMainTreatments] = useState("M1, M2, M3");
  const [subTreatments, setSubTreatments] = useState("S1, S2");
  const [subSubTreatments, setSubSubTreatments] = useState("SS1, SS2");
  const [factorA, setFactorA] = useState("N0, N1, N2");
  const [factorAName, setFactorAName] = useState("Nitrogen");
  const [factorB, setFactorB] = useState("V1, V2, V3");
  const [factorBName, setFactorBName] = useState("Variety");
  const [blockSize, setBlockSize] = useState(4);

  const [result, setResult] = useState<FieldLayoutResult | null>(null);
  const [layout, setLayout] = useState<{ design: Design; cells: PlotCell[]; nRows: number; nCols: number } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const isPro = mode === "pro";
  const selectedDesign = useMemo(() => DESIGNS.find((d) => d.value === design), [design]);
  const parseCsv = (s: string) => s.split(/[,\n]+/).map((x) => x.trim()).filter(Boolean);

  useEffect(() => {
    setMode(getVivaSenseMode());
    return subscribeVivaSenseMode(setMode);
  }, []);

  const treatments = useMemo(() => parseCsv(treatmentsRaw), [treatmentsRaw]);

  const buildPayload = () => {
    const base = {
      design_type: design,
      replications,
      plot_width_m: plotWidth,
      plot_length_m: plotLength,
      aisle_width_m: aisleWidth,
      seed,
    };

    if (design === "latin_square") {
      const labels = Array.from({ length: nTreatments }, (_, i) => `T${i + 1}`);
      return { ...base, treatments: labels, rows: nTreatments, columns: nTreatments };
    }
    if (design === "split_plot") {
      return {
        ...base,
        main_treatments: parseCsv(mainTreatments),
        sub_treatments: parseCsv(subTreatments),
      };
    }
    if (design === "split_split") {
      return {
        ...base,
        main_treatments: parseCsv(mainTreatments),
        sub_treatments: parseCsv(subTreatments),
        sub_sub_treatments: parseCsv(subSubTreatments),
      };
    }
    if (design === "factorial_rcbd") {
      return {
        ...base,
        factors: {
          [factorAName || "Factor A"]: parseCsv(factorA),
          [factorBName || "Factor B"]: parseCsv(factorB),
        },
      };
    }
    if (design === "lattice") {
      const k = Math.sqrt(nTreatments);
      const reps = k + 1;
      const labels = Array.from({ length: nTreatments }, (_, i) => `T${i + 1}`);
      return { ...base, treatments: labels, replications: reps };
    }
    if (design === "alpha_lattice") {
      const labels = Array.from({ length: nTreatments }, (_, i) => `G${i + 1}`);
      return { ...base, treatments: labels, block_size: blockSize };
    }
    return {
      ...base,
      treatments: parseCsv(treatmentsRaw),
    };
  };

  const flattenToGrid = (apiResult: FieldLayoutResult): Array<Array<{ plot_id: number; label: string; rep: number; blockLabel?: string }>> => {
    const designType = apiResult.design_type;
    const matrix: any = apiResult.plot_matrix;
    const fieldbook: any[] = apiResult.fieldbook ?? [];
    if (!matrix || !fieldbook.length) return [];

    const fbLookup = (pid: number) => fieldbook.find((f: any) => Number(f.plot_id) === Number(pid));

    if (designType === "lattice" || designType === "alpha_lattice") {
      const rows: any[] = [];
      (matrix as any[]).forEach((repBlocks: any[], repIdx: number) => {
        (repBlocks ?? []).forEach((block: any, blockIdx: number) => {
          const plots = block.plots ?? [];
          const blockLabel = `Rep ${repIdx + 1} · Block ${block.block_id ?? blockIdx + 1}`;
          const blockRow = plots.map((p: any) => {
            const fb = fbLookup(p.plot_id);
            return {
              plot_id: p.plot_id,
              label: String(p.treatment ?? fb?.treatment ?? "?"),
              rep: Number(fb?.rep ?? repIdx + 1),
              blockLabel,
            };
          });
          rows.push(blockRow);
        });
      });
      return rows;
    }

    if (designType === "split_plot") {
      const rows: any[] = [];
      (matrix as any[]).forEach((repRow: any[]) => {
        (repRow ?? []).forEach((mainPlot: any) => {
          const subRow = (mainPlot.sub_plots ?? []).map((sp: any) => ({
            plot_id: sp.plot_id,
            label: `${mainPlot.main_treatment}/${sp.sub_treatment}`,
            rep: Number(mainPlot.rep ?? fbLookup(sp.plot_id)?.rep ?? 1),
          }));
          rows.push(subRow);
        });
      });
      return rows;
    }

    if (designType === "split_split") {
      const rows: any[] = [];
      (matrix as any[]).forEach((repRow: any[]) => {
        (repRow ?? []).forEach((mainPlot: any) => {
          (mainPlot.sub_plots ?? []).forEach((sp: any) => {
            const subSubRow = (sp.sub_sub_plots ?? []).map((ssp: any) => ({
              plot_id: ssp.plot_id,
              label: `${mainPlot.main_treatment}/${sp.sub_treatment}/${ssp.sub_sub_treatment}`,
              rep: Number(fbLookup(ssp.plot_id)?.rep ?? 1),
            }));
            rows.push(subSubRow);
          });
        });
      });
      return rows;
    }

    if (designType === "factorial_rcbd") {
      const rows: any[] = [];
      (matrix as any[]).forEach((repRow: any[]) => {
        const row = (repRow ?? []).map((cell: any) => {
          const fb = fbLookup(cell.plot_id);
          return {
            plot_id: cell.plot_id,
            label: String(cell.treatment_combination ?? fb?.treatment_combination ?? cell.treatment ?? "?"),
            rep: Number(cell.rep ?? fb?.rep ?? 1),
          };
        });
        rows.push(row);
      });
      return rows;
    }

    // Default — flat 2D matrix (CRD, RCBD, Latin Square)
    return (matrix as any[]).map((row: any[]) =>
      (row ?? []).map((cell: any) => {
        const fb = fbLookup(cell.plot_id);
        return {
          plot_id: cell.plot_id,
          label: String(cell.treatment ?? cell.treatment_combination ?? fb?.treatment ?? "?"),
          rep: Number(cell.rep ?? fb?.rep ?? 1),
        };
      })
    );
  };

  const normalizeLayout = (apiResult: FieldLayoutResult) => {
    const grid = flattenToGrid(apiResult);

    if (!grid.length || !grid.some((r) => r.length)) {
      // Fallback: build a single column from fieldbook
      const cells: PlotCell[] = apiResult.fieldbook.map((row, idx) => ({
        plotId: Number(row.plot_id) || idx + 1,
        treatment: String(row.treatment ?? row.sub_treatment ?? row.sub_sub_treatment ?? row.main_treatment ?? "N/A"),
        rep: Number(row.rep) || 1,
        row: idx,
        col: 0,
      }));
      setLayout({ design: apiResult.design_type, cells, nRows: cells.length, nCols: 1 });
      return;
    }

    const nRows = grid.length;
    const nCols = Math.max(...grid.map((r) => r.length));
    const cells: PlotCell[] = [];
    grid.forEach((row, rIdx) => {
      row.forEach((cell, cIdx) => {
        cells.push({
          plotId: Number(cell.plot_id) || rIdx * nCols + cIdx + 1,
          treatment: cell.label,
          rep: Number(cell.rep) || 1,
          row: rIdx,
          col: cIdx,
          blockLabel: cell.blockLabel,
        });
      });
    });
    setLayout({ design: apiResult.design_type, cells, nRows, nCols });
  };

  const validate = (): string | null => {
    if (selectedDesign?.requiresPro && !isPro) {
      return "This design is available in VivaSense Pro only.";
    }
    if (replications < 2) return "Replications must be >= 2.";
    if (replications > 20 && design !== "lattice") return "Replications must be <= 20.";
    if (plotWidth <= 0 || plotLength <= 0) return "Plot dimensions must be positive.";
    if (aisleWidth < 0) return "Alley width cannot be negative.";

    if (design === "crd" || design === "rcbd") {
      if (treatments.length < 2) return "Please provide at least 2 treatments.";
      if (treatments.length > 50) return "Please provide no more than 50 treatments.";
      if (new Set(treatments).size !== treatments.length) return "Treatment labels must be unique.";
    }

    if (design === "alpha_lattice" && (nTreatments < 4 || blockSize < 2)) {
      return "Alpha lattice requires at least 4 treatments and block size >= 2.";
    }

    return null;
  };

  const generate = async () => {
    const v = validate();
    if (v) {
      setError(v);
      setLayout(null);
      setResult(null);
      return;
    }

    setError(null);

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/field-layout/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-VivaSense-Mode": isPro ? "pro" : "free",
        },
        body: JSON.stringify(buildPayload()),
      });

      const raw = await response.text();
      if (!raw.trim()) {
        throw new Error("Server returned an empty response.");
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(raw);
      } catch {
        throw new Error("Server returned malformed JSON.");
      }

      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        const data = parsed as { detail?: unknown };
        if (typeof data?.detail === "string") {
          detail = data.detail;
        } else if (data?.detail && typeof data.detail === "object") {
          const maybeMessage = (data.detail as { message?: unknown }).message;
          const maybeIssues = (data.detail as { issues?: unknown }).issues;
          if (typeof maybeMessage === "string") {
            detail = maybeMessage;
          }
          if (Array.isArray(maybeIssues) && maybeIssues.length > 0) {
            detail = `${detail}: ${maybeIssues.join("; ")}`;
          }
        }
        throw new Error(detail);
      }

      const data = parsed as Partial<FieldLayoutResult>;
      if (!data || typeof data !== "object") {
        throw new Error("Unexpected response format from server.");
      }

      const normalized: FieldLayoutResult = {
        design: (data.design ?? data.design_type) as Design | undefined,
        seed: typeof data.seed === "number" ? data.seed : undefined,
        treatments: Array.isArray(data.treatments) ? data.treatments : undefined,
        replications: typeof data.replications === "number" ? data.replications : undefined,
        total_plots: typeof data.total_plots === "number" ? data.total_plots : undefined,
        layout_matrix: Array.isArray(data.layout_matrix) ? data.layout_matrix : undefined,
        plot_labels: Array.isArray(data.plot_labels) ? data.plot_labels : undefined,
        timestamp: typeof data.timestamp === "string" ? data.timestamp : undefined,
        export_data: data.export_data,

        design_type: ((data.design_type ?? data.design ?? design) as Design),
        plot_matrix: (Array.isArray(data.plot_matrix)
          ? data.plot_matrix
          : (Array.isArray(data.layout_matrix) ? data.layout_matrix : [])),
        fieldbook: (Array.isArray(data.fieldbook) ? data.fieldbook : []),
        layout_summary: (data.layout_summary ?? {}),
        alpha_value: typeof data.alpha_value === "number" ? data.alpha_value : null,
      };

      setResult(normalized);
      normalizeLayout(normalized);
    } catch (err) {
      setResult(null);
      setLayout(null);
      if (err instanceof TypeError) {
        setError("Network interruption detected. Please check your connection and retry.");
      } else {
        setError(err instanceof Error ? err.message : "Unable to generate field layout.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const downloadPng = async () => {
    const svg = svgRef.current;
    if (!svg) return;
    const xml = new XMLSerializer().serializeToString(svg);
    const svg64 = btoa(unescape(encodeURIComponent(xml)));
    const img = new Image();
    img.onload = () => {
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = svg.clientWidth * scale || 800;
      canvas.height = svg.clientHeight * scale || 600;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `field-layout-${design}-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
      }, "image/png");
    };
    img.src = `data:image/svg+xml;base64,${svg64}`;
  };

  const downloadCsv = () => {
    if (!result || !layout) return;

    const csvRows: string[] = [];

    csvRows.push(`# VivaSense Field Layout Data Collection Sheet`);
    csvRows.push(
      `# Design: ${result.design_type}, Treatments: ${result.layout_summary.n_treatments ?? "N/A"}, Replications: ${result.layout_summary.replications ?? "N/A"}, Seed: ${result.layout_summary.seed ?? seed}`
    );
    csvRows.push(`# Plot size: ${plotWidth} x ${plotLength} m`);
    csvRows.push("");

    const rows = result.fieldbook;
    if (!rows.length) return;

    const columns = Object.keys(rows[0]);
    csvRows.push(columns.map((h) => `"${h}"`).join(","));
    rows.forEach((row) => {
      csvRows.push(
        columns
          .map((key) => {
            const str = String(row[key] ?? "");
            return /[",\n\r]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
          })
          .join(",")
      );
    });

    const csv = csvRows.join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `VivaSense_DataCollection_${design}_${rows.length}plots_Seed${String(seed).replace(/[^a-zA-Z0-9]/g, "")}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const treatmentUniverse = useMemo(() => {
    if (!layout) return [] as string[];
    return Array.from(new Set(layout.cells.map((c) => c.treatment)));
  }, [layout]);

  const isSplitDesign = layout?.design === "split_plot" || layout?.design === "split_split";
  const isLatticeDesign = layout?.design === "lattice" || layout?.design === "alpha_lattice";
  const cellW = isSplitDesign ? 110 : 90;
  const cellH = 64;
  const pad = 24;
  const blockLabelW = isLatticeDesign ? 110 : 0;
  const svgW = layout ? layout.nCols * cellW + pad * 2 + blockLabelW : 0;
  const svgH = layout ? layout.nRows * cellH + pad * 2 + 30 : 0;

  const colorFor = (tr: string) => {
    const idx = treatmentUniverse.indexOf(tr);
    const palette = [
      "#0A7F5A", "#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A",
      "#A5D6A7", "#C8E6C9", "#558B2F", "#7CB342", "#9CCC65", "#33691E",
      "#004D40", "#00695C", "#00796B", "#26A69A",
    ];
    return palette[Math.max(0, idx) % palette.length];
  };

  const totalArea = layout
    ? (layout.nRows * plotLength + (layout.nRows - 1) * aisleWidth) *
      (layout.nCols * plotWidth + (layout.nCols - 1) * aisleWidth)
    : 0;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-serif">
            <MapPin className="h-5 w-5 text-primary" />
            Field Layout Generator
          </CardTitle>
          <CardDescription>
            Generate randomized field layouts for classic and Pro experimental designs.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div>
            <Label className="mb-2 block">Experimental Design</Label>
            <select
              value={design}
              onChange={(e) => setDesign(e.target.value as Design)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
            >
              {DESIGNS.map((d) => (
                <option
                  key={d.value}
                  value={d.value}
                  disabled={d.requiresPro && !isPro}
                >
                  {d.label} - {d.desc}{d.requiresPro ? " [Pro]" : ""}
                </option>
              ))}
            </select>

            <div className="mt-2 flex flex-wrap gap-2">
              {DESIGNS.filter((d) => d.requiresPro).map((d) => (
                <span
                  key={`badge-${d.value}`}
                  className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium"
                  style={{
                    background: "#FAEEDA",
                    color: "#854F0B",
                    border: "0.5px solid #F0BC52",
                  }}
                >
                  {d.label} Pro
                </span>
              ))}
            </div>

            {selectedDesign?.requiresPro && !isPro && (
              <div
                style={{
                  marginTop: 8,
                  padding: "10px 14px",
                  background: "#FAEEDA",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "#854F0B",
                  border: "0.5px solid #F0BC52",
                }}
              >
                <span role="img" aria-label="locked">🔒</span> <strong>Pro feature.</strong> Upgrade to VivaSense Pro to access
                Latin square, split plot, factorial, lattice and alpha lattice designs.{" "}
                <a href="/pricing" style={{ color: "#854F0B", fontWeight: 600 }}>
                  Upgrade -&gt;
                </a>
              </div>
            )}
          </div>

          {(design === "crd" || design === "rcbd") && (
            <FormField label="Treatment Labels" helper="Enter each treatment on a new line or comma-separated" required>
              <Textarea
                value={treatmentsRaw}
                onChange={(e) => setTreatmentsRaw(e.target.value)}
                placeholder="T1, T2, T3, T4"
                rows={3}
                className="mt-1.5 font-mono text-sm"
              />
            </FormField>
          )}

          {/* LATIN SQUARE */}
          {design === "latin_square" && (
            <FormField label="Number of Treatments (n)" helper="Must equal number of rows and columns. Supported: 4 to 10." required>
              <input type="number" min={4} max={10}
                value={nTreatments} onChange={e => setNTreatments(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm" />
            </FormField>
          )}

          {/* SPLIT PLOT */}
          {design === "split_plot" && (
            <>
              <FormField label="Main Plot Treatments" helper="Enter each treatment on a new line or comma-separated" required>
                <textarea rows={3} value={mainTreatments}
                  onChange={e => setMainTreatments(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  placeholder="M1, M2, M3" />
              </FormField>
              <FormField label="Sub Plot Treatments" required>
                <textarea rows={3} value={subTreatments}
                  onChange={e => setSubTreatments(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  placeholder="S1, S2" />
              </FormField>
            </>
          )}

          {/* SPLIT-SPLIT PLOT */}
          {design === "split_split" && (
            <>
              <FormField label="Main Plot Treatments" required>
                <textarea rows={2} value={mainTreatments} onChange={e => setMainTreatments(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm" placeholder="M1, M2" />
              </FormField>
              <FormField label="Sub Plot Treatments" required>
                <textarea rows={2} value={subTreatments} onChange={e => setSubTreatments(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm" placeholder="S1, S2" />
              </FormField>
              <FormField label="Sub-Sub Plot Treatments" required>
                <textarea rows={2} value={subSubTreatments} onChange={e => setSubSubTreatments(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm" placeholder="SS1, SS2" />
              </FormField>
            </>
          )}

          {/* FACTORIAL RCBD */}
          {design === "factorial_rcbd" && (
            <>
              <FormField label="Factor A levels" helper="Comma-separated e.g. N0, N1, N2" required>
                <input type="text" value={factorA} onChange={e => setFactorA(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  placeholder="N0, N1, N2" />
              </FormField>
              <FormField label="Factor A name" required>
                <input type="text" value={factorAName} onChange={e => setFactorAName(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  placeholder="Nitrogen" />
              </FormField>
              <FormField label="Factor B levels" helper="Comma-separated e.g. V1, V2, V3" required>
                <input type="text" value={factorB} onChange={e => setFactorB(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  placeholder="V1, V2, V3" />
              </FormField>
              <FormField label="Factor B name" required>
                <input type="text" value={factorBName} onChange={e => setFactorBName(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  placeholder="Variety" />
              </FormField>
            </>
          )}

          {/* BALANCED LATTICE */}
          {design === "lattice" && (
            <FormField
              label="Number of Treatments"
              helper="Must be a perfect square with prime square root: 4, 9, 25, or 49 treatments."
              required
            >
              <select value={nTreatments} onChange={e => setNTreatments(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm">
                <option value={4}>4 treatments (2x2, replications = 3)</option>
                <option value={9}>9 treatments (3x3, replications = 4)</option>
                <option value={25}>25 treatments (5x5, replications = 6)</option>
                <option value={49}>49 treatments (7x7, replications = 8)</option>
              </select>
              <p className="text-xs text-gray-400 mt-1">
                Replications are fixed by the balanced lattice constraint: r = block_size + 1
              </p>
            </FormField>
          )}

          {/* ALPHA LATTICE */}
          {design === "alpha_lattice" && (
            <>
              <FormField label="Number of Treatments" helper="Must be divisible by block size" required>
                <input type="number" min={4} max={200}
                  value={nTreatments} onChange={e => setNTreatments(Number(e.target.value))}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm" />
              </FormField>
              <FormField label="Block Size (k)" helper="Number of treatments per incomplete block" required>
                <input type="number" min={2} max={20}
                  value={blockSize} onChange={e => setBlockSize(Number(e.target.value))}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm" />
              </FormField>
            </>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="reps">Replications</Label>
              <Input id="reps" type="number" min={2} max={20} value={replications}
                onChange={(e) => setReplications(parseInt(e.target.value, 10) || 0)} className="mt-1.5" />
            </div>
            <div>
              <Label htmlFor="seed">Randomization Seed</Label>
              <Input id="seed" type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)} className="mt-1.5" />
            </div>
            <div>
              <Label>&nbsp;</Label>
              <Button
                variant="outline"
                className="mt-1.5 w-full"
                onClick={() => setSeed(Math.floor(Math.random() * 1_000_000))}
              >
                <Shuffle className="h-4 w-4 mr-1.5" /> New Seed
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="plotW">Plot Width (m)</Label>
              <Input id="plotW" type="number" step="0.1" min={0.1} value={plotWidth}
                onChange={(e) => setPlotWidth(parseFloat(e.target.value) || 0)} className="mt-1.5" />
            </div>
            <div>
              <Label htmlFor="plotL">Plot Length (m)</Label>
              <Input id="plotL" type="number" step="0.1" min={0.1} value={plotLength}
                onChange={(e) => setPlotLength(parseFloat(e.target.value) || 0)} className="mt-1.5" />
            </div>
            <div>
              <Label htmlFor="alleyW">Alley Width (m)</Label>
              <Input id="alleyW" type="number" step="0.1" min={0} value={aisleWidth}
                onChange={(e) => setAisleWidth(parseFloat(e.target.value) || 0)} className="mt-1.5" />
            </div>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="flex flex-wrap gap-3">
            <Button
              onClick={generate}
              disabled={isLoading || (selectedDesign?.requiresPro && !isPro)}
              style={{ backgroundColor: "#1B5E20" }}
              className="text-white hover:opacity-90 disabled:opacity-60"
            >
              {isLoading ? "Generating..." : "Generate Layout"}
            </Button>
            {layout && (
              <>
                <Button variant="outline" onClick={downloadPng}>
                  <Download className="h-4 w-4 mr-1.5" /> Export PNG
                </Button>
                <Button
                  variant="outline"
                  onClick={downloadCsv}
                  className="border-emerald-600 text-emerald-600 hover:bg-emerald-50"
                >
                  <Download className="h-4 w-4 mr-1.5" /> Download Data Collection CSV
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {layout && (
        <Card>
          <CardHeader>
            <CardTitle className="font-serif text-lg">Layout Preview</CardTitle>
            <CardDescription>
              {(result?.layout_summary?.title ?? layout.design)} - {result?.layout_summary?.n_treatments ?? treatmentUniverse.length} treatments x {result?.layout_summary?.replications ?? replications} reps = {layout.cells.length} plots
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="overflow-auto border rounded-md bg-white p-2">
              <svg
                ref={svgRef}
                width={svgW}
                height={svgH}
                viewBox={`0 0 ${svgW} ${svgH}`}
                xmlns="http://www.w3.org/2000/svg"
                style={{ display: "block" }}
              >
                <rect x={0} y={0} width={svgW} height={svgH} fill="#ffffff" />
                <text x={pad} y={18} fontFamily="Montserrat, sans-serif" fontSize={13} fill="#1B5E20" fontWeight={600}>
                  {result?.design_type ?? layout.design} Field Layout - N (Top)
                </text>
                {isLatticeDesign && Array.from(new Map(layout.cells.filter(c => c.blockLabel).map(c => [c.row, c.blockLabel!])).entries()).map(([rowIdx, label]) => (
                  <g key={`blk-${rowIdx}`}>
                    <line
                      x1={pad} x2={pad + blockLabelW + layout.nCols * cellW}
                      y1={pad + 12 + rowIdx * cellH} y2={pad + 12 + rowIdx * cellH}
                      stroke="#1B5E20" strokeWidth={0.5} strokeDasharray="3,2"
                    />
                    <text
                      x={pad} y={pad + 12 + rowIdx * cellH + cellH / 2 + 4}
                      fontFamily="Montserrat, sans-serif" fontSize={10} fontWeight={600} fill="#1B5E20"
                    >
                      {label}
                    </text>
                  </g>
                ))}
                {layout.cells.map((c) => {
                  const x = pad + blockLabelW + c.col * cellW;
                  const y = pad + 12 + c.row * cellH;
                  return (
                    <g key={c.plotId}>
                      <rect
                        x={x + 2} y={y + 2}
                        width={cellW - 4} height={cellH - 4}
                        fill={colorFor(c.treatment)}
                        fillOpacity={0.18}
                        stroke={colorFor(c.treatment)}
                        strokeWidth={1.4}
                        rx={3}
                      />
                      <text x={x + cellW / 2} y={y + cellH / 2 - 4}
                        textAnchor="middle" fontFamily="Montserrat, sans-serif"
                        fontSize={isSplitDesign ? 10 : 14} fontWeight={600} fill="#1f2937">
                        {c.treatment}
                      </text>
                      <text x={x + cellW / 2} y={y + cellH / 2 + 14}
                        textAnchor="middle" fontFamily="Montserrat, sans-serif"
                        fontSize={10} fill="#4b5563">
                        Plot {c.plotId} - Rep {c.rep}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm bg-muted/40 rounded-md p-4">
              <div>
                <p className="text-xs text-muted-foreground">Total Plots</p>
                <p className="font-semibold">{result?.layout_summary?.n_plots ?? layout.cells.length}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Plot Size</p>
                <p className="font-semibold">{plotWidth} x {plotLength} m ({(plotWidth * plotLength).toFixed(2)} m2)</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Grid</p>
                <p className="font-semibold">{layout.nRows} rows x {layout.nCols} cols</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Approx. Field Area</p>
                <p className="font-semibold">{totalArea.toFixed(1)} m2</p>
              </div>
            </div>

            {result?.alpha_value !== null && result?.alpha_value !== undefined && (
              <div style={{
                padding: "8px 12px", background: "#EAF3DE",
                borderRadius: 8, fontSize: 12, color: "#3B6D11",
                marginTop: 8
              }}>
                alpha = {result.alpha_value.toFixed(4)} - Lambda value for this alpha lattice design.
                Lower alpha means better pairwise balance across blocks.
              </div>
            )}

            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription className="text-xs">
                Layouts are generated by the backend design engine. Reusing the same seed reproduces the same layout for traceability.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default FieldLayoutGenerator;
