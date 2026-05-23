import * as React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Copy, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export interface ScientificInterpretationSection {
  title: string;
  content: string;
}

export const fmt = (v: number | null | undefined, digits = 3): string => {
  if (typeof v === "number" && !isNaN(v)) {
    return v.toFixed(digits);
  }
  if (v === null || v === undefined) return "—";
  return String(v);
};

export function buildScientificInterpretationSections(text: string): ScientificInterpretationSection[] {
  const cleaned = text.trim();
  if (!cleaned) return [];

  const paragraphs = cleaned.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
  const sentences = cleaned.split(/(?<=[.!?])\s+/).map((part) => part.trim()).filter(Boolean);

  const sectionDefs = [
    { title: "Statistical implication", pattern: /(significant|anova|variance|heritability|mean separation|p-value|coefficient|component|model)/i },
    { title: "Biological implication", pattern: /(trait|performance|expression|yield|biological|agronomic|physiological|response)/i },
    { title: "Breeding implication", pattern: /(breeding|selection|genotype|advance|parent|improvement|heritability|gcv|pcv|gam)/i },
    { title: "Stability implication", pattern: /(stability|environment|adaptation|adapted|gxe|gge|ammi|aec|interaction)/i },
    { title: "Recommendation", pattern: /(recommend|should|prioriti|advance|select|consider|use|favour)/i },
    { title: "Cautionary note", pattern: /(caution|however|note that|interpret with caution|imbalance|limited|insufficient|uncertain|constraint)/i },
  ] as const;

  const assigned = new Set<number>();
  const sections: ScientificInterpretationSection[] = [];

  sectionDefs.forEach((def) => {
    const matches = sentences.filter((sentence, index) => {
      if (assigned.has(index)) return false;
      return def.pattern.test(sentence);
    });
    if (matches.length > 0) {
      matches.forEach((sentence) => {
        const idx = sentences.indexOf(sentence);
        if (idx >= 0) assigned.add(idx);
      });
      sections.push({ title: def.title, content: matches.join(" ") });
    }
  });

  if (sections.length === 0) {
    const defaults = [
      "Statistical implication",
      "Biological implication",
      "Recommendation",
    ];
    paragraphs.forEach((paragraph, index) => {
      sections.push({
        title: defaults[index] ?? `Interpretation note ${index + 1}`,
        content: paragraph,
      });
    });
    return sections;
  }

  const remaining = sentences.filter((_, index) => !assigned.has(index));
  if (remaining.length > 0) {
    sections.push({ title: "Additional note", content: remaining.join(" ") });
  }

  return sections;
}

export function SummaryCard({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  accent?: "primary" | "emerald" | "amber" | "muted";
}) {
  const accentClass =
    accent === "emerald"
      ? "border-l-4 border-l-emerald-500"
      : accent === "amber"
      ? "border-l-4 border-l-amber-500"
      : accent === "muted"
      ? "border-l-4 border-l-muted-foreground/30"
      : "border-l-4 border-l-primary";
  return (
    <Card className={accentClass}>
      <CardContent className="p-4">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
        <p className="mt-1 font-serif text-2xl font-semibold text-foreground">{value}</p>
        {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  );
}

export function InterpretationPanel({ text }: { text: string }) {
  const { toast } = useToast();
  if (!text) return null;
  const sections = buildScientificInterpretationSections(text);
  return (
    <Card className="bg-primary/[0.03] border-primary/20">
      <CardContent className="p-5">
        <div className="mb-3 flex items-center justify-between">
          <h4 className="font-serif text-base font-semibold text-foreground">AI-assisted academic interpretation</h4>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => {
              navigator.clipboard.writeText(text);
              toast({ title: "Copied", description: "Interpretation copied to clipboard." });
            }}
          >
            <Copy className="h-3.5 w-3.5" />
            Copy
          </Button>
        </div>
        <div className="space-y-3">
          {sections.length > 0 ? (
            sections.map((section) => (
              <div key={`${section.title}-${section.content.slice(0, 24)}`} className="rounded-xl border border-primary/10 bg-white/70 px-4 py-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-primary/80">{section.title}</p>
                <p className="mt-2 text-sm leading-relaxed text-foreground/90">{section.content}</p>
              </div>
            ))
          ) : (
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">{text}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/** Convert an array of row objects to CSV and trigger a download. */
export function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows || rows.length === 0) return;
  const headers = Array.from(
    rows.reduce((set, r) => {
      Object.keys(r).forEach((k) => set.add(k));
      return set;
    }, new Set<string>())
  );
  const escape = (v: unknown) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [
    headers.join(","),
    ...rows.map((r) => headers.map((h) => escape(r[h])).join(",")),
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Trigger PNG download of an SVG element inside a Recharts container. */
export async function exportChartPng(container: HTMLElement | null, filename: string) {
  if (!container) return;
  const svg = container.querySelector("svg");
  if (!svg) return;
  const svgClone = svg.cloneNode(true) as SVGSVGElement;
  // ensure xmlns
  svgClone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const svgString = new XMLSerializer().serializeToString(svgClone);
  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to load chart image"));
    img.src = url;
  });
  const canvas = document.createElement("canvas");
  const rect = svg.getBoundingClientRect();
  canvas.width = Math.max(rect.width, 800) * 2;
  canvas.height = Math.max(rect.height, 500) * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    const dlUrl = URL.createObjectURL(blob);
    a.href = dlUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(dlUrl);
  }, "image/png");
}

export function ExportToolbar({
  onCsv,
  onPng,
  onCopy,
  csvLabel = "Download CSV",
}: {
  onCsv?: () => void;
  onPng?: () => void;
  onCopy?: () => void;
  csvLabel?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {onCsv && (
        <Button variant="outline" size="sm" className="gap-1.5" onClick={onCsv}>
          <Download className="h-3.5 w-3.5" /> {csvLabel}
        </Button>
      )}
      {onPng && (
        <Button variant="outline" size="sm" className="gap-1.5" onClick={onPng}>
          <Download className="h-3.5 w-3.5" /> Export Chart (PNG)
        </Button>
      )}
      {onCopy && (
        <Button variant="outline" size="sm" className="gap-1.5" onClick={onCopy}>
          <Copy className="h-3.5 w-3.5" /> Copy Interpretation
        </Button>
      )}
    </div>
  );
}

export function DatasetTokenWarning() {
  return (
    <Card className="border-amber-300 bg-amber-50/60 dark:bg-amber-900/20">
      <CardContent className="p-4 text-sm text-amber-900 dark:text-amber-200">
        Upload and confirm a dataset above to enable advanced analyses. Each module needs a valid
        dataset token.
      </CardContent>
    </Card>
  );
}
