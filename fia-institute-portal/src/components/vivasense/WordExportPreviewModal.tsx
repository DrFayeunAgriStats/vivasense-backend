import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertTriangle, Info, Download, Loader2, FileText } from "lucide-react";

/**
 * Reusable preview modal shown before downloading a VivaSense Word report.
 * Renders an outline of the document the user is about to export so they can
 * verify the content before triggering the actual `.docx` build.
 */

export interface WordPreviewSection {
  /** Section heading (e.g. "1. Dataset & Variables") */
  heading: string;
  /** Optional plain-text bullet rows */
  bullets?: string[];
  /** Optional simple key/value table rows */
  rows?: Array<[string, string]>;
  /** Optional free-form paragraph text */
  paragraph?: string;
}

export interface WordExportPreviewModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** e.g. "Regression", "ANOVA" */
  moduleName: string;
  /** Title shown in the modal header */
  reportTitle: string;
  /** Short context summary (dataset, n, variables…) */
  datasetSummary: string;
  /** Ordered sections that mirror the eventual .docx layout */
  sections: WordPreviewSection[];
  /** Caution / amber items */
  warnings?: string[];
  /** Neutral muted notes */
  notes?: string[];
  /** Whether export is allowed */
  canExport: boolean;
  /** Disable reason shown when canExport=false */
  cannotExportReason?: string;
  /** Triggered when the user confirms the download */
  onConfirmExport: () => Promise<void> | void;
}

export function WordExportPreviewModal({
  open,
  onOpenChange,
  moduleName,
  reportTitle,
  datasetSummary,
  sections,
  warnings = [],
  notes = [],
  canExport,
  cannotExportReason,
  onConfirmExport,
}: WordExportPreviewModalProps) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    if (!canExport || isExporting) return;
    setIsExporting(true);
    try {
      await onConfirmExport();
      onOpenChange(false);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !isExporting && onOpenChange(o)}>
      <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-primary" />
            Preview Report
          </DialogTitle>
          <DialogDescription>
            Review the structure of your <span className="font-medium">{moduleName}</span> report
            before downloading.
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 pr-4 -mr-4">
          <div className="space-y-5 py-2">
            {/* Header block */}
            <div className="rounded-md border border-border bg-muted/30 p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                VivaSense Statistical Analysis Report
              </p>
              <p className="text-base font-semibold text-foreground">{reportTitle}</p>
              <p className="text-sm text-muted-foreground mt-1">{datasetSummary}</p>
              <p className="text-xs text-muted-foreground mt-2">
                Report generated: {new Date().toLocaleString()}
              </p>
            </div>

            {/* Warnings */}
            {warnings.length > 0 && (
              <div className="rounded-md border border-amber-300/60 bg-amber-50/70 dark:bg-amber-900/10 dark:border-amber-700/40 p-3">
                <div className="flex items-center gap-2 text-amber-800 dark:text-amber-200 font-medium mb-1.5">
                  <AlertTriangle className="h-4 w-4" />
                  Warnings & Reliability
                </div>
                <ul className="list-disc pl-5 space-y-1 text-amber-900 dark:text-amber-100 text-sm">
                  {warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Notes */}
            {notes.length > 0 && (
              <div className="rounded-md border border-border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-foreground font-medium mb-1.5">
                  <Info className="h-4 w-4 text-primary" />
                  Notes
                </div>
                <ul className="list-disc pl-5 space-y-1 text-muted-foreground text-sm">
                  {notes.map((n, i) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Sections */}
            {sections.map((sec, i) => (
              <div key={i} className="space-y-2">
                <p className="font-semibold text-foreground border-b border-border pb-1">
                  {sec.heading}
                </p>
                {sec.paragraph && (
                  <p className="text-sm text-muted-foreground leading-relaxed">{sec.paragraph}</p>
                )}
                {sec.rows && sec.rows.length > 0 && (
                  <div className="rounded-md border border-border overflow-hidden">
                    <table className="w-full text-sm">
                      <tbody>
                        {sec.rows.map(([k, v], j) => (
                          <tr key={j} className="border-b border-border last:border-b-0">
                            <td className="px-3 py-1.5 font-medium text-foreground bg-muted/30 w-1/3">
                              {k}
                            </td>
                            <td className="px-3 py-1.5 text-muted-foreground font-mono text-xs">
                              {v}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {sec.bullets && sec.bullets.length > 0 && (
                  <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                    {sec.bullets.map((b, j) => (
                      <li key={j}>{b}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}

            {/* Footer preview */}
            <div className="rounded-md border border-dashed border-border p-3 text-center text-xs text-muted-foreground">
              VivaSense Engine v1.0 · "From Statistical Output to Scientific Insight"
            </div>
          </div>
        </ScrollArea>

        <DialogFooter className="flex-col sm:flex-row gap-2 sm:justify-between">
          <div className="text-xs text-muted-foreground self-center">
            {!canExport && cannotExportReason ? cannotExportReason : "Ready to export"}
          </div>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              onClick={() => onOpenChange(false)}
              disabled={isExporting}
            >
              Cancel
            </Button>
            <Button
              onClick={handleExport}
              disabled={!canExport || isExporting}
              className="gap-2"
            >
              {isExporting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Download Word
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
