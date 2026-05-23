import { 
  Award, 
  FileText, 
  BarChart3, 
  GitBranch, 
  MessageSquare, 
  Download, 
  Code,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Layers,
  Trash2,
  ChevronsUpDown,
  Keyboard,
  Copy,
  Check,
  Table2,
  ShieldCheck,
  ShieldX,
  FlaskConical,
  Users
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useState, useEffect, useCallback } from "react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { VsResultSection } from "@/components/vivasense/results/VsResultSection";
import { VS_TYPOGRAPHY } from "@/lib/vivasenseDesignSystem";

export interface VivaSenseResultsData {
  datasetQuality?: {
    badge: "Gold" | "Silver" | "Bronze";
    cv?: number;
    sampleSize?: number;
  };
  audit?: string;
  interpretation?: string;
  regression?: {
    equation?: string;
    rSquared?: number;
    adjustedRSquared?: number;
    coefficients?: Array<{
      variable: string;
      coefficient: number;
      pValue: number;
    }>;
  };
  pathAnalysis?: {
    directEffects?: Array<{
      from: string;
      to: string;
      effect: number;
    }>;
    indirectEffects?: Array<{
      from: string;
      via: string;
      to: string;
      effect: number;
    }>;
    correlationMatrix?: Record<string, Record<string, number>>;
    diagramUrl?: string;
  };
  ammiGge?: {
    triggered: boolean;
    stability?: string;
    explanation?: string;
  };
  reviewerCritique?: {
    questions?: string[];
    weaknesses?: string[];
    suggestions?: string[];
  };
  downloads?: {
    wordReport?: string;
    figures?: string;
  };
  rCode?: string;
  anovaTable?: Array<Record<string, unknown>>;
  groupMeans?: Array<Record<string, unknown>>;
  postHoc?: {
    method?: string;
    comparisons?: Array<{
      group1: string;
      group2: string;
      diff: number;
      pValue: number;
      significant: boolean;
    }>;
    letters?: Array<{
      group: string;
      mean: number;
      letter: string;
    }>;
  };
  shapiroWilk?: {
    statistic: number;
    pValue: number;
    pass: boolean;
  };
  leveneTest?: {
    statistic: number;
    pValue: number;
    pass: boolean;
  };
}

interface VivaSenseResultsProps {
  results: VivaSenseResultsData | null;
  userLevel: string;
  onClearResults?: () => void;
}

type SectionKey = 'quality' | 'audit' | 'anova' | 'groupMeans' | 'assumptions' | 'postHoc' | 'interpretation' | 'regression' | 'path' | 'ammi' | 'critique' | 'downloads' | 'rcode';

function QualityBadge({ badge }: { badge: "Gold" | "Silver" | "Bronze" }) {
  const colors = {
    Gold: "bg-yellow-100 text-yellow-800 border-yellow-300",
    Silver: "bg-gray-100 text-gray-800 border-gray-300",
    Bronze: "bg-orange-100 text-orange-800 border-orange-300"
  };

  return (
    <Badge className={`${colors[badge]} text-lg px-4 py-1 font-semibold`}>
      <Award className="w-5 h-5 mr-2" />
      {badge} Standard
    </Badge>
  );
}

// Utility to download content as a file
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Convert array of objects to CSV string
function arrayToCSV(data: Array<Record<string, unknown>>): string {
  if (!data || data.length === 0) return "";
  const headers = Object.keys(data[0]);
  const rows = data.map(row => 
    headers.map(h => {
      const val = row[h];
      if (val === null || val === undefined) return "";
      const str = String(val);
      if (str.includes(",") || str.includes('"') || str.includes("\n")) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    }).join(",")
  );
  return [headers.join(","), ...rows].join("\n");
}

// Collapsible section component — delegates to the shared VsResultSection
// so the live UI mirrors the Word report's section hierarchy + typography.
function ResultSection({
  title,
  icon,
  children,
  isOpen,
  onToggle,
  actions,
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  actions?: React.ReactNode;
}) {
  return (
    <VsResultSection
      title={title}
      icon={icon}
      isOpen={isOpen}
      onToggle={onToggle}
      actions={actions}
    >
      {children}
    </VsResultSection>
  );
}

export function VivaSenseResults({ results, userLevel, onClearResults }: VivaSenseResultsProps) {
  const { toast } = useToast();
  
  // Track which copy buttons show checkmark
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Initialize all sections as open
  const [openSections, setOpenSections] = useState<Record<SectionKey, boolean>>({
    quality: true,
    audit: true,
    anova: true,
    groupMeans: true,
    assumptions: true,
    postHoc: true,
    interpretation: true,
    regression: true,
    path: true,
    ammi: true,
    critique: true,
    downloads: true,
    rcode: false,
  });

  const toggleSection = useCallback((key: SectionKey) => {
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const expandAll = useCallback(() => {
    setOpenSections({
      quality: true, audit: true, anova: true, groupMeans: true, assumptions: true, postHoc: true,
      interpretation: true, regression: true, path: true, ammi: true, critique: true, downloads: true, rcode: true,
    });
  }, []);

  const collapseAll = useCallback(() => {
    setOpenSections({
      quality: false, audit: false, anova: false, groupMeans: false, assumptions: false, postHoc: false,
      interpretation: false, regression: false, path: false, ammi: false, critique: false, downloads: false, rcode: false,
    });
  }, []);

  // (collapseAll already defined above)

  const toggleAll = useCallback(() => {
    const allOpen = Object.values(openSections).every(v => v);
    if (allOpen) {
      collapseAll();
    } else {
      expandAll();
    }
  }, [openSections, expandAll, collapseAll]);

  // Copy to clipboard handler
  const copyToClipboard = useCallback(async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      toast({
        title: "Copied to clipboard",
        description: "Content has been copied successfully.",
      });
      setTimeout(() => setCopiedKey(null), 2000);
    } catch (err) {
      toast({
        title: "Copy failed",
        description: "Unable to copy to clipboard. Please try again.",
        variant: "destructive",
      });
    }
  }, [toast]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!results) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Ctrl/Cmd + Shift + E = Expand all
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'E') {
        e.preventDefault();
        expandAll();
      }
      // Ctrl/Cmd + Shift + C = Collapse all
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        collapseAll();
      }
      // Ctrl/Cmd + Shift + T = Toggle all
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
        e.preventDefault();
        toggleAll();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [results, expandAll, collapseAll, toggleAll]);

  if (!results) return null;

  const isUndergraduate = userLevel === "undergraduate";
  const showCautionaryNote = isUndergraduate || results.datasetQuality?.badge === "Bronze";
  const allExpanded = Object.values(openSections).every(v => v);

  // Download handlers
  const handleDownloadAnovaCSV = () => {
    if (results.anovaTable && results.anovaTable.length > 0) {
      const csv = arrayToCSV(results.anovaTable);
      downloadFile(csv, "anova_table.csv", "text/csv");
    } else if (results.audit) {
      downloadFile(results.audit, "anova_audit.txt", "text/plain");
    }
  };

  const handleDownloadInterpretation = () => {
    if (results.interpretation) {
      downloadFile(results.interpretation, "interpretation.txt", "text/plain");
    }
  };

  const handleDownloadCritique = () => {
    const content = getCritiqueText();
    if (content) {
      downloadFile(content, "reviewer_critique.txt", "text/plain");
    }
  };

  // Helper to get critique as text (used for both copy and download)
  const getCritiqueText = () => {
    if (!results.reviewerCritique) return "";
    
    let content = "REVIEWER-STYLE CRITIQUE\n\n";
    
    if (results.reviewerCritique.questions?.length) {
      content += "LIKELY REVIEWER QUESTIONS:\n";
      results.reviewerCritique.questions.forEach((q, i) => {
        content += `${i + 1}. ${q}\n`;
      });
      content += "\n";
    }
    
    if (results.reviewerCritique.weaknesses?.length) {
      content += "METHODOLOGICAL WEAKNESSES:\n";
      results.reviewerCritique.weaknesses.forEach((w, i) => {
        content += `${i + 1}. ${w}\n`;
      });
      content += "\n";
    }
    
    if (results.reviewerCritique.suggestions?.length) {
      content += "SUGGESTIONS TO STRENGTHEN:\n";
      results.reviewerCritique.suggestions.forEach((s, i) => {
        content += `${i + 1}. ${s}\n`;
      });
    }
    
    return content;
  };

  return (
    <section className="py-20 bg-muted/30" id="results">
      <div className="container-wide">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center mb-12">
            <p className={`${VS_TYPOGRAPHY.figureLabel} mb-2`}>VivaSense Statistical Analysis</p>
            <h2 className={`${VS_TYPOGRAPHY.pageTitle} mb-3`}>
              Analysis Results
            </h2>
            <p className={`${VS_TYPOGRAPHY.pageSubtitle} mb-4`}>
              Comprehensive scientific report — mirrored in your downloadable Word document
            </p>
            <div className="flex items-center justify-center gap-3 flex-wrap">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={toggleAll}
                    >
                      <ChevronsUpDown className="w-4 h-4 mr-2" />
                      {allExpanded ? "Collapse All" : "Expand All"}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div className="flex items-center gap-2">
                      <Keyboard className="w-4 h-4" />
                      <span>Ctrl+Shift+T to toggle</span>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              {onClearResults && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={onClearResults}
                  className="text-muted-foreground"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear Results
                </Button>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-3">
              <Keyboard className="w-3 h-3 inline mr-1" />
              Shortcuts: Ctrl+Shift+E (expand) • Ctrl+Shift+C (collapse) • Ctrl+Shift+T (toggle)
            </p>
          </div>

          {/* Cautionary Note for Undergraduates or Bronze Quality */}
          {showCautionaryNote && (
            <Card className="border-yellow-300 bg-yellow-50">
              <CardContent className="p-6">
                <div className="flex gap-4">
                  <AlertTriangle className="w-6 h-6 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-yellow-800 mb-2">Important Notice</h3>
                    <p className="text-yellow-700 text-sm leading-relaxed">
                      {isUndergraduate 
                        ? "These results demonstrate associations, not causation. Please discuss interpretations with your supervisor before drawing conclusions. Academic rigour requires careful consideration of study limitations."
                        : "Dataset quality indicates results should be interpreted with caution. Consider reviewing data collection methods and sample adequacy."}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Dataset Quality */}
          {results.datasetQuality && (
            <ResultSection 
              title="Dataset Quality" 
              icon={Award}
              isOpen={openSections.quality}
              onToggle={() => toggleSection('quality')}
            >
              <div className="flex flex-wrap items-center gap-6">
                <QualityBadge badge={results.datasetQuality.badge} />
                {results.datasetQuality.cv !== undefined && (
                  <div>
                    <span className="text-muted-foreground text-sm">CV:</span>
                    <span className="ml-2 font-medium">{results.datasetQuality.cv.toFixed(2)}%</span>
                  </div>
                )}
                {results.datasetQuality.sampleSize !== undefined && (
                  <div>
                    <span className="text-muted-foreground text-sm">Sample Size:</span>
                    <span className="ml-2 font-medium">n = {results.datasetQuality.sampleSize}</span>
                  </div>
                )}
              </div>
            </ResultSection>
          )}

          {/* Audit / ANOVA Table */}
          {results.audit && (
            <ResultSection 
              title="Statistical Audit" 
              icon={FileText}
              isOpen={openSections.audit}
              onToggle={() => toggleSection('audit')}
              actions={
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleDownloadAnovaCSV}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV
                </Button>
              }
            >
              <div className="prose prose-sm max-w-none text-muted-foreground">
                <p className="whitespace-pre-wrap">{results.audit}</p>
              </div>
            </ResultSection>
          )}

          {/* ANOVA Table */}
          {results.anovaTable && results.anovaTable.length > 0 && (
            <ResultSection 
              title="ANOVA Table" 
              icon={Table2}
              isOpen={openSections.anova}
              onToggle={() => toggleSection('anova')}
              actions={
                <Button variant="outline" size="sm" onClick={handleDownloadAnovaCSV}>
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV
                </Button>
              }
            >
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      {Object.keys(results.anovaTable[0]).map(key => (
                        <th key={key} className="text-left py-2 px-3 font-medium">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.anovaTable.map((row, idx) => (
                      <tr key={idx} className="border-b border-border/50">
                        {Object.values(row).map((val, vIdx) => (
                          <td key={vIdx} className="py-2 px-3 font-mono text-sm">
                            {val === null || val === undefined ? "–" : typeof val === 'number' ? val.toFixed(4) : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ResultSection>
          )}

          {/* Group Means Table */}
          {results.groupMeans && results.groupMeans.length > 0 && (
            <ResultSection 
              title="Group Means" 
              icon={Users}
              isOpen={openSections.groupMeans}
              onToggle={() => toggleSection('groupMeans')}
              actions={
                <Button variant="outline" size="sm" onClick={() => {
                  const csv = arrayToCSV(results.groupMeans!);
                  downloadFile(csv, "group_means.csv", "text/csv");
                }}>
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV
                </Button>
              }
            >
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      {Object.keys(results.groupMeans[0]).map(key => (
                        <th key={key} className="text-left py-2 px-3 font-medium">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.groupMeans.map((row, idx) => (
                      <tr key={idx} className="border-b border-border/50">
                        {Object.values(row).map((val, vIdx) => (
                          <td key={vIdx} className="py-2 px-3 font-mono text-sm">
                            {val === null || val === undefined ? "–" : typeof val === 'number' ? val.toFixed(4) : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ResultSection>
          )}

          {/* Assumption Tests: Shapiro-Wilk & Levene's */}
          {(results.shapiroWilk || results.leveneTest) && (
            <ResultSection 
              title="Assumption Diagnostics" 
              icon={FlaskConical}
              isOpen={openSections.assumptions}
              onToggle={() => toggleSection('assumptions')}
            >
              <div className="grid gap-4 md:grid-cols-2">
                {results.shapiroWilk && (
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium text-foreground">Shapiro-Wilk Normality Test</h4>
                      <Badge className={results.shapiroWilk.pass 
                        ? "bg-emerald-100 text-emerald-800 border-emerald-300" 
                        : "bg-red-100 text-red-800 border-red-300"}>
                        {results.shapiroWilk.pass ? (
                          <><ShieldCheck className="w-3.5 h-3.5 mr-1" /> Pass</>
                        ) : (
                          <><ShieldX className="w-3.5 h-3.5 mr-1" /> Fail</>
                        )}
                      </Badge>
                    </div>
                    <div className="space-y-1 text-sm text-muted-foreground">
                      <p>W statistic: <span className="font-mono font-medium">{results.shapiroWilk.statistic.toFixed(4)}</span></p>
                      <p>p-value: <span className={`font-mono font-medium ${results.shapiroWilk.pValue < 0.05 ? "text-red-600" : "text-emerald-600"}`}>{results.shapiroWilk.pValue.toFixed(4)}</span></p>
                      <p className="text-xs mt-2 italic">
                        {results.shapiroWilk.pass 
                          ? "Residuals appear normally distributed (p ≥ 0.05)." 
                          : "Normality assumption may be violated (p < 0.05). Consider non-parametric alternatives."}
                      </p>
                    </div>
                  </div>
                )}
                {results.leveneTest && (
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium text-foreground">Levene's Test for Homogeneity</h4>
                      <Badge className={results.leveneTest.pass 
                        ? "bg-emerald-100 text-emerald-800 border-emerald-300" 
                        : "bg-red-100 text-red-800 border-red-300"}>
                        {results.leveneTest.pass ? (
                          <><ShieldCheck className="w-3.5 h-3.5 mr-1" /> Pass</>
                        ) : (
                          <><ShieldX className="w-3.5 h-3.5 mr-1" /> Fail</>
                        )}
                      </Badge>
                    </div>
                    <div className="space-y-1 text-sm text-muted-foreground">
                      <p>F statistic: <span className="font-mono font-medium">{results.leveneTest.statistic.toFixed(4)}</span></p>
                      <p>p-value: <span className={`font-mono font-medium ${results.leveneTest.pValue < 0.05 ? "text-red-600" : "text-emerald-600"}`}>{results.leveneTest.pValue.toFixed(4)}</span></p>
                      <p className="text-xs mt-2 italic">
                        {results.leveneTest.pass 
                          ? "Variances appear homogeneous across groups (p ≥ 0.05)." 
                          : "Variance homogeneity may be violated (p < 0.05). Consider Welch's ANOVA."}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </ResultSection>
          )}

          {/* Post-hoc Analysis (Tukey HSD) */}
          {results.postHoc && (
            <ResultSection 
              title={`Post-hoc Analysis${results.postHoc.method ? ` (${results.postHoc.method})` : ' (Tukey HSD)'}`}
              icon={BarChart3}
              isOpen={openSections.postHoc}
              onToggle={() => toggleSection('postHoc')}
            >
              <div className="space-y-6">
                {results.postHoc.comparisons && results.postHoc.comparisons.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Pairwise Comparisons</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-3 font-medium">Group 1</th>
                            <th className="text-left py-2 px-3 font-medium">Group 2</th>
                            <th className="text-right py-2 px-3 font-medium">Difference</th>
                            <th className="text-right py-2 px-3 font-medium">p-value</th>
                            <th className="text-center py-2 px-3 font-medium">Sig.</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.postHoc.comparisons.map((comp, idx) => (
                            <tr key={idx} className="border-b border-border/50">
                              <td className="py-2 px-3">{comp.group1}</td>
                              <td className="py-2 px-3">{comp.group2}</td>
                              <td className="py-2 px-3 text-right font-mono">{comp.diff.toFixed(4)}</td>
                              <td className={`py-2 px-3 text-right font-mono ${comp.significant ? "text-emerald-600 font-medium" : ""}`}>
                                {comp.pValue.toFixed(4)}
                              </td>
                              <td className="py-2 px-3 text-center">
                                {comp.significant ? (
                                  <Badge className="bg-emerald-100 text-emerald-800 border-emerald-300 text-xs">Yes</Badge>
                                ) : (
                                  <Badge variant="outline" className="text-xs">No</Badge>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                {results.postHoc.letters && results.postHoc.letters.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Mean Separation (Letter Grouping)</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-3 font-medium">Group</th>
                            <th className="text-right py-2 px-3 font-medium">Mean</th>
                            <th className="text-center py-2 px-3 font-medium">Letter</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.postHoc.letters.map((item, idx) => (
                            <tr key={idx} className="border-b border-border/50">
                              <td className="py-2 px-3">{item.group}</td>
                              <td className="py-2 px-3 text-right font-mono">{item.mean.toFixed(4)}</td>
                              <td className="py-2 px-3 text-center font-bold text-lg text-primary">{item.letter}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 italic">
                      Groups sharing the same letter are not significantly different (p ≥ 0.05).
                    </p>
                  </div>
                )}
              </div>
            </ResultSection>
          )}

          {/* Interpretation */}
          {results.interpretation && (
            <ResultSection 
              title="Interpretation" 
              icon={MessageSquare}
              isOpen={openSections.interpretation}
              onToggle={() => toggleSection('interpretation')}
              actions={
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => copyToClipboard(results.interpretation!, 'interpretation')}
                  >
                    {copiedKey === 'interpretation' ? (
                      <Check className="w-4 h-4 mr-2 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 mr-2" />
                    )}
                    Copy
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={handleDownloadInterpretation}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
              }
            >
              <div className="prose prose-sm max-w-none text-muted-foreground">
                <p className="whitespace-pre-wrap">{results.interpretation}</p>
              </div>
            </ResultSection>
          )}

          {/* Regression Results */}
          {results.regression && (
            <ResultSection 
              title="Regression Results" 
              icon={BarChart3}
              isOpen={openSections.regression}
              onToggle={() => toggleSection('regression')}
            >
              <div className="space-y-4">
                {results.regression.equation && (
                  <div className="p-4 bg-muted rounded-lg font-mono text-sm overflow-x-auto">
                    {results.regression.equation}
                  </div>
                )}
                <div className="flex flex-wrap gap-6">
                  {results.regression.rSquared !== undefined && (
                    <div>
                      <span className="text-muted-foreground text-sm">R²:</span>
                      <span className="ml-2 font-medium">{results.regression.rSquared.toFixed(4)}</span>
                    </div>
                  )}
                  {results.regression.adjustedRSquared !== undefined && (
                    <div>
                      <span className="text-muted-foreground text-sm">Adjusted R²:</span>
                      <span className="ml-2 font-medium">{results.regression.adjustedRSquared.toFixed(4)}</span>
                    </div>
                  )}
                </div>
                {results.regression.coefficients && results.regression.coefficients.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 px-3 font-medium">Variable</th>
                          <th className="text-right py-2 px-3 font-medium">Coefficient</th>
                          <th className="text-right py-2 px-3 font-medium">p-value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.regression.coefficients.map((coef, idx) => (
                          <tr key={idx} className="border-b border-border/50">
                            <td className="py-2 px-3">{coef.variable}</td>
                            <td className="py-2 px-3 text-right font-mono">{coef.coefficient.toFixed(4)}</td>
                            <td className={`py-2 px-3 text-right font-mono ${coef.pValue < 0.05 ? "text-green-600 font-medium" : ""}`}>
                              {coef.pValue.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </ResultSection>
          )}

          {/* Path Analysis */}
          {results.pathAnalysis && (
            <ResultSection 
              title="Path Analysis Results" 
              icon={GitBranch}
              isOpen={openSections.path}
              onToggle={() => toggleSection('path')}
            >
              <div className="space-y-6">
                {results.pathAnalysis.directEffects && results.pathAnalysis.directEffects.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Direct Effects</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-3 font-medium">From</th>
                            <th className="text-left py-2 px-3 font-medium">To</th>
                            <th className="text-right py-2 px-3 font-medium">Effect</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.pathAnalysis.directEffects.map((effect, idx) => (
                            <tr key={idx} className="border-b border-border/50">
                              <td className="py-2 px-3">{effect.from}</td>
                              <td className="py-2 px-3">{effect.to}</td>
                              <td className="py-2 px-3 text-right font-mono">{effect.effect.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {results.pathAnalysis.indirectEffects && results.pathAnalysis.indirectEffects.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Indirect Effects</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-3 font-medium">From</th>
                            <th className="text-left py-2 px-3 font-medium">Via</th>
                            <th className="text-left py-2 px-3 font-medium">To</th>
                            <th className="text-right py-2 px-3 font-medium">Effect</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.pathAnalysis.indirectEffects.map((effect, idx) => (
                            <tr key={idx} className="border-b border-border/50">
                              <td className="py-2 px-3">{effect.from}</td>
                              <td className="py-2 px-3">{effect.via}</td>
                              <td className="py-2 px-3">{effect.to}</td>
                              <td className="py-2 px-3 text-right font-mono">{effect.effect.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {results.pathAnalysis.diagramUrl && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Path Diagram</h4>
                    <div className="bg-muted rounded-lg p-4">
                      <img 
                        src={results.pathAnalysis.diagramUrl} 
                        alt="Path Analysis Diagram" 
                        className="max-w-full mx-auto"
                      />
                      <p className="text-sm text-muted-foreground text-center mt-3 italic">
                        Exploratory path diagram based on standardized coefficients
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </ResultSection>
          )}

          {/* AMMI/GGE Results */}
          {results.ammiGge && (
            <ResultSection 
              title="AMMI / GGE Analysis" 
              icon={Layers}
              isOpen={openSections.ammi}
              onToggle={() => toggleSection('ammi')}
            >
              {results.ammiGge.triggered ? (
                <div className="space-y-4">
                  {results.ammiGge.stability && (
                    <div>
                      <h4 className="font-medium text-foreground mb-2">Stability Interpretation</h4>
                      <p className="text-muted-foreground">{results.ammiGge.stability}</p>
                    </div>
                  )}
                  {results.ammiGge.explanation && (
                    <p className="text-muted-foreground">{results.ammiGge.explanation}</p>
                  )}
                </div>
              ) : (
                <p className="text-muted-foreground">
                  AMMI/GGE analysis was not triggered. This analysis requires data from 2 or more environments.
                </p>
              )}
            </ResultSection>
          )}

          {/* Reviewer Critique */}
          {results.reviewerCritique && (
            <ResultSection 
              title="Reviewer-Style Critique" 
              icon={MessageSquare}
              isOpen={openSections.critique}
              onToggle={() => toggleSection('critique')}
              actions={
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => copyToClipboard(getCritiqueText(), 'critique')}
                  >
                    {copiedKey === 'critique' ? (
                      <Check className="w-4 h-4 mr-2 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 mr-2" />
                    )}
                    Copy
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={handleDownloadCritique}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
              }
            >
              <div className="space-y-6">
                {results.reviewerCritique.questions && results.reviewerCritique.questions.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Likely Reviewer Questions</h4>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                      {results.reviewerCritique.questions.map((q, idx) => (
                        <li key={idx}>{q}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {results.reviewerCritique.weaknesses && results.reviewerCritique.weaknesses.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Methodological Weaknesses</h4>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                      {results.reviewerCritique.weaknesses.map((w, idx) => (
                        <li key={idx}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {results.reviewerCritique.suggestions && results.reviewerCritique.suggestions.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Suggestions to Strengthen</h4>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                      {results.reviewerCritique.suggestions.map((s, idx) => (
                        <li key={idx}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </ResultSection>
          )}

          {/* Downloads from backend */}
          {results.downloads && (
            <ResultSection 
              title="Download Results" 
              icon={Download}
              isOpen={openSections.downloads}
              onToggle={() => toggleSection('downloads')}
            >
              <div className="flex flex-wrap gap-4">
                {results.downloads.wordReport && (
                  <Button variant="outline" size="lg" asChild>
                    <a href={results.downloads.wordReport} download>
                      <FileText className="w-5 h-5 mr-2" />
                      Download Word Report
                    </a>
                  </Button>
                )}
                {results.downloads.figures && (
                  <Button variant="outline" size="lg" asChild>
                    <a href={results.downloads.figures} download>
                      <Download className="w-5 h-5 mr-2" />
                      Download Figures
                    </a>
                  </Button>
                )}
              </div>
            </ResultSection>
          )}

          {/* R Code */}
          {results.rCode && (
            <ResultSection 
              title="R Code" 
              icon={Code}
              isOpen={openSections.rcode}
              onToggle={() => toggleSection('rcode')}
            >
              <div className="bg-muted rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm font-mono whitespace-pre-wrap">{results.rCode}</pre>
              </div>
            </ResultSection>
          )}

          {/* Branding Footer */}
          <div className="text-center pt-8 border-t border-border/50 mt-8">
            <p className="text-sm font-semibold text-foreground">
              VivaSense™ – A Statistical Intelligence Engine by Field-to-Insight Academy
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              © Dr. Fayeun Lawrence Stephen
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
