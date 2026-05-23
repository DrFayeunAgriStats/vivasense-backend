import { useEffect, useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { useNavigate } from "react-router-dom";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  FileText,
  AlertTriangle,
  CheckCircle2,
  BookOpen,
  Award,
  ArrowLeft,
  ListChecks,
  ShieldAlert,
  Copy,
  Download,
  Loader2,
} from "lucide-react";
import { toast } from "sonner";
import type { ReviewResult } from "@/config/manuscriptReviewer";
import { MANUSCRIPT_REVIEWER_BASE } from "@/config/manuscriptReviewer";

const severityColor: Record<string, string> = {
  Minor: "bg-primary/10 text-primary",
  Moderate: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400",
  Major: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",
  Critical: "bg-destructive/10 text-destructive",
};

const recommendationColor: Record<string, string> = {
  Accept: "bg-green-600",
  "Minor Revision": "bg-primary",
  "Major Revision": "bg-yellow-600",
  Reject: "bg-destructive",
};

function ConcernList({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="mb-6">
      <h4 className="font-semibold text-foreground mb-2">{title}</h4>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
            <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function CitationSection({ title, items }: { title: string; items: string[] }) {
  if (!items.length)
    return (
      <div className="mb-6">
        <h4 className="font-semibold text-foreground mb-2">{title}</h4>
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-green-600" /> No issues found.
        </p>
      </div>
    );
  return (
    <div className="mb-6">
      <h4 className="font-semibold text-foreground mb-2">{title}</h4>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
            <BookOpen className="w-4 h-4 text-primary shrink-0 mt-0.5" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DownloadButton({
  label,
  url,
}: {
  label: string;
  url: string | undefined;
}) {
  const [loading, setLoading] = useState(false);

  const handleDownload = async () => {
    if (!url) return;
    setLoading(true);
    try {
      const fullUrl = url.startsWith("http") ? url : `${MANUSCRIPT_REVIEWER_BASE}${url}`;
      const res = await fetch(fullUrl);
      if (!res.ok) throw new Error(`Download failed (${res.status})`);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      const filename = url.split("/").pop() || "download";
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
      toast.success(`Downloaded: ${label}`);
    } catch (err: any) {
      toast.error(`Failed to download ${label}: ${err?.message || "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button
      variant="outline"
      size="sm"
      className="gap-2 justify-start"
      disabled={!url || loading}
      onClick={handleDownload}
    >
      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
      {label}
    </Button>
  );
}

function AiProbabilityBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct < 20 ? "bg-green-500" : pct < 50 ? "bg-yellow-500" : pct < 75 ? "bg-orange-500" : "bg-destructive";
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-2.5 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-sm font-medium text-foreground w-12 text-right">{pct}%</span>
    </div>
  );
}

export default function ManuscriptReviewerResults() {
  const [result, setResult] = useState<ReviewResult | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const raw = sessionStorage.getItem("manuscript_review_result");
    if (raw) {
      try {
        setResult(JSON.parse(raw));
      } catch {
        navigate("/manuscript-reviewer/upload");
      }
    } else {
      navigate("/manuscript-reviewer/upload");
    }
  }, [navigate]);

  if (!result) return null;

  const { similarity_review, ai_writing_risk_review, exports_available } = result;

  // Safe defaults to prevent crashes on partial API responses
  const language_review = {
    manuscript_title: result.language_review?.manuscript_title ?? "",
    issues: Array.isArray(result.language_review?.issues) ? result.language_review.issues : [],
  };
  const technical_review = {
    major_scientific_concerns: Array.isArray(result.technical_review?.major_scientific_concerns) ? result.technical_review.major_scientific_concerns : [],
    methodological_concerns: Array.isArray(result.technical_review?.methodological_concerns) ? result.technical_review.methodological_concerns : [],
    statistical_concerns: Array.isArray(result.technical_review?.statistical_concerns) ? result.technical_review.statistical_concerns : [],
    interpretation_problems: Array.isArray(result.technical_review?.interpretation_problems) ? result.technical_review.interpretation_problems : [],
  };
  const final_summary = {
    overall_assessment: result.final_summary?.overall_assessment ?? "",
    top_10_priority_revisions: Array.isArray(result.final_summary?.top_10_priority_revisions) ? result.final_summary.top_10_priority_revisions : [],
    final_recommendation: result.final_summary?.final_recommendation ?? "Pending",
  };
  const citation_audit = {
    missing_from_references: Array.isArray(result.citation_audit?.missing_from_references) ? result.citation_audit.missing_from_references : [],
    uncited_references: Array.isArray(result.citation_audit?.uncited_references) ? result.citation_audit.uncited_references : [],
    inconsistencies: Array.isArray(result.citation_audit?.inconsistencies) ? result.citation_audit.inconsistencies : [],
    formatting_issues: Array.isArray(result.citation_audit?.formatting_issues) ? result.citation_audit.formatting_issues : [],
  };

  const totalLanguageIssues = language_review.issues.length;
  const totalTechnicalConcerns =
    technical_review.major_scientific_concerns.length +
    technical_review.methodological_concerns.length +
    technical_review.statistical_concerns.length +
    technical_review.interpretation_problems.length;
  const totalCitationIssues =
    citation_audit.missing_from_references.length +
    citation_audit.uncited_references.length +
    citation_audit.inconsistencies.length +
    citation_audit.formatting_issues.length;

  return (
    <Layout>
      <section className="py-12 md:py-20 bg-background">
        <div className="container-wide px-4 max-w-5xl mx-auto">
          <Button
            variant="ghost"
            size="sm"
            className="mb-6 gap-1 text-muted-foreground"
            onClick={() => navigate("/manuscript-reviewer/upload")}
          >
            <ArrowLeft className="w-4 h-4" /> New Review
          </Button>

          <h1 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-1">
            Review Results
          </h1>
          <p className="text-muted-foreground mb-8 text-sm">{result.manuscript_title}</p>

          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="w-full flex flex-wrap h-auto gap-1 bg-muted/50 p-1 rounded-lg">
              <TabsTrigger value="overview" className="gap-1.5 text-xs sm:text-sm">
                <ListChecks className="w-4 h-4" /> Overview
              </TabsTrigger>
              <TabsTrigger value="line" className="gap-1.5 text-xs sm:text-sm">
                <FileText className="w-4 h-4" /> Line Review
              </TabsTrigger>
              <TabsTrigger value="technical" className="gap-1.5 text-xs sm:text-sm">
                <AlertTriangle className="w-4 h-4" /> Technical
              </TabsTrigger>
              <TabsTrigger value="citations" className="gap-1.5 text-xs sm:text-sm">
                <BookOpen className="w-4 h-4" /> Citations
              </TabsTrigger>
              {similarity_review && (
                <TabsTrigger value="similarity" className="gap-1.5 text-xs sm:text-sm">
                  <Copy className="w-4 h-4" /> Similarity
                </TabsTrigger>
              )}
              {ai_writing_risk_review && (
                <TabsTrigger value="airisk" className="gap-1.5 text-xs sm:text-sm">
                  <ShieldAlert className="w-4 h-4" /> AI Risk
                </TabsTrigger>
              )}
              <TabsTrigger value="verdict" className="gap-1.5 text-xs sm:text-sm">
                <Award className="w-4 h-4" /> Final Verdict
              </TabsTrigger>
              {exports_available && (
                <TabsTrigger value="downloads" className="gap-1.5 text-xs sm:text-sm">
                  <Download className="w-4 h-4" /> Downloads
                </TabsTrigger>
              )}
            </TabsList>

            {/* ─── Overview ─── */}
            <TabsContent value="overview" className="mt-6 space-y-6">
              <div className="grid sm:grid-cols-3 gap-4">
                <SummaryCard label="Language Issues" count={totalLanguageIssues} />
                <SummaryCard label="Technical Concerns" count={totalTechnicalConcerns} />
                <SummaryCard label="Citation Issues" count={totalCitationIssues} />
              </div>

              {(similarity_review || ai_writing_risk_review) && (
                <div className="grid sm:grid-cols-2 gap-4">
                  {similarity_review && (
                    <div className="border border-border rounded-xl p-5 bg-card text-center">
                      <p className="text-3xl font-bold text-foreground">
                        {Math.round(similarity_review.overall_similarity_percent)}%
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">Similarity Score</p>
                    </div>
                  )}
                  {ai_writing_risk_review && (
                    <div className="border border-border rounded-xl p-5 bg-card text-center">
                      <p className="text-3xl font-bold text-foreground">
                        {Math.round(ai_writing_risk_review.overall_ai_probability * 100)}%
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">AI Writing Probability</p>
                    </div>
                  )}
                </div>
              )}

              <div className="border border-border rounded-xl p-6 bg-card">
                <h3 className="font-semibold text-foreground mb-2">Overall Assessment</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {final_summary.overall_assessment}
                </p>
              </div>

              <div className="border border-border rounded-xl p-6 bg-card">
                <h3 className="font-semibold text-foreground mb-3">Top Priority Revisions</h3>
                <ol className="list-decimal list-inside space-y-1.5 text-sm text-muted-foreground">
                  {final_summary.top_10_priority_revisions.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ol>
              </div>

              <RecommendationBanner recommendation={final_summary.final_recommendation} />
            </TabsContent>

            {/* ─── Line Review ─── */}
            <TabsContent value="line" className="mt-6 space-y-4">
              {language_review.issues.length === 0 ? (
                <p className="text-sm text-muted-foreground">No language issues detected.</p>
              ) : (
                language_review.issues.map((issue, i) => (
                  <div key={i} className="border border-border rounded-xl p-5 bg-card">
                    <div className="flex flex-wrap items-center gap-2 mb-3">
                      <Badge variant="outline" className="text-xs">{issue.paragraph_line}</Badge>
                      <Badge variant="outline" className="text-xs">{issue.issue_type}</Badge>
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${severityColor[issue.severity] || ""}`}>
                        {issue.severity}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground italic mb-2">"{issue.excerpt}"</p>
                    <p className="text-sm text-foreground mb-1">
                      <strong>Problem:</strong> {issue.problem}
                    </p>
                    <p className="text-sm text-primary">
                      <strong>Suggested:</strong> {issue.suggested_correction}
                    </p>
                  </div>
                ))
              )}
            </TabsContent>

            {/* ─── Technical ─── */}
            <TabsContent value="technical" className="mt-6">
              <div className="border border-border rounded-xl p-6 bg-card">
                <ConcernList title="Major Scientific Concerns" items={technical_review.major_scientific_concerns} />
                <ConcernList title="Methodological Concerns" items={technical_review.methodological_concerns} />
                <ConcernList title="Statistical Concerns" items={technical_review.statistical_concerns} />
                <ConcernList title="Interpretation Problems" items={technical_review.interpretation_problems} />
                {totalTechnicalConcerns === 0 && (
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-600" /> No technical concerns identified.
                  </p>
                )}
              </div>
            </TabsContent>

            {/* ─── Citations ─── */}
            <TabsContent value="citations" className="mt-6">
              <div className="border border-border rounded-xl p-6 bg-card">
                <CitationSection title="Citations Missing from References" items={citation_audit.missing_from_references} />
                <CitationSection title="References Not Cited in Text" items={citation_audit.uncited_references} />
                <CitationSection title="Inconsistencies" items={citation_audit.inconsistencies} />
                <CitationSection title="Formatting Issues" items={citation_audit.formatting_issues} />
              </div>
            </TabsContent>

            {/* ─── Similarity Review ─── */}
            {similarity_review && (
              <TabsContent value="similarity" className="mt-6 space-y-6">
                <div className="border border-border rounded-xl p-6 bg-card">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-foreground">Similarity Overview</h3>
                    <Badge
                      variant="outline"
                      className={
                        similarity_review.overall_similarity_percent < 15
                          ? "border-green-500 text-green-700"
                          : similarity_review.overall_similarity_percent < 30
                          ? "border-yellow-500 text-yellow-700"
                          : "border-destructive text-destructive"
                      }
                    >
                      {Math.round(similarity_review.overall_similarity_percent)}% Overall
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">{similarity_review.verdict}</p>

                  {(similarity_review.flagged_sections ?? []).length === 0 ? (
                    <p className="text-sm text-muted-foreground flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-600" /> No flagged sections.
                    </p>
                  ) : (
                    <div className="space-y-3">
                      <h4 className="font-semibold text-foreground text-sm">Flagged Sections</h4>
                      {(similarity_review.flagged_sections ?? []).map((s, i) => (
                        <div key={i} className="border border-border rounded-lg p-4 bg-muted/20">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-muted-foreground">{s.source}</span>
                            <Badge variant="outline" className="text-xs">{Math.round(s.similarity_percent)}%</Badge>
                          </div>
                          <p className="text-sm text-foreground italic">"{s.matched_text}"</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </TabsContent>
            )}

            {/* ─── AI Risk ─── */}
            {ai_writing_risk_review && (
              <TabsContent value="airisk" className="mt-6 space-y-6">
                <div className="border border-border rounded-xl p-6 bg-card">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-foreground">AI Writing Risk Assessment</h3>
                    <Badge
                      variant="outline"
                      className={
                        ai_writing_risk_review.overall_ai_probability < 0.2
                          ? "border-green-500 text-green-700"
                          : ai_writing_risk_review.overall_ai_probability < 0.5
                          ? "border-yellow-500 text-yellow-700"
                          : "border-destructive text-destructive"
                      }
                    >
                      {Math.round(ai_writing_risk_review.overall_ai_probability * 100)}% AI Probability
                    </Badge>
                  </div>

                  <div className="mb-4">
                    <p className="text-xs text-muted-foreground mb-1">Overall AI Probability</p>
                    <AiProbabilityBar value={ai_writing_risk_review.overall_ai_probability} />
                  </div>

                  <p className="text-sm text-muted-foreground mb-4">{ai_writing_risk_review.verdict}</p>

                  {(ai_writing_risk_review.flagged_sections ?? []).length === 0 ? (
                    <p className="text-sm text-muted-foreground flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-600" /> No flagged sections.
                    </p>
                  ) : (
                    <div className="space-y-3">
                      <h4 className="font-semibold text-foreground text-sm">Flagged Sections</h4>
                      {(ai_writing_risk_review.flagged_sections ?? []).map((s, i) => (
                        <div key={i} className="border border-border rounded-lg p-4 bg-muted/20">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-foreground">{s.section}</span>
                            <Badge variant="outline" className="text-xs">{Math.round(s.ai_probability * 100)}%</Badge>
                          </div>
                          <AiProbabilityBar value={s.ai_probability} />
                          <p className="text-sm text-muted-foreground mt-2">{s.explanation}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </TabsContent>
            )}

            {/* ─── Final Verdict ─── */}
            <TabsContent value="verdict" className="mt-6 space-y-6">
              <RecommendationBanner recommendation={final_summary.final_recommendation} />

              <div className="border border-border rounded-xl p-6 bg-card">
                <h3 className="font-semibold text-foreground mb-2">Overall Assessment</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {final_summary.overall_assessment}
                </p>
              </div>

              <div className="border border-border rounded-xl p-6 bg-card">
                <h3 className="font-semibold text-foreground mb-3">Top 10 Priority Revisions</h3>
                <ol className="list-decimal list-inside space-y-1.5 text-sm text-muted-foreground">
                  {final_summary.top_10_priority_revisions.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ol>
              </div>
            </TabsContent>

            {/* ─── Downloads ─── */}
            {exports_available && (
              <TabsContent value="downloads" className="mt-6">
                <div className="border border-border rounded-xl p-6 bg-card">
                  <h3 className="font-semibold text-foreground mb-4">Download Reports</h3>
                  <p className="text-sm text-muted-foreground mb-6">
                    Download individual review reports generated by the backend.
                  </p>
                  <div className="grid sm:grid-cols-2 gap-3">
                    <DownloadButton label="Full Review Report" url={exports_available.full_review_report} />
                    <DownloadButton label="Comments to Author" url={exports_available.comments_to_author} />
                    <DownloadButton label="Confidential Editor Note" url={exports_available.confidential_editor_note} />
                    <DownloadButton label="Citation Review" url={exports_available.citation_review} />
                    <DownloadButton label="Similarity Review" url={exports_available.similarity_review} />
                    <DownloadButton label="AI Risk Review" url={exports_available.ai_risk_review} />
                    <DownloadButton label="Annotated Manuscript" url={exports_available.annotated_manuscript} />
                  </div>
                </div>
              </TabsContent>
            )}
          </Tabs>
        </div>
      </section>
    </Layout>
  );
}

function SummaryCard({ label, count }: { label: string; count: number }) {
  return (
    <div className="border border-border rounded-xl p-5 bg-card text-center">
      <p className="text-3xl font-bold text-foreground">{count}</p>
      <p className="text-sm text-muted-foreground mt-1">{label}</p>
    </div>
  );
}

function RecommendationBanner({ recommendation }: { recommendation: string }) {
  const bg = recommendationColor[recommendation] || "bg-primary";
  return (
    <div className={`rounded-xl px-6 py-5 text-center ${bg} text-white`}>
      <p className="text-xs uppercase tracking-widest font-medium mb-1 opacity-80">
        Editorial Recommendation
      </p>
      <p className="text-2xl font-bold">{recommendation}</p>
    </div>
  );
}
