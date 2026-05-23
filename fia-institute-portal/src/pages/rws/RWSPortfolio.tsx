import { useEffect, useState } from "react";
import { useNavigate, Link, useSearchParams } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  ArrowLeft, Award, BookOpen, Brain, FlaskConical, GraduationCap,
  Loader2, Shield, Target, User, Download, Link2, Lock, Eye,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";
import jsPDF from "jspdf";

const COMPETENCY_META: Record<string, { label: string; icon: any }> = {
  research_structure: { label: "Research Structure", icon: BookOpen },
  literature_synthesis: { label: "Literature Synthesis", icon: Brain },
  statistical_interpretation: { label: "Statistical Interpretation", icon: FlaskConical },
  results_writing: { label: "Results Writing", icon: Target },
  scientific_reasoning: { label: "Scientific Reasoning", icon: GraduationCap },
  defense_readiness: { label: "Defense Readiness", icon: Shield },
  clarity_of_explanation: { label: "Clarity of Explanation", icon: Brain },
  understanding_of_methods: { label: "Understanding of Methods", icon: FlaskConical },
  data_interpretation: { label: "Data Interpretation", icon: Target },
  critical_thinking: { label: "Critical Thinking", icon: GraduationCap },
};

const TRACK_LABELS: Record<string, string> = {
  undergraduate_project: "Undergraduate Final-Year Project",
  msc_thesis: "MSc Thesis Development",
  phd_research: "PhD Research Writing and Defense",
  research_paper: "Research Paper Writing",
};

const LEVEL_LABELS: Record<string, string> = {
  beginner: "Beginner Researcher",
  developing: "Developing Researcher",
  advanced: "Advanced Researcher",
};

const VISIBILITY_OPTIONS = [
  { value: "private", label: "Private", icon: Lock },
  { value: "supervisor", label: "Supervisor Only", icon: Eye },
  { value: "shareable", label: "Shareable Link", icon: Link2 },
];

export default function RWSPortfolio() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user, profile, loading } = useAuth();
  const [competencies, setCompetencies] = useState<any[]>([]);
  const [milestones, setMilestones] = useState<any[]>([]);
  const [defenseAttempts, setDefenseAttempts] = useState<any[]>([]);
  const [conversationCount, setConversationCount] = useState(0);
  const [certStatus, setCertStatus] = useState<string>("not_eligible");
  const [fetching, setFetching] = useState(true);
  const [visibility, setVisibility] = useState("private");
  const [shareToken, setShareToken] = useState<string | null>(null);

  // Check if viewing via share token
  const shareView = searchParams.get("token");

  useEffect(() => {
    if (!loading && !user && !shareView) navigate("/research-writing/signin");
  }, [loading, user, shareView]);

  useEffect(() => {
    if (shareView) {
      // Load portfolio via share token
      const fetchShared = async () => {
        const { data: prof } = await supabase
          .from("profiles")
          .select("*")
          .eq("portfolio_share_token", shareView)
          .eq("portfolio_visibility", "shareable")
          .single();
        if (!prof) { setFetching(false); return; }
        const [compRes, mileRes, defRes] = await Promise.all([
          supabase.from("competency_scores").select("*").eq("user_id", prof.id),
          supabase.from("milestones").select("*").eq("user_id", prof.id).eq("is_completed", true).order("created_at"),
          supabase.from("defense_simulator_attempts").select("*").eq("user_id", prof.id).not("completed_at", "is", null).order("total_score", { ascending: false }).limit(1),
        ]);
        if (compRes.data) setCompetencies(compRes.data);
        if (mileRes.data) setMilestones(mileRes.data);
        if (defRes.data) setDefenseAttempts(defRes.data);
        setFetching(false);
      };
      fetchShared();
      return;
    }

    if (!user) return;
    const fetchAll = async () => {
      const [compRes, mileRes, defRes, convRes, certRes] = await Promise.all([
        supabase.from("competency_scores").select("*").eq("user_id", user.id),
        supabase.from("milestones").select("*").eq("user_id", user.id).order("created_at"),
        supabase.from("defense_simulator_attempts").select("*").eq("user_id", user.id).not("completed_at", "is", null).order("total_score", { ascending: false }).limit(3),
        supabase.from("ai_conversations").select("id", { count: "exact" }).eq("user_id", user.id),
        supabase.from("certificate_eligibility").select("certificate_status").eq("user_id", user.id).single(),
      ]);
      if (compRes.data) setCompetencies(compRes.data);
      if (mileRes.data) setMilestones(mileRes.data);
      if (defRes.data) setDefenseAttempts(defRes.data);
      if (convRes.count !== null) setConversationCount(convRes.count);
      if (certRes.data) setCertStatus(certRes.data.certificate_status);

      // Get privacy settings
      if (profile) {
        setVisibility((profile as any).portfolio_visibility || "private");
        setShareToken((profile as any).portfolio_share_token || null);
      }
      setFetching(false);
    };
    fetchAll();
  }, [user, profile, shareView]);

  const handleVisibilityChange = async (value: string) => {
    if (!user) return;
    const updates: any = { portfolio_visibility: value };
    if (value === "shareable" && !shareToken) {
      const token = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
      updates.portfolio_share_token = token;
      setShareToken(token);
    }
    await supabase.from("profiles").update(updates).eq("id", user.id);
    setVisibility(value);
    toast({ title: "Privacy updated", description: `Portfolio is now ${value}.` });
  };

  const copyShareLink = () => {
    if (!shareToken) return;
    const url = `${window.location.origin}/research-writing/portfolio?token=${shareToken}`;
    navigator.clipboard.writeText(url);
    toast({ title: "Link copied", description: "Share this link with your supervisor or institution." });
  };

  const exportPDF = () => {
    if (!profile) return;
    const doc = new jsPDF();
    const margin = 20;
    let y = margin;

    doc.setFontSize(18);
    doc.text("Research Competency Portfolio", margin, y);
    y += 12;
    doc.setFontSize(10);
    doc.text("Field-to-Insight Academy (FIA)", margin, y);
    y += 10;
    doc.setDrawColor(0);
    doc.line(margin, y, 190, y);
    y += 10;

    doc.setFontSize(12);
    doc.text("Student Profile", margin, y);
    y += 8;
    doc.setFontSize(10);
    doc.text(`Name: ${profile.full_name}`, margin, y); y += 6;
    doc.text(`Track: ${TRACK_LABELS[profile.academic_track || ""] || "—"}`, margin, y); y += 6;
    if (profile.discipline) { doc.text(`Discipline: ${profile.discipline}`, margin, y); y += 6; }
    if (profile.institution) { doc.text(`Institution: ${profile.institution}`, margin, y); y += 6; }
    doc.text(`Diagnostic Level: ${LEVEL_LABELS[profile.diagnostic_level || ""] || "Not assessed"}`, margin, y); y += 6;
    doc.text(`AI Mentoring Sessions: ${conversationCount}`, margin, y); y += 6;
    doc.text(`Certificate Status: ${certStatus.replace(/_/g, " ")}`, margin, y); y += 12;

    if (competencies.length > 0) {
      doc.setFontSize(12);
      doc.text("Competency Scores", margin, y); y += 8;
      doc.setFontSize(10);
      competencies.forEach((c) => {
        const label = COMPETENCY_META[c.category]?.label || c.category;
        doc.text(`${label}: ${c.score}%`, margin, y); y += 6;
      });
      y += 6;
    }

    const completedMilestones = milestones.filter((m) => m.is_completed);
    if (completedMilestones.length > 0) {
      doc.setFontSize(12);
      doc.text("Completed Milestones", margin, y); y += 8;
      doc.setFontSize(10);
      completedMilestones.forEach((m) => {
        doc.text(`• ${m.title} (${m.stage})`, margin, y); y += 6;
      });
      y += 6;
    }

    if (defenseAttempts.length > 0) {
      doc.setFontSize(12);
      doc.text("Defense Performance", margin, y); y += 8;
      doc.setFontSize(10);
      doc.text(`Best Score: ${defenseAttempts[0].total_score}/100`, margin, y); y += 10;
    }

    doc.setFontSize(8);
    doc.setTextColor(128);
    doc.text(`Generated on ${new Date().toLocaleDateString()} — Field-to-Insight Academy`, margin, 285);

    doc.save(`FIA_Portfolio_${profile.full_name.replace(/\s+/g, "_")}.pdf`);
    toast({ title: "PDF exported", description: "Your portfolio has been downloaded." });
  };

  if (loading || fetching) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (shareView && competencies.length === 0 && milestones.length === 0) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="max-w-md"><CardContent className="pt-6 text-center">
          <Lock className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">This portfolio is not available or has been set to private.</p>
        </CardContent></Card>
      </div>
    );
  }

  if (!profile && !shareView) return null;

  const completedMilestones = milestones.filter((m) => m.is_completed);
  const bestDefense = defenseAttempts[0];
  const strongAreas = competencies.filter((c) => c.score >= 70).map((c) => c.category);
  const weakAreas = competencies.filter((c) => c.score < 50).map((c) => c.category);

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-4xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold flex items-center gap-2">
              <User className="w-6 h-6" /> Research Competency Portfolio
            </h1>
          </div>
          <div className="flex items-center gap-2">
            {!shareView && (
              <>
                <Button variant="secondary" size="sm" className="gap-1" onClick={exportPDF}>
                  <Download className="w-4 h-4" /> PDF
                </Button>
                <Link to="/research-writing/dashboard">
                  <Button variant="secondary" size="sm" className="gap-1"><ArrowLeft className="w-4 h-4" /> Dashboard</Button>
                </Link>
              </>
            )}
          </div>
        </div>
      </header>

      <div className="container max-w-4xl py-8 space-y-6">
        {/* Privacy Settings */}
        {!shareView && (
          <Card>
            <CardContent className="pt-4 pb-4">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-sm font-medium">Portfolio Visibility:</span>
                <Select value={visibility} onValueChange={handleVisibilityChange}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VISIBILITY_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <span className="flex items-center gap-1.5">
                          <opt.icon className="w-3 h-3" /> {opt.label}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {visibility === "shareable" && shareToken && (
                  <Button variant="outline" size="sm" className="gap-1" onClick={copyShareLink}>
                    <Link2 className="w-3 h-3" /> Copy Link
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Profile Summary */}
        <Card>
          <CardContent className="pt-6">
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h2 className="font-serif text-xl font-bold">{profile?.full_name}</h2>
                <p className="text-sm text-muted-foreground">{profile?.email}</p>
                <p className="text-sm">{TRACK_LABELS[profile?.academic_track || ""] || "—"}</p>
                {profile?.discipline && <p className="text-sm text-muted-foreground">{profile.discipline}</p>}
                {profile?.institution && <p className="text-sm text-muted-foreground">{profile.institution}</p>}
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground uppercase tracking-wider">Diagnostic Level</span>
                  <Badge variant="secondary">{LEVEL_LABELS[profile?.diagnostic_level || ""] || "Not assessed"}</Badge>
                </div>
                {!shareView && (
                  <>
                    <p className="text-sm text-muted-foreground">{conversationCount} AI mentoring session{conversationCount !== 1 ? "s" : ""}</p>
                    <p className="text-sm text-muted-foreground">{completedMilestones.length} milestone{completedMilestones.length !== 1 ? "s" : ""} completed</p>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground uppercase tracking-wider">Certificate</span>
                      <Badge variant={certStatus === "issued" ? "default" : "secondary"}>{certStatus.replace(/_/g, " ")}</Badge>
                    </div>
                  </>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Competency Scores */}
        <Card>
          <CardHeader><CardTitle className="text-lg">Competency Profile</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {competencies.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                Complete your defense simulation to generate competency scores.
              </p>
            ) : (
              competencies.map((c) => {
                const meta = COMPETENCY_META[c.category] || { label: c.category, icon: Target };
                return (
                  <div key={c.id}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm flex items-center gap-1.5">
                        <meta.icon className="w-4 h-4 text-primary" />
                        {meta.label}
                      </span>
                      <span className="text-sm font-medium">{c.score}%</span>
                    </div>
                    <Progress value={c.score} className="h-2" />
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>

        {/* Defense Performance */}
        {bestDefense && (
          <Card>
            <CardHeader><CardTitle className="text-lg flex items-center gap-2"><Shield className="w-5 h-5" /> Defense Performance</CardTitle></CardHeader>
            <CardContent>
              <div className="text-center mb-4">
                <p className="text-3xl font-bold text-primary">{bestDefense.total_score}/100</p>
                <p className="text-xs text-muted-foreground">Best score from {defenseAttempts.length} attempt{defenseAttempts.length !== 1 ? "s" : ""}</p>
              </div>
              {bestDefense.ai_feedback_summary && (
                <p className="text-sm text-muted-foreground">{bestDefense.ai_feedback_summary.substring(0, 300)}...</p>
              )}
            </CardContent>
          </Card>
        )}

        {/* Completed Milestones */}
        {completedMilestones.length > 0 && (
          <Card>
            <CardHeader><CardTitle className="text-lg">Completed Milestones</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-2">
                {completedMilestones.map((m) => (
                  <div key={m.id} className="flex items-center gap-2 p-2 rounded border border-border">
                    <Award className="w-4 h-4 text-green-600" />
                    <div>
                      <p className="text-sm font-medium">{m.title}</p>
                      <p className="text-xs text-muted-foreground">{m.stage} • {new Date(m.completed_at || m.created_at).toLocaleDateString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Strengths & Weaknesses */}
        {(strongAreas.length > 0 || weakAreas.length > 0) && (
          <div className="grid sm:grid-cols-2 gap-4">
            {strongAreas.length > 0 && (
              <Card>
                <CardHeader><CardTitle className="text-base text-green-600">Strengths</CardTitle></CardHeader>
                <CardContent>
                  {strongAreas.map((a) => (
                    <p key={a} className="text-sm text-muted-foreground">• {COMPETENCY_META[a]?.label || a}</p>
                  ))}
                </CardContent>
              </Card>
            )}
            {weakAreas.length > 0 && (
              <Card>
                <CardHeader><CardTitle className="text-base text-orange-600">Areas for Improvement</CardTitle></CardHeader>
                <CardContent>
                  {weakAreas.map((a) => (
                    <p key={a} className="text-sm text-muted-foreground">• {COMPETENCY_META[a]?.label || a}</p>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
