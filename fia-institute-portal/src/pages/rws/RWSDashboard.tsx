import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { ReadingLogSection } from "@/components/rws/ReadingLogSection";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { supabase } from "@/integrations/supabase/client";
import {
  GraduationCap, BookOpen, ArrowRight, Brain, FlaskConical, Target,
  MessageSquare, Shield, Lock, CheckCircle, FileText, Microscope, Award,
  RefreshCw, Calendar, User, BarChart3, Zap, Flame,
  ChevronDown, ChevronUp, Download, Mail, Loader2, Pencil, Save, X,
} from "lucide-react";
import { NotificationBell } from "@/components/rws/NotificationBell";

const TRACK_LABELS: Record<string, string> = {
  undergraduate_project: "Undergraduate Final-Year Project",
  msc_thesis: "MSc Thesis Development",
  phd_research: "PhD Research Writing and Defense",
  research_paper: "Research Paper Writing",
};

const LEVEL_META: Record<string, { label: string; color: string }> = {
  beginner:  { label: "Beginner Researcher",   color: "text-orange-600" },
  developing:{ label: "Developing Researcher",  color: "text-blue-600" },
  advanced:  { label: "Advanced Researcher",    color: "text-primary" },
};

const STAGE_LABELS: Record<string, string> = {
  topic_proposal:      "Topic / Proposal",
  literature_review:   "Literature Review",
  methodology:         "Methodology",
  data_analysis:       "Data Analysis",
  results_writing:     "Results Writing",
  discussion:          "Discussion",
  defense_preparation: "Defense Preparation",
};

const ROADMAP = [
  { key: "topic_proposal",      label: "Proposal / Topic",        icon: FileText },
  { key: "literature_review",   label: "Literature Review",        icon: BookOpen },
  { key: "methodology",         label: "Methodology",              icon: FlaskConical },
  { key: "data_analysis",       label: "Data Analysis",            icon: Microscope },
  { key: "results_writing",     label: "Results Interpretation",   icon: Target },
  { key: "discussion",          label: "Discussion",               icon: MessageSquare },
  { key: "defense_preparation", label: "Defense",                  icon: Shield },
  { key: "certification",       label: "Certification",            icon: Award },
];

const MODULES = [
  { key: "thesis_mentor", label: "FIA Research Writing Mentor",       description: "AI-assisted guidance for proposal, thesis, and research paper development",    href: "/thesis-mentor",               available: true, icon: Brain },
  { key: "results_lab",   label: "Guided Results Interpretation Lab", description: "Interpret ANOVA, correlations, PCA, and more with the 4-Level Framework",    href: "/research-writing/results-lab", available: true, icon: FlaskConical },
  { key: "defense_sim",   label: "Defense Simulator",                 description: "Practice defending your research findings with AI viva questions",             href: "/research-writing/defense",     available: true, icon: Shield },
  { key: "supervisor",    label: "Supervisor Guidance",               description: "Structured supervisor-student interaction tools",                              href: "/research-writing/bookings",    available: true, icon: GraduationCap },
];

const NEXT_ACTIONS = [
  { condition: "no_diagnostic",       label: "Complete Diagnostic Assessment",     description: "Identify your knowledge gaps and starting level",         href: "/research-writing/diagnostic" },
  { condition: "topic_proposal",      label: "Open Research Writing Mentor",       description: "Start developing your proposal with AI guidance",         href: "/thesis-mentor" },
  { condition: "literature_review",   label: "Review Literature with AI Guide",    description: "Use Guide mode to synthesise your literature review",      href: "/thesis-mentor" },
  { condition: "methodology",         label: "Develop Your Methodology",           description: "Use AI guidance to structure your methods chapter",        href: "/thesis-mentor" },
  { condition: "data_analysis",       label: "Interpret Your Results",             description: "Upload statistical output to the Results Interpretation Lab", href: "/research-writing/results-lab" },
  { condition: "results_writing",     label: "Write Your Results Section",         description: "Use the AI mentor to structure your results chapter",      href: "/thesis-mentor" },
  { condition: "discussion",          label: "Build Your Discussion",              description: "Connect your results to the literature",                   href: "/thesis-mentor" },
  { condition: "defense_preparation", label: "Prepare for Your Defense",           description: "Use the Defense Simulator for viva practice",              href: "/research-writing/defense" },
];

interface Briefing {
  id: string;
  briefing_text: string;
  mode: string;
  stage: string;
  topic: string;
  exchange_count: number;
  created_at: string;
}

export default function RWSDashboard() {
  const navigate = useNavigate();
  const { user, profile, loading } = useAuth() as {
    user: { id: string } | null;
    profile: Record<string, unknown> | null;
    loading: boolean;
  };

  const [milestones,         setMilestones]         = useState<unknown[]>([]);
  const [conversationCount,  setConversationCount]  = useState(0);
  const [defenseAttempt,     setDefenseAttempt]     = useState<Record<string, unknown> | null>(null);
  const [certStatus,         setCertStatus]         = useState<string | null>(null);
  const [bookings,           setBookings]           = useState<unknown[]>([]);
  const [streak,             setStreak]             = useState({ current: 0, longest: 0 });
  const [drillCount,         setDrillCount]         = useState(0);
  const [badgeCount,         setBadgeCount]         = useState(0);
  const [briefings,          setBriefings]          = useState<Briefing[]>([]);
  const [expandedBriefingId, setExpandedBriefingId] = useState<string | null>(null);

  // Profile edit state
  const [editingProfile, setEditingProfile]   = useState(false);
  const [thesisTitle,    setThesisTitle]      = useState("");
  const [supervisorEmail,setSupervisorEmail]  = useState("");
  const [savingProfile,  setSavingProfile]    = useState(false);
  const [profileSaved,   setProfileSaved]     = useState(false);

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
    if (!loading && user && profile && !profile.onboarding_completed)
      navigate("/research-writing/onboarding");
  }, [loading, user, profile]);

  // Initialise edit fields from profile when it loads
  useEffect(() => {
    if (profile) {
      setThesisTitle((profile.thesis_title as string) || "");
      setSupervisorEmail((profile.supervisor_email as string) || "");
    }
  }, [profile]);

  useEffect(() => {
    if (!user) return;
    const fetchData = async () => {
      const [
        milestonesRes, convsRes, defenseRes, certRes,
        bookingsRes, streakRes, drillRes, badgeRes, briefingsRes,
      ] = await Promise.all([
        supabase.from("milestones").select("*").eq("user_id", user.id).order("created_at"),
        supabase.from("ai_conversations").select("id", { count: "exact" }).eq("user_id", user.id),
        supabase.from("defense_simulator_attempts").select("*").eq("user_id", user.id).not("completed_at", "is", null).order("total_score", { ascending: false }).limit(1),
        supabase.from("certificate_eligibility").select("certificate_status").eq("user_id", user.id).single(),
        supabase.from("booking_requests").select("*").eq("student_id", user.id).order("created_at", { ascending: false }).limit(3),
        supabase.from("user_streaks").select("*").eq("user_id", user.id).single(),
        supabase.from("daily_drill_attempts").select("id", { count: "exact" }).eq("user_id", user.id),
        supabase.from("user_badges").select("id", { count: "exact" }).eq("user_id", user.id),
        supabase.from("supervisor_briefings" as never).select("*").eq("user_id", user.id).order("created_at", { ascending: false }).limit(5),
      ]);

      if (milestonesRes.data)     setMilestones(milestonesRes.data);
      if (convsRes.count !== null) setConversationCount(convsRes.count);
      if (defenseRes.data?.[0])   setDefenseAttempt(defenseRes.data[0] as Record<string, unknown>);
      if (certRes.data)           setCertStatus((certRes.data as Record<string, unknown>).certificate_status as string);
      if (bookingsRes.data)       setBookings(bookingsRes.data);
      if (streakRes.data)         setStreak({ current: (streakRes.data as Record<string, unknown>).current_streak as number, longest: (streakRes.data as Record<string, unknown>).longest_streak as number });
      if (drillRes.count !== null) setDrillCount(drillRes.count);
      if (badgeRes.count !== null) setBadgeCount(badgeRes.count);
      if (briefingsRes.data)      setBriefings(briefingsRes.data as Briefing[]);
    };
    fetchData();
  }, [user]);

  const handleSaveProfile = async () => {
    if (!user) return;
    setSavingProfile(true);
    setProfileSaved(false);
    try {
      const { error } = await supabase
        .from("profiles")
        .update({ thesis_title: thesisTitle.trim(), supervisor_email: supervisorEmail.trim() })
        .eq("id", user.id);
      if (error) throw error;
      setProfileSaved(true);
      setEditingProfile(false);
    } catch {
      // show inline error if needed — keep it minimal for now
    } finally {
      setSavingProfile(false);
    }
  };

  const handleDownloadBriefingPDF = (b: Briefing) => {
    const w = window.open("", "_blank");
    if (!w) return;
    const date = new Date(b.created_at).toLocaleDateString("en-GB", {
      day: "numeric", month: "long", year: "numeric",
    });
    const stageLabel = STAGE_LABELS[b.stage] || b.stage.replace(/_/g, " ");
    const safeText = b.briefing_text
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    w.document.write(`<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Supervisor Briefing — ${date}</title>
<style>
  body{font-family:Georgia,serif;max-width:680px;margin:48px auto;line-height:1.8;color:#1a1a1a;font-size:13px}
  .hdr{border-bottom:2px solid #1a5c38;padding-bottom:14px;margin-bottom:22px}
  h1{font-size:17px;color:#1a5c38;margin:0 0 3px}
  .inst{font-size:11px;color:#555;margin:0}
  .meta{display:grid;grid-template-columns:1fr 1fr;gap:3px 28px;font-size:12px;color:#333;margin-bottom:24px}
  .meta strong{color:#111}
  .body{font-size:13px;line-height:1.85;white-space:pre-wrap}
  .ftr{margin-top:44px;padding-top:10px;border-top:1px solid #ccc;font-size:10px;color:#888}
</style></head><body>
<div class="hdr"><h1>Supervisor Briefing Note</h1>
<p class="inst">Field-to-Insight Academy — Research Writing Mentor</p></div>
<div class="meta">
  <div><strong>Stage:</strong> ${stageLabel}</div>
  <div><strong>Date:</strong> ${date}</div>
  <div><strong>Mode:</strong> ${b.mode}</div>
  <div><strong>Exchanges:</strong> ${b.exchange_count}</div>
</div>
<div class="body">${safeText}</div>
<div class="ftr">Generated by the FIA Research Writing Mentor · NOT the AI conversation text · fieldtoinsightacademy.com.ng</div>
<script>window.onload=()=>{window.print();}<\/script>
</body></html>`);
    w.document.close();
  };

  const handleEmailBriefing = (b: Briefing) => {
    const date = new Date(b.created_at).toLocaleDateString("en-GB", {
      day: "numeric", month: "long", year: "numeric",
    });
    const to = ((profile?.supervisor_email as string) || "");
    const studentName = (profile?.full_name as string) || "your student";
    const subject = encodeURIComponent(`Research Progress Briefing — ${studentName} — ${date}`);
    const body = encodeURIComponent(
      `Dear Supervisor,\n\nPlease find below a progress briefing from ${studentName}'s FIA Research Writing Mentor session (${date}).\n\n` +
      `─────────────────────────────────────────\n\n${b.briefing_text}\n\n` +
      `─────────────────────────────────────────\n\n` +
      `AI-assisted professional summary — not a transcript.\nField-to-Insight Academy · fieldtoinsightacademy.com.ng`
    );
    window.open(`mailto:${to}?subject=${subject}&body=${body}`, "_blank");
  };

  if (loading || !profile) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  const levelMeta = LEVEL_META[profile.diagnostic_level as string || "beginner"] || LEVEL_META.beginner;
  const currentStageIdx = ROADMAP.findIndex((r) => r.key === profile.current_research_stage);
  const progressPct = currentStageIdx >= 0 ? Math.round(((currentStageIdx + 1) / ROADMAP.length) * 100) : 10;

  let nextAction = NEXT_ACTIONS[0];
  if (profile.diagnostic_level !== null && profile.diagnostic_level !== undefined) {
    const stageAction = NEXT_ACTIONS.find((a) => a.condition === profile.current_research_stage);
    nextAction = stageAction || { condition: "default", label: "Open Research Writing Mentor", description: "Continue your research journey", href: "/thesis-mentor" };
  }

  const completedMilestones = (milestones as Array<Record<string, unknown>>).filter((m) => m.is_completed).length;
  const hasThesisInfo = !!(profile.thesis_title || profile.supervisor_email);

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-5xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">
              FIA Research Writing System
            </p>
            <h1 className="font-serif text-2xl font-bold">
              Welcome, {(profile.full_name as string) || "Researcher"}
            </h1>
          </div>
          <div className="flex items-center gap-2">
            {user && <NotificationBell userId={user.id} />}
            <Link to="/research-writing">
              <Button variant="secondary" size="sm">Back to Home</Button>
            </Link>
          </div>
        </div>
      </header>

      <div className="container max-w-5xl py-8 space-y-8">

        {/* ── Status Cards ─────────────────────────────────────────────────── */}
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Track</p>
            <p className="font-medium text-foreground mt-1 text-sm">
              {TRACK_LABELS[profile.academic_track as string || ""] || "Not set"}
            </p>
          </CardContent></Card>
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Diagnostic Level</p>
            <p className={`font-medium mt-1 text-sm ${levelMeta.color}`}>{levelMeta.label}</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Research Stage</p>
            <p className="font-medium text-foreground mt-1 text-sm">
              {STAGE_LABELS[profile.current_research_stage as string || ""] || "Not set"}
            </p>
          </CardContent></Card>
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">AI Sessions</p>
            <p className="font-medium text-foreground mt-1 text-sm">
              {conversationCount} conversation{conversationCount !== 1 ? "s" : ""}
            </p>
          </CardContent></Card>
        </div>

        {/* ── Next Action ──────────────────────────────────────────────────── */}
        <Card className="border-l-4 border-l-accent">
          <CardContent className="pt-6 flex items-center justify-between gap-4">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Next Action</p>
              <p className="font-semibold text-foreground">{nextAction.label}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{nextAction.description}</p>
            </div>
            <Link to={nextAction.href}>
              <Button className="gap-2 shrink-0">Start <ArrowRight className="w-4 h-4" /></Button>
            </Link>
          </CardContent>
        </Card>

        {/* ── Phase 3: Defense, Bookings, Certificate ──────────────────────── */}
        <div className="grid md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Defense Simulator</h3>
              </div>
              {defenseAttempt ? (
                <>
                  <p className="text-2xl font-bold text-primary">{defenseAttempt.total_score as number}/100</p>
                  <p className="text-xs text-muted-foreground">Best score</p>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">Not attempted yet</p>
              )}
              <Link to="/research-writing/defense">
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">
                  {defenseAttempt ? "Try Again" : "Start Simulation"} <ArrowRight className="w-3 h-3" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Calendar className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Supervisor Sessions</h3>
              </div>
              {(bookings as Array<Record<string, unknown>>).length > 0 ? (
                <div className="space-y-1">
                  {(bookings as Array<Record<string, unknown>>).slice(0, 2).map((b) => (
                    <p key={b.id as string} className="text-xs text-muted-foreground">
                      {(b.milestone_type as string).replace(/_/g, " ")} —{" "}
                      <Badge variant="secondary" className="text-[10px]">{b.booking_status as string}</Badge>
                    </p>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No sessions booked</p>
              )}
              <Link to="/research-writing/bookings">
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">
                  Manage Bookings <ArrowRight className="w-3 h-3" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Award className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Certificate</h3>
              </div>
              <Badge variant={certStatus === "issued" ? "default" : "secondary"}>
                {(certStatus || "not_eligible").replace(/_/g, " ")}
              </Badge>
              <Link to="/research-writing/certificate">
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">
                  View Eligibility <ArrowRight className="w-3 h-3" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* ── Research Journey ─────────────────────────────────────────────── */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Research Journey</CardTitle>
              {completedMilestones > 0 && (
                <span className="text-xs text-muted-foreground">
                  {completedMilestones} milestone{completedMilestones !== 1 ? "s" : ""} completed
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Overall Progress</span>
                <span className="font-medium text-foreground">{progressPct}%</span>
              </div>
              <Progress value={progressPct} className="h-2" />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
              {ROADMAP.map((stage, i) => {
                const isActive = stage.key === profile.current_research_stage;
                const isPast   = i < currentStageIdx;
                const isLocked = stage.key === "certification";
                return (
                  <div
                    key={stage.key}
                    className={`p-3 rounded-lg border text-center transition-colors ${
                      isActive ? "border-primary bg-primary/5"
                      : isPast  ? "border-primary/30 bg-primary/5"
                      : "border-border"
                    }`}
                  >
                    <div
                      className="mx-auto w-8 h-8 rounded-full flex items-center justify-center mb-2"
                      style={{ backgroundColor: isActive || isPast ? "hsl(var(--primary))" : "hsl(var(--muted))" }}
                    >
                      {isLocked
                        ? <Lock className="w-4 h-4 text-muted-foreground" />
                        : isPast
                        ? <CheckCircle className="w-4 h-4 text-primary-foreground" />
                        : <stage.icon className={`w-4 h-4 ${isActive ? "text-primary-foreground" : "text-muted-foreground"}`} />}
                    </div>
                    <p className={`text-xs font-medium ${isActive ? "text-primary" : "text-muted-foreground"}`}>
                      {stage.label}
                    </p>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-center pt-2">
              <Link to="/research-writing/onboarding">
                <Button variant="ghost" size="sm" className="gap-1.5 text-xs text-muted-foreground">
                  <RefreshCw className="w-3 h-3" /> Update Research Stage
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* ── Profile Card: Thesis Title + Supervisor Email ────────────────── */}
        <Card className={`border ${hasThesisInfo && !editingProfile ? "border-primary/20" : "border-dashed"}`}>
          <CardContent className="pt-5 pb-5">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <p className="font-medium text-sm text-foreground">Research Profile</p>
                  <p className="text-xs text-muted-foreground">
                    {hasThesisInfo
                      ? "Used in supervisor briefings and PDF exports"
                      : "Add your thesis title and supervisor email to personalise briefings"}
                  </p>
                </div>
              </div>
              {!editingProfile && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-1.5 text-xs text-muted-foreground shrink-0"
                  onClick={() => { setEditingProfile(true); setProfileSaved(false); }}
                >
                  <Pencil className="w-3 h-3" /> Edit
                </Button>
              )}
            </div>

            {!editingProfile && hasThesisInfo && (
              <div className="mt-3 grid sm:grid-cols-2 gap-2 text-xs text-muted-foreground">
                {profile.thesis_title && (
                  <div>
                    <span className="font-medium text-foreground">Thesis / Project Title: </span>
                    {profile.thesis_title as string}
                  </div>
                )}
                {profile.supervisor_email && (
                  <div>
                    <span className="font-medium text-foreground">Supervisor Email: </span>
                    {profile.supervisor_email as string}
                  </div>
                )}
              </div>
            )}

            {editingProfile && (
              <div className="mt-4 space-y-3">
                <div className="space-y-1.5">
                  <Label className="text-xs font-medium">Thesis / Project Title (optional)</Label>
                  <Input
                    value={thesisTitle}
                    onChange={(e) => setThesisTitle(e.target.value)}
                    placeholder="e.g. Heritability and Genetic Advance of Grain Yield in Sorghum"
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-medium">Supervisor Email (optional)</Label>
                  <Input
                    type="email"
                    value={supervisorEmail}
                    onChange={(e) => setSupervisorEmail(e.target.value)}
                    placeholder="supervisor@university.edu.ng"
                    className="text-sm"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Used to pre-fill the email field when you send a briefing. Never shared without your action.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    className="gap-1.5 text-xs"
                    onClick={handleSaveProfile}
                    disabled={savingProfile}
                  >
                    {savingProfile
                      ? <><Loader2 className="w-3 h-3 animate-spin" /> Saving…</>
                      : <><Save className="w-3 h-3" /> Save</>}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs text-muted-foreground"
                    onClick={() => setEditingProfile(false)}
                  >
                    <X className="w-3 h-3 mr-1" /> Cancel
                  </Button>
                  {profileSaved && (
                    <span className="text-xs text-primary flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> Saved
                    </span>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Learning Modules ─────────────────────────────────────────────── */}
        <div>
          <h2 className="font-serif text-xl font-bold text-foreground mb-4">Learning Modules</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {MODULES.map((mod) => (
              <Card key={mod.key} className={!mod.available ? "opacity-60" : ""}>
                <CardContent className="pt-6">
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <mod.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-sm text-foreground">{mod.label}</p>
                      <p className="text-xs text-muted-foreground mt-1">{mod.description}</p>
                      {mod.available && (
                        <Link to={mod.href}>
                          <Button size="sm" variant="outline" className="mt-3 gap-1 text-xs">
                            Open <ArrowRight className="w-3 h-3" />
                          </Button>
                        </Link>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* ── Reading Log ──────────────────────────────────────────────────── */}
        {user && (
          <ReadingLogSection userId={user.id} profile={profile} />
        )}

        {/* ── Supervisor Briefings ─────────────────────────────────────────── */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-serif text-xl font-bold text-foreground">Supervisor Briefings</h2>
            {briefings.length > 0 && (
              <span className="text-xs text-muted-foreground">
                {briefings.length} briefing{briefings.length !== 1 ? "s" : ""} saved
              </span>
            )}
          </div>

          {briefings.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="pt-6 pb-6 text-center space-y-2">
                <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center mx-auto">
                  <FileText className="w-5 h-5 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground font-medium">No briefings yet</p>
                <p className="text-xs text-muted-foreground">
                  After 3 or more exchanges in any AI session, you can generate a professional
                  briefing note for your supervisor.
                </p>
                <Link to="/thesis-mentor">
                  <Button size="sm" variant="outline" className="mt-2 gap-1 text-xs">
                    Start a Session <ArrowRight className="w-3 h-3" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {briefings.map((b) => {
                const isExpanded = expandedBriefingId === b.id;
                const date = new Date(b.created_at).toLocaleDateString("en-GB", {
                  day: "numeric", month: "short", year: "numeric",
                });
                const stageLabel = STAGE_LABELS[b.stage] || b.stage?.replace(/_/g, " ") || "—";

                return (
                  <Card key={b.id} className="border border-border">
                    <CardContent className="pt-4 pb-4">
                      {/* Card header row */}
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-start gap-2.5 min-w-0">
                          <div className="w-7 h-7 rounded-md bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                            <FileText className="w-3.5 h-3.5 text-primary" />
                          </div>
                          <div className="min-w-0">
                            <p className="text-sm font-medium text-foreground truncate">
                              {b.topic ? `"${b.topic}…"` : `${b.mode} mode session`}
                            </p>
                            <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-0.5 text-[11px] text-muted-foreground">
                              <span>{date}</span>
                              <span>{stageLabel}</span>
                              <span className="capitalize">{b.mode} mode</span>
                              <span>{b.exchange_count} exchanges</span>
                            </div>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-xs text-muted-foreground shrink-0 gap-1 -mr-1"
                          onClick={() => setExpandedBriefingId(isExpanded ? null : b.id)}
                        >
                          {isExpanded ? (
                            <><ChevronUp className="w-3 h-3" /> Hide</>
                          ) : (
                            <><ChevronDown className="w-3 h-3" /> Read</>
                          )}
                        </Button>
                      </div>

                      {/* Expanded briefing text */}
                      {isExpanded && (
                        <div className="mt-3 space-y-3">
                          <div className="rounded-lg border border-border bg-muted/20 p-3">
                            <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap font-serif">
                              {b.briefing_text}
                            </p>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              className="gap-1.5 text-xs"
                              onClick={() => handleDownloadBriefingPDF(b)}
                            >
                              <Download className="w-3 h-3" /> Download PDF
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="gap-1.5 text-xs"
                              onClick={() => handleEmailBriefing(b)}
                            >
                              <Mail className="w-3 h-3" /> Send to Supervisor
                            </Button>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </div>

        {/* ── Phase 4: Drill, Streak, Badges ──────────────────────────────── */}
        <div className="grid md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Daily Research Drill</h3>
              </div>
              <p className="text-2xl font-bold text-foreground">{drillCount}</p>
              <p className="text-xs text-muted-foreground">drills completed</p>
              <Link to="/research-writing/daily-drill">
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">
                  Start Drill <ArrowRight className="w-3 h-3" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Flame className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Research Streak</h3>
              </div>
              <p className="text-2xl font-bold text-foreground">{streak.current}</p>
              <p className="text-xs text-muted-foreground">
                {streak.current === 1 ? "day" : "days"} current · {streak.longest} best
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Award className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Badges Earned</h3>
              </div>
              <p className="text-2xl font-bold text-foreground">{badgeCount}</p>
              <p className="text-xs text-muted-foreground">recognition badges</p>
              <Link to="/research-writing/portfolio">
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">
                  View Portfolio <ArrowRight className="w-3 h-3" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* ── Portfolio Link ───────────────────────────────────────────────── */}
        <Card className="border-dashed">
          <CardContent className="pt-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium text-sm">Research Competency Portfolio</p>
                <p className="text-xs text-muted-foreground">
                  View your competency profile, strengths, and progress
                </p>
              </div>
            </div>
            <Link to="/research-writing/portfolio">
              <Button size="sm" variant="outline" className="gap-1 text-xs">
                View Portfolio <ArrowRight className="w-3 h-3" />
              </Button>
            </Link>
          </CardContent>
        </Card>

      </div>
    </div>
  );
}
