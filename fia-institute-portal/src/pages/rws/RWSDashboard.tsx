import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { supabase } from "@/integrations/supabase/client";
import {
  GraduationCap, BookOpen, ArrowRight, Brain, FlaskConical, Target,
  MessageSquare, Shield, Lock, CheckCircle, FileText, Microscope, Award,
  RefreshCw, Calendar, User, BarChart3, Zap, Flame, Bell,
} from "lucide-react";
import { NotificationBell } from "@/components/rws/NotificationBell";

const TRACK_LABELS: Record<string, string> = {
  undergraduate_project: "Undergraduate Final-Year Project",
  msc_thesis: "MSc Thesis Development",
  phd_research: "PhD Research Writing and Defense",
  research_paper: "Research Paper Writing",
};

const LEVEL_META: Record<string, { label: string; color: string }> = {
  beginner: { label: "Beginner Researcher", color: "text-orange-600" },
  developing: { label: "Developing Researcher", color: "text-blue-600" },
  advanced: { label: "Advanced Researcher", color: "text-primary" },
};

const STAGE_LABELS: Record<string, string> = {
  topic_proposal: "Topic / Proposal",
  literature_review: "Literature Review",
  methodology: "Methodology",
  data_analysis: "Data Analysis",
  results_writing: "Results Writing",
  discussion: "Discussion",
  defense_preparation: "Defense Preparation",
};

const ROADMAP = [
  { key: "topic_proposal", label: "Proposal / Topic", icon: FileText },
  { key: "literature_review", label: "Literature Review", icon: BookOpen },
  { key: "methodology", label: "Methodology", icon: FlaskConical },
  { key: "data_analysis", label: "Data Analysis", icon: Microscope },
  { key: "results_writing", label: "Results Interpretation", icon: Target },
  { key: "discussion", label: "Discussion", icon: MessageSquare },
  { key: "defense_preparation", label: "Defense", icon: Shield },
  { key: "certification", label: "Certification", icon: Award },
];

const MODULES = [
  { key: "thesis_mentor", label: "FIA Research Writing Mentor", description: "AI-assisted guidance for proposal, thesis, and research paper development", href: "/thesis-mentor", available: true, icon: Brain },
  { key: "results_lab", label: "Guided Results Interpretation Lab", description: "Interpret ANOVA, correlations, PCA, and more with the 4-Level Framework", href: "/research-writing/results-lab", available: true, icon: FlaskConical },
  { key: "defense_sim", label: "Defense Simulator", description: "Practice defending your research findings with AI viva questions", href: "/research-writing/defense", available: true, icon: Shield },
  { key: "supervisor", label: "Supervisor Guidance", description: "Structured supervisor-student interaction tools", href: "/research-writing/bookings", available: true, icon: GraduationCap },
];

const NEXT_ACTIONS = [
  { condition: "no_diagnostic", label: "Complete Diagnostic Assessment", description: "Identify your knowledge gaps and starting level", href: "/research-writing/diagnostic" },
  { condition: "topic_proposal", label: "Open Research Writing Mentor", description: "Start developing your proposal with AI guidance", href: "/thesis-mentor" },
  { condition: "literature_review", label: "Review Literature with AI Guide", description: "Use Guide mode to synthesise your literature review", href: "/thesis-mentor" },
  { condition: "methodology", label: "Develop Your Methodology", description: "Use AI guidance to structure your methods chapter", href: "/thesis-mentor" },
  { condition: "data_analysis", label: "Interpret Your Results", description: "Upload statistical output to the Results Interpretation Lab", href: "/research-writing/results-lab" },
  { condition: "results_writing", label: "Write Your Results Section", description: "Use the AI mentor to structure your results chapter", href: "/thesis-mentor" },
  { condition: "discussion", label: "Build Your Discussion", description: "Connect your results to the literature", href: "/thesis-mentor" },
  { condition: "defense_preparation", label: "Prepare for Your Defense", description: "Use the Defense Simulator for viva practice", href: "/research-writing/defense" },
];

export default function RWSDashboard() {
  const navigate = useNavigate();
  const { user, profile, loading } = useAuth();
  const [milestones, setMilestones] = useState<any[]>([]);
  const [conversationCount, setConversationCount] = useState(0);
  const [defenseAttempt, setDefenseAttempt] = useState<any>(null);
  const [certStatus, setCertStatus] = useState<string | null>(null);
  const [bookings, setBookings] = useState<any[]>([]);
  const [streak, setStreak] = useState({ current: 0, longest: 0 });
  const [drillCount, setDrillCount] = useState(0);
  const [badgeCount, setBadgeCount] = useState(0);

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
    if (!loading && user && profile && !profile.onboarding_completed) navigate("/research-writing/onboarding");
  }, [loading, user, profile]);

  useEffect(() => {
    if (!user) return;
    const fetchData = async () => {
      const [milestonesRes, convsRes, defenseRes, certRes, bookingsRes, streakRes, drillRes, badgeRes] = await Promise.all([
        supabase.from("milestones").select("*").eq("user_id", user.id).order("created_at"),
        supabase.from("ai_conversations").select("id", { count: "exact" }).eq("user_id", user.id),
        supabase.from("defense_simulator_attempts").select("*").eq("user_id", user.id).not("completed_at", "is", null).order("total_score", { ascending: false }).limit(1),
        supabase.from("certificate_eligibility").select("certificate_status").eq("user_id", user.id).single(),
        supabase.from("booking_requests").select("*").eq("student_id", user.id).order("created_at", { ascending: false }).limit(3),
        supabase.from("user_streaks").select("*").eq("user_id", user.id).single(),
        supabase.from("daily_drill_attempts").select("id", { count: "exact" }).eq("user_id", user.id),
        supabase.from("user_badges").select("id", { count: "exact" }).eq("user_id", user.id),
      ]);
      if (milestonesRes.data) setMilestones(milestonesRes.data);
      if (convsRes.count !== null) setConversationCount(convsRes.count);
      if (defenseRes.data?.[0]) setDefenseAttempt(defenseRes.data[0]);
      if (certRes.data) setCertStatus(certRes.data.certificate_status);
      if (bookingsRes.data) setBookings(bookingsRes.data);
      if (streakRes.data) setStreak({ current: streakRes.data.current_streak, longest: streakRes.data.longest_streak });
      if (drillRes.count !== null) setDrillCount(drillRes.count);
      if (badgeRes.count !== null) setBadgeCount(badgeRes.count);
    };
    fetchData();
  }, [user]);

  if (loading || !profile) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  const levelMeta = LEVEL_META[profile.diagnostic_level || "beginner"] || LEVEL_META.beginner;
  const currentStageIdx = ROADMAP.findIndex((r) => r.key === profile.current_research_stage);
  const progressPct = currentStageIdx >= 0 ? Math.round(((currentStageIdx + 1) / ROADMAP.length) * 100) : 10;

  let nextAction = NEXT_ACTIONS[0];
  if (profile.diagnostic_score !== null && profile.diagnostic_score !== undefined) {
    const stageAction = NEXT_ACTIONS.find((a) => a.condition === profile.current_research_stage);
    nextAction = stageAction || { condition: "default", label: "Open Research Writing Mentor", description: "Continue your research journey", href: "/thesis-mentor" };
  }

  const completedMilestones = milestones.filter((m) => m.is_completed).length;

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-5xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold">Welcome, {profile.full_name || "Researcher"}</h1>
          </div>
          <div className="flex items-center gap-2">
            {user && <NotificationBell userId={user.id} />}
            <Link to="/research-writing"><Button variant="secondary" size="sm">Back to Home</Button></Link>
          </div>
        </div>
      </header>

      <div className="container max-w-5xl py-8 space-y-8">
        {/* Status Cards */}
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Track</p>
            <p className="font-medium text-foreground mt-1 text-sm">{TRACK_LABELS[profile.academic_track || ""] || "Not set"}</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Diagnostic Level</p>
            <p className={`font-medium mt-1 text-sm ${levelMeta.color}`}>{levelMeta.label}</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Research Stage</p>
            <p className="font-medium text-foreground mt-1 text-sm">{STAGE_LABELS[profile.current_research_stage || ""] || "Not set"}</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">AI Sessions</p>
            <p className="font-medium text-foreground mt-1 text-sm">{conversationCount} conversation{conversationCount !== 1 ? "s" : ""}</p>
          </CardContent></Card>
        </div>

        {/* Next Action */}
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

        {/* Phase 3 Cards: Defense, Bookings, Certificate */}
        <div className="grid md:grid-cols-3 gap-4">
          {/* Defense Simulator Card */}
          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Defense Simulator</h3>
              </div>
              {defenseAttempt ? (
                <>
                  <p className="text-2xl font-bold text-primary">{defenseAttempt.total_score}/100</p>
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

          {/* Bookings Card */}
          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Calendar className="w-5 h-5 text-primary" />
                <h3 className="font-medium text-sm">Supervisor Sessions</h3>
              </div>
              {bookings.length > 0 ? (
                <div className="space-y-1">
                  {bookings.slice(0, 2).map((b) => (
                    <p key={b.id} className="text-xs text-muted-foreground">
                      {b.milestone_type.replace(/_/g, " ")} — <Badge variant="secondary" className="text-[10px]">{b.booking_status}</Badge>
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

          {/* Certificate Card */}
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

        {/* Progress & Roadmap */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Research Journey</CardTitle>
              {completedMilestones > 0 && (
                <span className="text-xs text-muted-foreground">{completedMilestones} milestone{completedMilestones !== 1 ? "s" : ""} completed</span>
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
                const isPast = i < currentStageIdx;
                const isLocked = stage.key === "certification";
                return (
                  <div key={stage.key} className={`p-3 rounded-lg border text-center transition-colors ${isActive ? "border-primary bg-primary/5" : isPast ? "border-primary/30 bg-primary/5" : "border-border"}`}>
                    <div className="mx-auto w-8 h-8 rounded-full flex items-center justify-center mb-2" style={{ backgroundColor: isActive || isPast ? "hsl(var(--primary))" : "hsl(var(--muted))" }}>
                      {isLocked ? <Lock className="w-4 h-4 text-muted-foreground" /> : isPast ? <CheckCircle className="w-4 h-4 text-primary-foreground" /> : <stage.icon className={`w-4 h-4 ${isActive ? "text-primary-foreground" : "text-muted-foreground"}`} />}
                    </div>
                    <p className={`text-xs font-medium ${isActive ? "text-primary" : "text-muted-foreground"}`}>{stage.label}</p>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-center pt-2">
              <Link to="/research-writing/onboarding">
                <Button variant="ghost" size="sm" className="gap-1.5 text-xs text-muted-foreground"><RefreshCw className="w-3 h-3" /> Update Research Stage</Button>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Modules */}
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
                          <Button size="sm" variant="outline" className="mt-3 gap-1 text-xs">Open <ArrowRight className="w-3 h-3" /></Button>
                        </Link>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Phase 4: Drill, Streak, Badges */}
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
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">Start Drill <ArrowRight className="w-3 h-3" /></Button>
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
              <p className="text-xs text-muted-foreground">{streak.current === 1 ? "day" : "days"} current · {streak.longest} best</p>
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
                <Button size="sm" variant="outline" className="w-full gap-1 text-xs">View Portfolio <ArrowRight className="w-3 h-3" /></Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* Portfolio Link */}
        <Card className="border-dashed">
          <CardContent className="pt-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <User className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium text-sm">Research Competency Portfolio</p>
                <p className="text-xs text-muted-foreground">View your competency profile, strengths, and progress</p>
              </div>
            </div>
            <Link to="/research-writing/portfolio">
              <Button size="sm" variant="outline" className="gap-1 text-xs">View Portfolio <ArrowRight className="w-3 h-3" /></Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
