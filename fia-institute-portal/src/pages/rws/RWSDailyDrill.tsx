import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Zap, Brain, PenLine, BarChart3, FlaskConical, Target,
  ArrowRight, CheckCircle, Flame, RefreshCw,
} from "lucide-react";
import { rwsStreamChat } from "@/lib/rwsStreamChat";

const DRILL_TYPES = [
  { key: "table_interpretation", label: "Table Interpretation", icon: BarChart3, description: "Interpret a short statistical table" },
  { key: "writing_precision", label: "Writing Precision", icon: PenLine, description: "Improve a poorly written research sentence" },
  { key: "results_vs_discussion", label: "Results vs Discussion", icon: Target, description: "Classify statements correctly" },
  { key: "research_design", label: "Research Design", icon: FlaskConical, description: "Identify flaws in experimental design" },
  { key: "data_reasoning", label: "Data Reasoning", icon: Brain, description: "Explain what a result implies" },
];

export default function RWSDailyDrill() {
  const navigate = useNavigate();
  const { user, profile, loading } = useAuth();
  const [drillType, setDrillType] = useState<string | null>(null);
  const [drillPrompt, setDrillPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [feedback, setFeedback] = useState("");
  const [generating, setGenerating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [todayCount, setTodayCount] = useState(0);
  const [streak, setStreak] = useState({ current: 0, longest: 0 });

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
  }, [loading, user]);

  useEffect(() => {
    if (!user) return;
    const fetchStats = async () => {
      const today = new Date().toISOString().split("T")[0];
      const [drillsRes, streakRes] = await Promise.all([
        supabase.from("daily_drill_attempts").select("id", { count: "exact" }).eq("user_id", user.id).gte("created_at", today),
        supabase.from("user_streaks").select("*").eq("user_id", user.id).single(),
      ]);
      if (drillsRes.count !== null) setTodayCount(drillsRes.count);
      if (streakRes.data) setStreak({ current: streakRes.data.current_streak, longest: streakRes.data.longest_streak });
    };
    fetchStats();
  }, [user]);

  const startDrill = async (type: string) => {
    setDrillType(type);
    setDrillPrompt("");
    setResponse("");
    setFeedback("");
    setCompleted(false);
    setGenerating(true);
    try {
      let prompt = "";
      const discipline = profile?.discipline || "crop science";
      const systemMsg = `Generate a short ${type.replace(/_/g, " ")} exercise for a research student in ${discipline}. The exercise should take 3-5 minutes. Present the exercise clearly with any necessary data or text. End with a clear question the student must answer. Do NOT provide the answer.`;
      await rwsStreamChat({
        mode: "explain",
        messages: [{ role: "user", content: systemMsg }],
        context: { drill_type: type, discipline },
        onDelta: (chunk) => { prompt += chunk; setDrillPrompt(prompt); },
        onDone: () => {},
        onError: () => { setDrillPrompt("Failed to generate drill. Please try again."); },
      });
    } catch { setDrillPrompt("Failed to generate drill. Please try again."); }
    setGenerating(false);
  };

  const submitResponse = async () => {
    if (!response.trim() || !drillType) return;
    setSubmitting(true);
    let fb = "";
    try {
      const evalPrompt = `The student was given this exercise:\n\n${drillPrompt}\n\nThe student responded:\n\n${response}\n\nEvaluate the response. Give a score out of 10. Highlight strengths and areas for improvement. Be constructive but rigorous. Use agricultural science context.`;
      await rwsStreamChat({
        mode: "review",
        messages: [{ role: "user", content: evalPrompt }],
        context: { drill_type: drillType, discipline: profile?.discipline || "crop science" },
        onDelta: (chunk) => { fb += chunk; setFeedback(fb); },
        onDone: () => {},
        onError: () => { setFeedback("Failed to evaluate. Please try again."); },
      });
      // Save attempt
      await supabase.from("daily_drill_attempts").insert({
        user_id: user!.id,
        drill_type: drillType,
        drill_content: { prompt: drillPrompt },
        student_response: response,
        ai_feedback: fb,
        completed_at: new Date().toISOString(),
      });
      // Update streak
      await updateStreak();
      setCompleted(true);
      setTodayCount((c) => c + 1);
    } catch { setFeedback("Failed to evaluate. Please try again."); }
    setSubmitting(false);
  };

  const updateStreak = async () => {
    if (!user) return;
    const today = new Date().toISOString().split("T")[0];
    const { data: existing } = await supabase.from("user_streaks").select("*").eq("user_id", user.id).single();
    if (existing) {
      const lastDate = existing.last_activity_date;
      const yesterday = new Date(Date.now() - 86400000).toISOString().split("T")[0];
      let newStreak = existing.current_streak;
      if (lastDate === today) return;
      else if (lastDate === yesterday) newStreak += 1;
      else newStreak = 1;
      const longest = Math.max(newStreak, existing.longest_streak);
      await supabase.from("user_streaks").update({ current_streak: newStreak, longest_streak: longest, last_activity_date: today, updated_at: new Date().toISOString() }).eq("user_id", user.id);
      setStreak({ current: newStreak, longest });
    } else {
      await supabase.from("user_streaks").insert({ user_id: user.id, current_streak: 1, longest_streak: 1, last_activity_date: today });
      setStreak({ current: 1, longest: 1 });
    }
  };

  if (loading || !profile) return <div className="min-h-screen bg-background flex items-center justify-center"><div className="animate-pulse text-muted-foreground">Loading...</div></div>;

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-4xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold">Daily Research Drill</h1>
          </div>
          <div className="flex items-center gap-3">
            {streak.current > 0 && (
              <Badge variant="secondary" className="gap-1"><Flame className="w-3 h-3" />{streak.current}-day streak</Badge>
            )}
            <Button variant="secondary" size="sm" onClick={() => navigate("/research-writing/dashboard")}>Dashboard</Button>
          </div>
        </div>
      </header>

      <div className="container max-w-4xl py-8 space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-3 gap-4">
          <Card><CardContent className="pt-6 text-center">
            <p className="text-2xl font-bold text-primary">{todayCount}</p>
            <p className="text-xs text-muted-foreground">Drills Today</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6 text-center">
            <Flame className="w-5 h-5 text-primary mx-auto mb-1" />
            <p className="text-2xl font-bold text-foreground">{streak.current}</p>
            <p className="text-xs text-muted-foreground">Current Streak</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6 text-center">
            <p className="text-2xl font-bold text-foreground">{streak.longest}</p>
            <p className="text-xs text-muted-foreground">Longest Streak</p>
          </CardContent></Card>
        </div>

        {!drillType ? (
          <>
            <h2 className="font-serif text-lg font-semibold text-foreground">Choose Your Drill</h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {DRILL_TYPES.map((d) => (
                <Card key={d.key} className="cursor-pointer hover:border-primary/50 transition-colors" onClick={() => startDrill(d.key)}>
                  <CardContent className="pt-6 flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <d.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-sm text-foreground">{d.label}</p>
                      <p className="text-xs text-muted-foreground mt-1">{d.description}</p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
        ) : (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="gap-1">
                <Zap className="w-3 h-3" />{DRILL_TYPES.find((d) => d.key === drillType)?.label}
              </Badge>
              <Button variant="ghost" size="sm" onClick={() => setDrillType(null)} className="text-xs gap-1"><RefreshCw className="w-3 h-3" />New Drill</Button>
            </div>

            <Card>
              <CardHeader><CardTitle className="text-base">Exercise</CardTitle></CardHeader>
              <CardContent>
                {generating ? (
                  <div className="animate-pulse text-muted-foreground text-sm">{drillPrompt || "Generating exercise..."}</div>
                ) : (
                  <div className="prose prose-sm max-w-none text-foreground whitespace-pre-wrap">{drillPrompt}</div>
                )}
              </CardContent>
            </Card>

            {!generating && drillPrompt && !completed && (
              <Card>
                <CardHeader><CardTitle className="text-base">Your Response</CardTitle></CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    value={response}
                    onChange={(e) => setResponse(e.target.value)}
                    placeholder="Type your answer here..."
                    rows={6}
                    className="text-sm"
                  />
                  <Button onClick={submitResponse} disabled={submitting || !response.trim()} className="gap-2">
                    {submitting ? "Evaluating..." : "Submit Response"}<ArrowRight className="w-4 h-4" />
                  </Button>
                </CardContent>
              </Card>
            )}

            {feedback && (
              <Card className="border-l-4 border-l-accent">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    {completed && <CheckCircle className="w-4 h-4 text-primary" />}AI Feedback
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-sm max-w-none text-foreground whitespace-pre-wrap">{feedback}</div>
                </CardContent>
              </Card>
            )}

            {completed && (
              <div className="flex gap-3 justify-center">
                <Button variant="outline" onClick={() => startDrill(drillType)} className="gap-1"><RefreshCw className="w-3 h-3" />Another Drill</Button>
                <Button onClick={() => setDrillType(null)} className="gap-1">Choose Different Type</Button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
