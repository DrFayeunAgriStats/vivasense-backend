import { useEffect, useState, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { rwsStreamChat, RWSChatMessage } from "@/lib/rwsStreamChat";
import {
  Shield, ArrowLeft, Send, Loader2, CheckCircle, AlertTriangle,
  Trophy, BarChart3, Brain, FlaskConical, Target,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

const EXAMINER_ROLES = [
  { key: "methodology", label: "Methodology Examiner", icon: FlaskConical, questions: 3 },
  { key: "subject", label: "Subject Specialist", icon: Brain, questions: 3 },
  { key: "critical", label: "Critical Reviewer", icon: Target, questions: 2 },
];

const SCORE_CATEGORIES = [
  "clarity_of_explanation",
  "understanding_of_methods",
  "data_interpretation",
  "scientific_reasoning",
  "critical_thinking",
];

const SCORE_LABELS: Record<string, string> = {
  clarity_of_explanation: "Clarity of Explanation",
  understanding_of_methods: "Understanding of Methods",
  data_interpretation: "Data Interpretation",
  scientific_reasoning: "Scientific Reasoning",
  critical_thinking: "Critical Thinking",
};

const TRACK_DEFENSE_LABEL: Record<string, string> = {
  undergraduate_project: "Project Defense Simulator",
  msc_thesis: "Thesis Defense Simulator",
  phd_research: "Thesis Defense Simulator",
  research_paper: "Research Defense Simulator",
};

const PASS_THRESHOLDS: Record<string, number> = {
  undergraduate_project: 70,
  msc_thesis: 75,
  phd_research: 80,
  research_paper: 75,
};

type SimPhase = "intro" | "active" | "scoring" | "results";

export default function RWSDefenseSimulator() {
  const navigate = useNavigate();
  const { user, profile, loading, session } = useAuth();
  const [phase, setPhase] = useState<SimPhase>("intro");
  const [attemptId, setAttemptId] = useState<string | null>(null);
  const [messages, setMessages] = useState<RWSChatMessage[]>([]);
  const [currentExaminer, setCurrentExaminer] = useState(0);
  const [questionCount, setQuestionCount] = useState(0);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [totalScore, setTotalScore] = useState(0);
  const [feedback, setFeedback] = useState("");
  const [pastAttempts, setPastAttempts] = useState<any[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
  }, [loading, user]);

  useEffect(() => {
    if (!user) return;
    supabase
      .from("defense_simulator_attempts")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(5)
      .then(({ data }) => {
        if (data) setPastAttempts(data);
      });
  }, [user]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (loading || !profile) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const track = profile.academic_track || "msc_thesis";
  const title = TRACK_DEFENSE_LABEL[track] || "Defense Simulator";
  const threshold = PASS_THRESHOLDS[track] || 75;
  const totalQuestions = EXAMINER_ROLES.reduce((s, e) => s + e.questions, 0);

  const startSimulation = async () => {
    const { data } = await supabase
      .from("defense_simulator_attempts")
      .insert({
        user_id: user!.id,
        track,
        simulation_type: "full",
        scores: {},
      })
      .select("id")
      .single();

    if (data) {
      setAttemptId(data.id);
      setPhase("active");
      setCurrentExaminer(0);
      setQuestionCount(0);
      setMessages([]);
      askQuestion([], 0);
    }
  };

  const askQuestion = (prevMessages: RWSChatMessage[], examinerIdx: number) => {
    const examiner = EXAMINER_ROLES[examinerIdx];
    const systemContext = `You are now acting as the ${examiner.label}. This is question ${questionCount + 1} of ${totalQuestions} in a ${title.toLowerCase()} session. The student's track is ${track}, discipline is ${profile.discipline || "agricultural science"}, and current stage is ${profile.current_research_stage || "unknown"}.

Ask ONE specific, probing question relevant to your role as ${examiner.label}. Be rigorous but constructive. Use agricultural science context where relevant. For undergraduate students, keep questioning simpler. For PhD students, challenge theoretical and methodological depth.`;

    const promptMsg: RWSChatMessage = {
      role: "user",
      content: prevMessages.length === 0
        ? `Begin the ${title.toLowerCase()} session. I am ready for the first question from the ${examiner.label}.`
        : prevMessages[prevMessages.length - 1].content,
    };

    const allMsgs = prevMessages.length === 0 ? [promptMsg] : prevMessages;
    setStreaming(true);
    let buffer = "";

    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    rwsStreamChat({
      mode: "defense",
      messages: allMsgs,
      context: {
        track,
        discipline: profile.discipline || undefined,
        stage: profile.current_research_stage || undefined,
      },
      authToken: session?.access_token,
      onDelta: (text) => {
        buffer += text;
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: buffer };
          return copy;
        });
      },
      onDone: () => {
        setStreaming(false);
        if (attemptId) {
          supabase.from("defense_simulator_messages").insert({
            attempt_id: attemptId,
            role: "assistant",
            content: buffer,
            examiner_role: examiner.key,
          });
        }
      },
      onError: (err) => {
        setStreaming(false);
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: `Error: ${err}` };
          return copy;
        });
      },
    });
  };

  const submitAnswer = () => {
    if (!input.trim() || streaming) return;

    const answer = input.trim();
    setInput("");
    const newMessages: RWSChatMessage[] = [...messages, { role: "user", content: answer }];
    setMessages(newMessages);

    if (attemptId) {
      supabase.from("defense_simulator_messages").insert({
        attempt_id: attemptId,
        role: "user",
        content: answer,
        examiner_role: EXAMINER_ROLES[currentExaminer].key,
      });
    }

    const newCount = questionCount + 1;
    setQuestionCount(newCount);

    if (newCount >= totalQuestions) {
      requestScoring(newMessages);
      return;
    }

    // Check if we need to move to next examiner
    const examinerQuestionsSoFar = EXAMINER_ROLES.slice(0, currentExaminer + 1).reduce((s, e) => s + e.questions, 0);
    let nextExaminer = currentExaminer;
    if (newCount >= examinerQuestionsSoFar && currentExaminer < EXAMINER_ROLES.length - 1) {
      nextExaminer = currentExaminer + 1;
      setCurrentExaminer(nextExaminer);
    }

    askQuestion(newMessages, nextExaminer);
  };

  const requestScoring = (allMessages: RWSChatMessage[]) => {
    setPhase("scoring");
    setStreaming(true);
    let buffer = "";

    const scoringPrompt: RWSChatMessage = {
      role: "user",
      content: `The defense simulation is now complete. Please evaluate the student's overall performance.

Score each category out of 20:
1. Clarity of Explanation
2. Understanding of Methods
3. Data Interpretation
4. Scientific Reasoning
5. Critical Thinking

Respond in EXACTLY this JSON format (no other text):
{
  "clarity_of_explanation": <number>,
  "understanding_of_methods": <number>,
  "data_interpretation": <number>,
  "scientific_reasoning": <number>,
  "critical_thinking": <number>,
  "total": <number>,
  "feedback": "<2-3 paragraphs of constructive feedback including strengths and areas for improvement>",
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>", "<weakness2>"]
}`,
    };

    rwsStreamChat({
      mode: "defense",
      messages: [...allMessages, scoringPrompt],
      context: { track, discipline: profile.discipline || undefined },
      authToken: session?.access_token,
      onDelta: (text) => { buffer += text; },
      onDone: async () => {
        setStreaming(false);
        try {
          // Extract JSON from the response
          const jsonMatch = buffer.match(/\{[\s\S]*\}/);
          if (!jsonMatch) throw new Error("No JSON found");
          const parsed = JSON.parse(jsonMatch[0]);

          const scoreMap: Record<string, number> = {};
          let total = 0;
          for (const cat of SCORE_CATEGORIES) {
            const val = Math.min(20, Math.max(0, Number(parsed[cat]) || 0));
            scoreMap[cat] = val;
            total += val;
          }
          setScores(scoreMap);
          setTotalScore(total);
          setFeedback(parsed.feedback || "");

          // Save to DB
          if (attemptId) {
            await supabase
              .from("defense_simulator_attempts")
              .update({
                scores: scoreMap,
                total_score: total,
                ai_feedback_summary: parsed.feedback || "",
                completed_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              })
              .eq("id", attemptId);

            // Update competency scores
            await supabase.from("competency_scores").upsert(
              Object.entries(scoreMap).map(([cat, score]) => ({
                user_id: user!.id,
                category: cat,
                score: score * 5,
                max_score: 100,
                updated_at: new Date().toISOString(),
              })),
              { onConflict: "user_id,category" }
            );
          }

          setPhase("results");
        } catch {
          setFeedback(buffer);
          setTotalScore(0);
          setPhase("results");
        }
      },
      onError: () => {
        setStreaming(false);
        setFeedback("Scoring failed. Please try again.");
        setPhase("results");
      },
    });
  };

  const progressPct = Math.round((questionCount / totalQuestions) * 100);
  const currentExaminerRole = EXAMINER_ROLES[currentExaminer];

  const getBand = (score: number) => {
    if (score >= 80) return { label: "Competent", color: "text-green-600", variant: "default" as const };
    if (score >= 60) return { label: "Needs Revision", color: "text-orange-600", variant: "secondary" as const };
    return { label: "Further Preparation Required", color: "text-destructive", variant: "destructive" as const };
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-4xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold flex items-center gap-2">
              <Shield className="w-6 h-6" /> {title}
            </h1>
          </div>
          <Link to="/research-writing/dashboard">
            <Button variant="secondary" size="sm" className="gap-1">
              <ArrowLeft className="w-4 h-4" /> Dashboard
            </Button>
          </Link>
        </div>
      </header>

      <div className="container max-w-4xl py-8 space-y-6">
        {/* Intro Phase */}
        {phase === "intro" && (
          <>
            <Card>
              <CardContent className="pt-6 space-y-4">
                <h2 className="font-serif text-xl font-bold">About This Simulation</h2>
                <p className="text-sm text-muted-foreground">
                  This simulator prepares you for your oral defense or viva voce examination. You will be questioned
                  by three AI examiners who will probe your understanding of your research.
                </p>
                <div className="grid sm:grid-cols-3 gap-3">
                  {EXAMINER_ROLES.map((ex) => (
                    <div key={ex.key} className="p-3 rounded-lg border border-border text-center">
                      <ex.icon className="w-6 h-6 mx-auto mb-2 text-primary" />
                      <p className="text-sm font-medium">{ex.label}</p>
                      <p className="text-xs text-muted-foreground">{ex.questions} questions</p>
                    </div>
                  ))}
                </div>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <p className="text-sm font-medium mb-2">Scoring (out of 100)</p>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {SCORE_CATEGORIES.map((c) => (
                      <p key={c} className="text-xs text-muted-foreground">• {SCORE_LABELS[c]} (20)</p>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Pass threshold for your track: <span className="font-medium text-foreground">{threshold}%</span>
                  </p>
                </div>
                <Button onClick={startSimulation} className="gap-2">
                  <Shield className="w-4 h-4" /> Begin Simulation
                </Button>
              </CardContent>
            </Card>

            {pastAttempts.length > 0 && (
              <Card>
                <CardHeader><CardTitle className="text-lg">Previous Attempts</CardTitle></CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {pastAttempts.map((a) => {
                      const band = getBand(a.total_score || 0);
                      return (
                        <div key={a.id} className="flex items-center justify-between p-3 rounded border border-border">
                          <div>
                            <p className="text-sm font-medium">{new Date(a.created_at).toLocaleDateString()}</p>
                            <p className="text-xs text-muted-foreground">{a.simulation_type}</p>
                          </div>
                          <div className="text-right">
                            <p className={`text-lg font-bold ${band.color}`}>{a.total_score || "—"}/100</p>
                            <Badge variant={band.variant}>{band.label}</Badge>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {/* Active Phase */}
        {phase === "active" && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <currentExaminerRole.icon className="w-5 h-5 text-primary" />
                  <CardTitle className="text-lg">{currentExaminerRole.label}</CardTitle>
                </div>
                <span className="text-xs text-muted-foreground">
                  Question {Math.min(questionCount + 1, totalQuestions)} of {totalQuestions}
                </span>
              </div>
              <Progress value={progressPct} className="h-2 mt-2" />
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="max-h-[400px] overflow-y-auto space-y-3 pr-2">
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[85%] p-3 rounded-lg text-sm ${
                      m.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}>
                      <ReactMarkdown>{m.content}</ReactMarkdown>
                    </div>
                  </div>
                ))}
                <div ref={scrollRef} />
              </div>

              <div className="flex gap-2">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your response..."
                  className="resize-none min-h-[80px]"
                  disabled={streaming}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      submitAnswer();
                    }
                  }}
                />
                <Button onClick={submitAnswer} disabled={streaming || !input.trim()} className="shrink-0">
                  {streaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Scoring Phase */}
        {phase === "scoring" && (
          <Card>
            <CardContent className="pt-6 text-center space-y-4">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-primary" />
              <h2 className="font-serif text-xl font-bold">Evaluating Your Performance</h2>
              <p className="text-sm text-muted-foreground">The examination panel is reviewing your responses...</p>
            </CardContent>
          </Card>
        )}

        {/* Results Phase */}
        {phase === "results" && (
          <>
            <Card>
              <CardContent className="pt-6 text-center space-y-4">
                <Trophy className="w-12 h-12 mx-auto text-primary" />
                <h2 className="font-serif text-2xl font-bold">Simulation Complete</h2>
                {totalScore > 0 && (
                  <>
                    <p className={`text-4xl font-bold ${getBand(totalScore).color}`}>
                      {totalScore}/100
                    </p>
                    <Badge variant={getBand(totalScore).variant} className="text-sm">
                      {getBand(totalScore).label}
                    </Badge>
                    {totalScore >= threshold ? (
                      <p className="text-sm text-green-600 flex items-center justify-center gap-1">
                        <CheckCircle className="w-4 h-4" /> Meets the threshold for your track ({threshold}%)
                      </p>
                    ) : (
                      <p className="text-sm text-orange-600 flex items-center justify-center gap-1">
                        <AlertTriangle className="w-4 h-4" /> Below threshold ({threshold}%). Further practice recommended.
                      </p>
                    )}
                  </>
                )}
              </CardContent>
            </Card>

            {totalScore > 0 && (
              <Card>
                <CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="w-5 h-5" /> Score Breakdown</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  {SCORE_CATEGORIES.map((cat) => (
                    <div key={cat}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-muted-foreground">{SCORE_LABELS[cat]}</span>
                        <span className="font-medium">{scores[cat] || 0}/20</span>
                      </div>
                      <Progress value={((scores[cat] || 0) / 20) * 100} className="h-2" />
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {feedback && (
              <Card>
                <CardHeader><CardTitle className="text-lg">Examiner Feedback</CardTitle></CardHeader>
                <CardContent>
                  <div className="prose prose-sm max-w-none text-muted-foreground">
                    <ReactMarkdown>{feedback}</ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            )}

            <div className="flex gap-3 justify-center">
              <Button onClick={() => { setPhase("intro"); setMessages([]); setScores({}); setTotalScore(0); setFeedback(""); }}>
                Try Again
              </Button>
              <Link to="/research-writing/dashboard">
                <Button variant="outline">Back to Dashboard</Button>
              </Link>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
