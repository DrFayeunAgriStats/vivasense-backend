import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { diagnosticQuestions } from "@/data/diagnosticQuestions";
import { useToast } from "@/hooks/use-toast";
import { ArrowRight, ArrowLeft, CheckCircle, Brain, BookOpen, Target, Award } from "lucide-react";

function classifyLevel(score: number, total: number): { level: "beginner" | "developing" | "advanced"; recommendation: string } {
  const pct = (score / total) * 100;
  if (pct >= 75) return { level: "advanced", recommendation: "Start with Discussion Builder or Defense Preparation" };
  if (pct >= 45) return { level: "developing", recommendation: "Start with Results Writing Foundations" };
  return { level: "beginner", recommendation: "Start with Statistical Interpretation Basics" };
}

const LEVEL_META = {
  beginner: { label: "Beginner Researcher", color: "text-orange-600", icon: BookOpen, bg: "bg-orange-50" },
  developing: { label: "Developing Researcher", color: "text-blue-600", icon: Target, bg: "bg-blue-50" },
  advanced: { label: "Advanced Researcher", color: "text-primary", icon: Award, bg: "bg-primary/5" },
};

export default function RWSDiagnostic() {
  const navigate = useNavigate();
  const { user, refreshProfile } = useAuth();
  const { toast } = useToast();
  const [currentQ, setCurrentQ] = useState(0);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [showResults, setShowResults] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!user) navigate("/research-writing/signin");
  }, [user]);

  const total = diagnosticQuestions.length;
  const progress = ((currentQ + 1) / total) * 100;
  const q = diagnosticQuestions[currentQ];

  const handleSubmit = async () => {
    setSaving(true);
    let score = 0;
    const responses: { question_index: number; answer: string; is_correct: boolean }[] = [];

    diagnosticQuestions.forEach((q, i) => {
      const userAnswer = answers[i] || "";
      const correct = userAnswer === q.correctAnswer;
      if (correct) score++;
      responses.push({ question_index: i, answer: userAnswer, is_correct: correct });
    });

    const { level, recommendation } = classifyLevel(score, total);

    try {
      // Create attempt
      const { data: attempt, error: attemptErr } = await supabase
        .from("diagnostic_attempts")
        .insert({
          user_id: user!.id,
          score,
          total_questions: total,
          level,
          recommended_starting_point: recommendation,
        })
        .select()
        .single();

      if (attemptErr) throw attemptErr;

      // Insert responses
      const { error: respErr } = await supabase.from("diagnostic_responses").insert(
        responses.map((r) => ({ ...r, attempt_id: attempt.id }))
      );
      if (respErr) throw respErr;

      // Update profile
      await supabase.from("profiles").update({
        diagnostic_level: level,
        diagnostic_score: score,
        onboarding_completed: true,
      }).eq("id", user!.id);

      await refreshProfile();
      setShowResults(true);
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  if (showResults) {
    let score = 0;
    diagnosticQuestions.forEach((q, i) => {
      if (answers[i] === q.correctAnswer) score++;
    });
    const { level, recommendation } = classifyLevel(score, total);
    const meta = LEVEL_META[level];

    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="w-full max-w-lg">
          <CardHeader className="text-center">
            <div className={`mx-auto w-16 h-16 rounded-full ${meta.bg} flex items-center justify-center mb-3`}>
              <meta.icon className={`w-8 h-8 ${meta.color}`} />
            </div>
            <CardTitle className="font-serif text-2xl">Diagnostic Complete</CardTitle>
            <CardDescription>Your research writing readiness assessment</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="text-center">
              <p className="text-4xl font-bold text-foreground">{score} / {total}</p>
              <p className="text-sm text-muted-foreground mt-1">
                {Math.round((score / total) * 100)}% correct
              </p>
            </div>

            <div className={`p-4 rounded-lg ${meta.bg} text-center`}>
              <p className={`font-semibold ${meta.color}`}>{meta.label}</p>
            </div>

            <div className="p-4 rounded-lg border border-border">
              <p className="text-sm font-medium text-foreground mb-1">Recommended Starting Point</p>
              <p className="text-sm text-muted-foreground">{recommendation}</p>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">Topic Breakdown</p>
              {["Understanding results vs discussion", "Statistical reasoning basics", "Use of numerical evidence", "Interpreting tables", "Scientific writing clarity"].map((topic) => {
                const topicQs = diagnosticQuestions.filter((q) => q.topic === topic);
                const topicCorrect = topicQs.filter((q, _) => {
                  const idx = diagnosticQuestions.indexOf(q);
                  return answers[idx] === q.correctAnswer;
                }).length;
                return (
                  <div key={topic} className="flex justify-between items-center text-sm">
                    <span className="text-muted-foreground">{topic}</span>
                    <span className="font-medium text-foreground">{topicCorrect}/{topicQs.length}</span>
                  </div>
                );
              })}
            </div>

            <Button className="w-full gap-2" onClick={() => navigate("/research-writing/dashboard")}>
              Go to Dashboard <ArrowRight className="w-4 h-4" />
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-6">
        <div className="text-center">
          <div className="mx-auto w-12 h-12 rounded-lg bg-primary flex items-center justify-center mb-3">
            <Brain className="w-6 h-6 text-primary-foreground" />
          </div>
          <h1 className="font-serif text-2xl font-bold text-foreground">Diagnostic Assessment</h1>
          <p className="text-muted-foreground text-sm mt-1">Assessing your research writing and interpretation skills</p>
        </div>

        <div className="space-y-1">
          <Progress value={progress} className="h-2" />
          <p className="text-xs text-muted-foreground text-right">Question {currentQ + 1} of {total}</p>
        </div>

        <Card>
          <CardHeader>
            <div className="text-xs font-medium text-primary uppercase tracking-wider mb-1">{q.topic}</div>
            <CardTitle className="text-base leading-relaxed">{q.question}</CardTitle>
          </CardHeader>
          <CardContent>
            <RadioGroup
              value={answers[currentQ] || ""}
              onValueChange={(v) => setAnswers((p) => ({ ...p, [currentQ]: v }))}
              className="space-y-3"
            >
              {q.options?.map((opt, i) => (
                <label
                  key={i}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    answers[currentQ] === opt ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
                  }`}
                >
                  <RadioGroupItem value={opt} className="mt-0.5" />
                  <span className="text-sm">{opt}</span>
                </label>
              ))}
            </RadioGroup>
          </CardContent>
        </Card>

        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={() => setCurrentQ(currentQ - 1)}
            disabled={currentQ === 0}
            className="gap-2"
          >
            <ArrowLeft className="w-4 h-4" /> Previous
          </Button>

          {currentQ < total - 1 ? (
            <Button
              onClick={() => setCurrentQ(currentQ + 1)}
              disabled={!answers[currentQ]}
              className="gap-2"
            >
              Next <ArrowRight className="w-4 h-4" />
            </Button>
          ) : (
            <Button
              onClick={handleSubmit}
              disabled={!answers[currentQ] || saving}
              className="gap-2"
            >
              <CheckCircle className="w-4 h-4" /> Submit Assessment
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
