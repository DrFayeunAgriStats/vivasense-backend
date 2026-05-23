import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, GraduationCap, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { MODULE_CONTENT, MODULE_NAMES, type AssessmentQuestion } from "@/data/bgmModuleContent";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import type { BgmStudent } from "@/hooks/useBgmSession";

type Props = {
  student: BgmStudent;
  onBack: () => void;
  onModuleComplete: () => void;
};

type GradedAnswer = {
  questionId: string;
  studentAnswer: string;
  score: number; // 0, 0.5, 1
  explanation: string;
};

function gradeAnswer(q: AssessmentQuestion, answer: string): number {
  const ans = answer.trim().toLowerCase();
  const correct = q.correctAnswer.toLowerCase();

  if (!ans) return 0;

  if (q.type === "mcq") {
    return ans === correct ? 1 : 0;
  }

  if (q.type === "numerical") {
    // Parse numbers for comparison
    const numAns = parseFloat(ans);
    const numCorrect = parseFloat(correct);
    if (!isNaN(numAns) && !isNaN(numCorrect)) {
      if (Math.abs(numAns - numCorrect) < 0.02) return 1;
      if (Math.abs(numAns - numCorrect) < 0.1) return 0.5;
    }
    // Check partial credit keywords
    if (q.partialCreditKeywords?.some((k) => ans.includes(k.toLowerCase()))) return 0.5;
    return 0;
  }

  // Short answer
  if (ans.includes(correct)) return 1;
  if (q.partialCreditKeywords?.some((k) => ans.includes(k.toLowerCase()))) return 0.5;
  return 0;
}

export function BgmAssessment({ student, onBack, onModuleComplete }: Props) {
  const mod = student.current_module;
  const content = MODULE_CONTENT[mod];
  const questions = content?.assessment || [];
  const { toast } = useToast();

  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [graded, setGraded] = useState<GradedAnswer[] | null>(null);
  const [totalScore, setTotalScore] = useState(0);
  const [passed, setPassed] = useState(false);
  const [remedialQuestions, setRemedialQuestions] = useState<AssessmentQuestion[]>([]);
  const [isRemedial, setIsRemedial] = useState(false);

  const allAnswered = questions.every((q) => answers[q.id]?.trim());

  const handleGrade = async () => {
    const results: GradedAnswer[] = questions.map((q) => ({
      questionId: q.id,
      studentAnswer: answers[q.id] || "",
      score: gradeAnswer(q, answers[q.id] || ""),
      explanation: q.explanation,
    }));

    const score = results.reduce((s, r) => s + r.score, 0);
    const didPass = score >= 7;

    setGraded(results);
    setTotalScore(score);
    setPassed(didPass);

    if (didPass) {
      // Update database
      const newCompleted = [...new Set([...student.completed_modules, mod])].sort((a, b) => a - b);
      const nextModule = mod < 11 ? mod + 1 : mod;
      const newScores = { ...student.best_scores, [mod]: Math.max(score, student.best_scores[mod] || 0) };
      const progressPercent = Math.round((newCompleted.length / 11) * 100);

      const isAllComplete = newCompleted.length === 11;
      const token = isAllComplete ? `BGM-ADV-${Math.floor(100000 + Math.random() * 900000)}` : null;

      await supabase
        .from("bgm_students")
        .update({
          current_module: nextModule,
          completed_modules: newCompleted,
          best_scores: newScores,
          progress_percent: progressPercent,
          token_status: isAllComplete ? "Generated" : "Locked",
          completion_token: token || student.completion_token,
          updated_at: new Date().toISOString(),
        })
        .eq("code", student.code);

      toast({
        title: `Module ${mod} Passed! 🎉`,
        description: isAllComplete
          ? `All modules complete! Token: ${token}`
          : `Score: ${score}/10. Advancing to Module ${nextModule}.`,
      });

      onModuleComplete();
    } else {
      // Identify weak areas and generate remedial questions
      const wrongQuestions = results
        .filter((r) => r.score < 1)
        .map((r) => questions.find((q) => q.id === r.questionId)!)
        .filter(Boolean)
        .slice(0, 3);

      setRemedialQuestions(wrongQuestions);
      setIsRemedial(true);

      // Update best score if higher
      const currentBest = student.best_scores[mod] || 0;
      if (score > currentBest) {
        await supabase
          .from("bgm_students")
          .update({
            best_scores: { ...student.best_scores, [mod]: score },
            updated_at: new Date().toISOString(),
          })
          .eq("code", student.code);
      }

      toast({
        title: "Remedial Required",
        description: `Score: ${score}/10 (need ≥7). Review the weak areas below.`,
        variant: "destructive",
      });
    }
  };

  const handleRetry = () => {
    setAnswers({});
    setGraded(null);
    setTotalScore(0);
    setPassed(false);
    setRemedialQuestions([]);
    setIsRemedial(false);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="bg-primary text-primary-foreground py-3">
        <div className="container-wide flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}
            className="text-primary-foreground hover:bg-primary-foreground/10">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <GraduationCap className="w-5 h-5" />
          <div>
            <h1 className="font-serif text-lg font-bold">Module Assessment</h1>
            <p className="text-primary-foreground/70 text-[11px]">
              Module {mod}: {MODULE_NAMES[mod - 1]}
            </p>
          </div>
        </div>
      </div>

      <div className="container-wide py-6 max-w-3xl mx-auto space-y-4">
        {/* Score display if graded */}
        {graded && (
          <Card className={passed ? "border-primary" : "border-destructive"}>
            <CardContent className="pt-4 text-center space-y-2">
              {passed ? (
                <CheckCircle2 className="w-10 h-10 text-primary mx-auto" />
              ) : (
                <XCircle className="w-10 h-10 text-destructive mx-auto" />
              )}
              <p className="text-2xl font-bold">{totalScore}/10</p>
              <p className="text-sm text-muted-foreground">
                {passed ? "PASS — Module Complete! 🎉" : "REMEDIAL REQUIRED — Score below 7/10"}
              </p>
              <Progress value={totalScore * 10} className="h-2 max-w-xs mx-auto" />
            </CardContent>
          </Card>
        )}

        {/* Remedial section */}
        {isRemedial && !passed && remedialQuestions.length > 0 && (
          <Card className="border-destructive/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2 text-destructive">
                <AlertTriangle className="w-4 h-4" />
                Weak Areas Identified — Review Required
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {remedialQuestions.map((q, i) => (
                <div key={q.id} className="bg-muted rounded-lg p-3 text-sm">
                  <p className="font-medium mb-1">Diagnostic {i + 1}: {q.question}</p>
                  <p className="text-muted-foreground">
                    <strong>Correct answer:</strong> {q.correctAnswer}
                  </p>
                  <p className="text-muted-foreground">{q.explanation}</p>
                </div>
              ))}
              <Button onClick={handleRetry} className="w-full mt-2">
                Re-take Assessment
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Questions */}
        {!graded && questions.map((q, i) => (
          <Card key={q.id}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Q{i + 1}. {q.question}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {q.type === "mcq" && q.options ? (
                <RadioGroup
                  value={answers[q.id] || ""}
                  onValueChange={(v) => setAnswers((prev) => ({ ...prev, [q.id]: v }))}
                >
                  {q.options.map((opt) => (
                    <div key={opt} className="flex items-center space-x-2">
                      <RadioGroupItem value={opt} id={`${q.id}-${opt}`} />
                      <Label htmlFor={`${q.id}-${opt}`} className="text-sm">{opt}</Label>
                    </div>
                  ))}
                </RadioGroup>
              ) : q.type === "numerical" ? (
                <Input
                  type="text"
                  value={answers[q.id] || ""}
                  onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                  placeholder="Enter numerical answer..."
                  className="max-w-xs"
                />
              ) : (
                <Textarea
                  value={answers[q.id] || ""}
                  onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                  placeholder="Write your answer..."
                  rows={2}
                  className="text-sm"
                />
              )}
            </CardContent>
          </Card>
        ))}

        {/* Graded results per question */}
        {graded && !passed && (
          <div className="space-y-3">
            <h3 className="font-medium text-sm">Detailed Results:</h3>
            {graded.map((r, i) => {
              const q = questions.find((q) => q.id === r.questionId)!;
              return (
                <Card key={r.questionId} className={r.score === 1 ? "border-primary/30" : r.score === 0.5 ? "border-accent/50" : "border-destructive/30"}>
                  <CardContent className="pt-3 text-sm space-y-1">
                    <div className="flex items-center gap-2">
                      {r.score === 1 ? (
                        <CheckCircle2 className="w-4 h-4 text-primary" />
                      ) : r.score === 0.5 ? (
                        <AlertTriangle className="w-4 h-4 text-accent" />
                      ) : (
                        <XCircle className="w-4 h-4 text-destructive" />
                      )}
                      <span className="font-medium">Q{i + 1}: {r.score === 1 ? "Correct" : r.score === 0.5 ? "Partial (0.5)" : "Incorrect"}</span>
                    </div>
                    <p className="text-muted-foreground">Your answer: {r.studentAnswer || "(blank)"}</p>
                    <p className="text-muted-foreground">Correct: {q.correctAnswer}</p>
                    <p className="text-xs text-muted-foreground">{r.explanation}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        {/* Submit / Back */}
        <div className="flex gap-3">
          <Button onClick={onBack} variant="outline" className="flex-1">
            Back to Dashboard
          </Button>
          {!graded && (
            <Button
              onClick={handleGrade}
              disabled={!allAnswered}
              className="flex-1"
            >
              <GraduationCap className="w-4 h-4 mr-2" />
              Submit Assessment ({Object.values(answers).filter(a => a?.trim()).length}/{questions.length})
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
