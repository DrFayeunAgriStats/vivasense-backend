import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, CheckCircle2, ClipboardList, Save } from "lucide-react";
import { MODULE_CONTENT, MODULE_NAMES } from "@/data/bgmModuleContent";
import type { BgmStudent } from "@/hooks/useBgmSession";

type Props = {
  student: BgmStudent;
  onBack: () => void;
  onComplete: () => void;
};

export function BgmWorkbook({ student, onBack, onComplete }: Props) {
  const mod = student.current_module;
  const content = MODULE_CONTENT[mod];
  const questions = content?.workbook || [];

  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);

  const answeredCount = Object.values(answers).filter((a) => a.trim().length > 20).length;
  const canSubmit = answeredCount >= 3;

  const handleSubmit = () => {
    setSubmitted(true);
    onComplete();
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="bg-primary text-primary-foreground py-3">
        <div className="container-wide flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}
            className="text-primary-foreground hover:bg-primary-foreground/10">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <ClipboardList className="w-5 h-5" />
          <div>
            <h1 className="font-serif text-lg font-bold">Workbook Tasks</h1>
            <p className="text-primary-foreground/70 text-[11px]">
              Module {mod}: {MODULE_NAMES[mod - 1]}
            </p>
          </div>
        </div>
      </div>

      <div className="container-wide py-6 max-w-3xl mx-auto space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <ClipboardList className="w-4 h-4 text-primary" />
              Instructions
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <p>Answer at least <strong>3 out of {questions.length}</strong> questions with substantive written responses (minimum 20 characters each). This is required before taking the module assessment.</p>
          </CardContent>
        </Card>

        <div className="text-sm text-muted-foreground text-right">
          Answered: <strong>{answeredCount}/{questions.length}</strong> (min 3 required)
        </div>

        {questions.map((q, i) => (
          <Card key={q.id} className={submitted ? "opacity-80" : ""}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Q{i + 1}. {q.question}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                value={answers[q.id] || ""}
                onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                placeholder="Write your answer here..."
                rows={4}
                disabled={submitted}
                className="text-sm"
              />
              {answers[q.id]?.trim().length > 20 && (
                <div className="flex items-center gap-1 mt-1 text-xs text-primary">
                  <CheckCircle2 className="w-3 h-3" /> Answered
                </div>
              )}
            </CardContent>
          </Card>
        ))}

        <div className="flex gap-3">
          <Button onClick={onBack} variant="outline" className="flex-1">
            Back to Dashboard
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!canSubmit || submitted}
            className="flex-1"
          >
            <Save className="w-4 h-4 mr-2" />
            {submitted ? "Submitted ✅" : `Submit Workbook (${answeredCount}/3 min)`}
          </Button>
        </div>
      </div>
    </div>
  );
}
