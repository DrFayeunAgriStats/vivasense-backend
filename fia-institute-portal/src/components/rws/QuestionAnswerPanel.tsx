import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { X, CheckCircle, Loader2, MessageSquare } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import type { ReadingLogEntry } from "./AddPaperModal";

interface Props {
  entry: ReadingLogEntry;
  onAnswersSaved: (id: string, answers: string[]) => void;
  onClose: () => void;
}

function sentenceCount(text: string): number {
  return text.trim().split(/[.!?]+\s+/).filter((s) => s.trim().length > 5).length;
}

export function QuestionAnswerPanel({ entry, onAnswersSaved, onClose }: Props) {
  const questions = entry.ai_questions || [];
  const [answers,  setAnswers]  = useState<string[]>(
    // Pre-fill if re-opening a completed entry
    entry.student_answers || ["", "", ""]
  );
  const [saving,   setSaving]   = useState(false);
  const [saved,    setSaved]    = useState(entry.answer_completed);
  const [error,    setError]    = useState<string | null>(null);

  const allValid = answers.every((a) => sentenceCount(a) >= 2);

  const handleSave = async () => {
    if (!allValid) return;
    setSaving(true);
    setError(null);

    try {
      const { error: updateErr } = await supabase
        .from("reading_log" as never)
        .update({ student_answers: answers, answer_completed: true })
        .eq("id", entry.id);

      if (updateErr) throw updateErr;
      setSaved(true);
      onAnswersSaved(entry.id, answers);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not save answers. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">

        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-border">
          <div className="flex items-start gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
              <MessageSquare className="w-4 h-4 text-primary" />
            </div>
            <div className="min-w-0">
              <h2 className="font-serif text-base font-semibold text-foreground leading-tight">
                Answer Questions
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5 truncate">
                <span className="font-medium text-foreground">{entry.title}</span>
                {" — "}{entry.authors} ({entry.year})
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="shrink-0 ml-2">
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Relevance note */}
        <div className="px-5 pt-4 pb-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Your relevance note
          </p>
          <p className="text-sm text-foreground italic leading-relaxed border-l-2 border-primary/30 pl-3">
            "{entry.relevance_note}"
          </p>
        </div>

        {/* Questions and answers */}
        <div className="px-5 pt-4 pb-5 space-y-5">
          {questions.map((q, i) => {
            const asc     = sentenceCount(answers[i] || "");
            const isValid = asc >= 2;
            return (
              <div key={i} className="space-y-2">
                <div className="flex items-start gap-2.5">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-primary text-primary-foreground text-[11px] font-bold flex items-center justify-center mt-0.5">
                    {i + 1}
                  </span>
                  <p className="text-sm font-medium text-foreground leading-relaxed">{q}</p>
                </div>
                <div className="ml-7">
                  <Textarea
                    value={answers[i] || ""}
                    onChange={(e) => {
                      const next = [...answers];
                      next[i] = e.target.value;
                      setAnswers(next);
                    }}
                    placeholder="Write your answer here (at least 2 sentences)…"
                    rows={4}
                    className="resize-none text-sm"
                    disabled={saving || saved}
                  />
                  <p className={`text-[11px] mt-1 ${
                    isValid ? "text-primary" : "text-muted-foreground"
                  }`}>
                    {!(answers[i] || "").trim()
                      ? "At least 2 sentences required"
                      : isValid
                      ? "✓ Good"
                      : "Keep writing — need at least 2 sentences"}
                  </p>
                </div>
              </div>
            );
          })}

          {/* Saved confirmation */}
          {saved && (
            <div className="flex items-center gap-2 rounded-lg bg-primary/10 border border-primary/20 px-4 py-3 text-sm text-primary">
              <CheckCircle className="w-4 h-4 shrink-0" />
              <span>Answers saved. This paper will now inform your AI mentor sessions.</span>
            </div>
          )}

          {error && (
            <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 pt-1">
            {!saved && (
              <Button
                className="gap-2"
                onClick={handleSave}
                disabled={!allValid || saving}
              >
                {saving ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Saving…</>
                ) : (
                  "Save Answers"
                )}
              </Button>
            )}
            <Button
              variant={saved ? "default" : "ghost"}
              size="sm"
              className="text-xs text-muted-foreground"
              onClick={onClose}
            >
              {saved ? "Close" : "Answer later"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
