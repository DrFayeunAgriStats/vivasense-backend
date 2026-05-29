import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { X, Loader2, BookOpen } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

export interface ReadingLogEntry {
  id: string;
  title: string;
  authors: string;
  journal: string;
  year: number;
  relevance_note: string;
  ai_questions: string[] | null;
  student_answers: string[] | null;
  answer_completed: boolean;
  created_at: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  userId: string;
  profile: Record<string, unknown> | null;
  onQuestionsReady: (entry: ReadingLogEntry) => void;
}

const CURRENT_YEAR = new Date().getFullYear();

export function AddPaperModal({ isOpen, onClose, userId, profile, onQuestionsReady }: Props) {
  const [title,        setTitle]        = useState("");
  const [authors,      setAuthors]      = useState("");
  const [journal,      setJournal]      = useState("");
  const [year,         setYear]         = useState<string>("");
  const [relevance,    setRelevance]    = useState("");
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState<string | null>(null);

  if (!isOpen) return null;

  const yearNum    = parseInt(year, 10);
  const yearValid  = !isNaN(yearNum) && yearNum >= 1900 && yearNum <= CURRENT_YEAR;
  const canSubmit  = title.trim() && authors.trim() && journal.trim() && yearValid && relevance.trim() && !loading;

  const handleReset = () => {
    setTitle(""); setAuthors(""); setJournal(""); setYear(""); setRelevance(""); setError(null);
  };

  const handleClose = () => {
    handleReset();
    onClose();
  };

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);

    try {
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token || "";
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;

      // Call edge function to generate 3 paper-specific questions
      const resp = await fetch(`${supabaseUrl}/functions/v1/reading-log-questions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          title:          title.trim(),
          authors:        authors.trim(),
          journal:        journal.trim(),
          year:           yearNum,
          relevance_note: relevance.trim(),
          track:          profile?.academic_track || "",
          discipline:     profile?.discipline     || "",
          thesis_title:   profile?.thesis_title   || "not yet set",
          stage:          profile?.current_research_stage || "",
        }),
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to generate questions.");

      const questions: string[] = data.questions || [];

      // Insert the reading_log row with generated questions
      const { data: newRow, error: insertErr } = await supabase
        .from("reading_log" as never)
        .insert({
          user_id:        userId,
          title:          title.trim(),
          authors:        authors.trim(),
          journal:        journal.trim(),
          year:           yearNum,
          relevance_note: relevance.trim(),
          ai_questions:   questions,
          student_answers: null,
          answer_completed: false,
        })
        .select()
        .single();

      if (insertErr) throw insertErr;

      handleReset();
      onQuestionsReady(newRow as ReadingLogEntry);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center">
              <BookOpen className="w-4 h-4 text-primary" />
            </div>
            <div>
              <h2 className="font-serif text-base font-semibold text-foreground leading-tight">
                Log a Paper
              </h2>
              <p className="text-[11px] text-muted-foreground">
                Add a paper you have read to your reading record
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={handleClose} disabled={loading}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Form */}
        <div className="px-5 py-4 space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Paper Title <span className="text-destructive">*</span></Label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Full title of the paper"
              className="text-sm"
              disabled={loading}
            />
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Authors <span className="text-destructive">*</span></Label>
            <Input
              value={authors}
              onChange={(e) => setAuthors(e.target.value)}
              placeholder="e.g. Oluwaseun, A., Bello, M. K."
              className="text-sm"
              disabled={loading}
            />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="col-span-2 space-y-1.5">
              <Label className="text-xs font-medium">Journal / Source <span className="text-destructive">*</span></Label>
              <Input
                value={journal}
                onChange={(e) => setJournal(e.target.value)}
                placeholder="Journal or book title"
                className="text-sm"
                disabled={loading}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">Year <span className="text-destructive">*</span></Label>
              <Input
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="2024"
                min={1900}
                max={CURRENT_YEAR}
                className="text-sm"
                disabled={loading}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">
              Why does this paper matter to your research? <span className="text-destructive">*</span>
            </Label>
            <Textarea
              value={relevance}
              onChange={(e) => setRelevance(e.target.value)}
              placeholder="In one sentence, why does this paper matter to your research?"
              rows={3}
              className="resize-none text-sm"
              disabled={loading}
            />
            <p className="text-[10px] text-muted-foreground">
              Write this in your own words before seeing the AI questions — it forces you to articulate the connection.
            </p>
          </div>

          {error && (
            <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          <Button
            className="w-full gap-2"
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            {loading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Generating questions…</>
            ) : (
              "Log Paper & Get Questions"
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
