import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BookOpen, Plus, ArrowRight, CheckCircle } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { AddPaperModal, type ReadingLogEntry } from "./AddPaperModal";
import { QuestionAnswerPanel } from "./QuestionAnswerPanel";

const PREVIEW_COUNT = 5;

interface Props {
  userId: string;
  profile: Record<string, unknown> | null;
}

export function ReadingLogSection({ userId, profile }: Props) {
  const [entries,     setEntries]     = useState<ReadingLogEntry[]>([]);
  const [loading,     setLoading]     = useState(true);
  const [showAdd,     setShowAdd]     = useState(false);
  const [activePaper, setActivePaper] = useState<ReadingLogEntry | null>(null);
  const [expanded,    setExpanded]    = useState(false);

  // ── Fetch on mount ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!userId) return;
    const fetchEntries = async () => {
      setLoading(true);
      const { data } = await supabase
        .from("reading_log" as never)
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(50);
      if (data) setEntries(data as ReadingLogEntry[]);
      setLoading(false);
    };
    fetchEntries();
  }, [userId]);

  // ── Event handlers from child modals ─────────────────────────────────────

  const handleQuestionsReady = (newEntry: ReadingLogEntry) => {
    setEntries((prev) => [newEntry, ...prev]);
    setShowAdd(false);
    setActivePaper(newEntry);
  };

  const handleAnswersSaved = (id: string, answers: string[]) => {
    setEntries((prev) =>
      prev.map((e) =>
        e.id === id
          ? { ...e, answer_completed: true, student_answers: answers }
          : e
      )
    );
    setActivePaper(null);
  };

  // ── Derived display ───────────────────────────────────────────────────────
  const displayedEntries = expanded ? entries : entries.slice(0, PREVIEW_COUNT);
  const answeredCount    = entries.filter((e) => e.answer_completed).length;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Modals — rendered at root level so they're always on top */}
      <AddPaperModal
        isOpen={showAdd}
        onClose={() => setShowAdd(false)}
        userId={userId}
        profile={profile}
        onQuestionsReady={handleQuestionsReady}
      />
      {activePaper && (
        <QuestionAnswerPanel
          entry={activePaper}
          onAnswersSaved={handleAnswersSaved}
          onClose={() => setActivePaper(null)}
        />
      )}

      {/* Section */}
      <div>
        {/* Section header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="font-serif text-xl font-bold text-foreground">Reading Log</h2>
            {!loading && entries.length > 0 && (
              <p className="text-xs text-muted-foreground mt-0.5">
                {entries.length} paper{entries.length !== 1 ? "s" : ""} logged
                {answeredCount > 0 && ` · ${answeredCount} answered`}
              </p>
            )}
          </div>
          <Button
            size="sm"
            variant="outline"
            className="gap-1.5 text-xs"
            onClick={() => setShowAdd(true)}
          >
            <Plus className="w-3.5 h-3.5" /> Add Paper
          </Button>
        </div>

        {/* Loading */}
        {loading && (
          <div className="rounded-xl border border-border p-6 text-center">
            <p className="text-sm text-muted-foreground animate-pulse">Loading reading log…</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && entries.length === 0 && (
          <div className="rounded-xl border border-dashed border-border p-6 text-center space-y-3">
            <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center mx-auto">
              <BookOpen className="w-5 h-5 text-muted-foreground" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground font-medium">No papers logged yet.</p>
              <p className="text-xs text-muted-foreground mt-1">
                Add your first paper to begin building your reading record.
              </p>
              <p className="text-xs italic text-muted-foreground/70 mt-2">
                Papers you log here will inform your AI mentor sessions and appear in your Supervisor Briefing.
              </p>
            </div>
            <Button
              size="sm"
              variant="outline"
              className="gap-1.5 text-xs mt-1"
              onClick={() => setShowAdd(true)}
            >
              <Plus className="w-3.5 h-3.5" /> Add Your First Paper
            </Button>
          </div>
        )}

        {/* Paper cards */}
        {!loading && entries.length > 0 && (
          <div className="space-y-3">
            {displayedEntries.map((entry) => (
              <PaperCard
                key={entry.id}
                entry={entry}
                onAnswerQuestions={() => setActivePaper(entry)}
              />
            ))}

            {/* Expand / collapse toggle */}
            {entries.length > PREVIEW_COUNT && (
              <button
                className="w-full text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center justify-center gap-1.5 py-2"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? (
                  "Show fewer papers"
                ) : (
                  <>
                    View all {entries.length} papers <ArrowRight className="w-3 h-3" />
                  </>
                )}
              </button>
            )}
          </div>
        )}
      </div>
    </>
  );
}

// ── PaperCard ─────────────────────────────────────────────────────────────────

function PaperCard({
  entry,
  onAnswerQuestions,
}: {
  entry: ReadingLogEntry;
  onAnswerQuestions: () => void;
}) {
  const date = new Date(entry.created_at).toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric",
  });

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-2.5">
      {/* Title row */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p
            className="text-sm font-medium text-foreground leading-snug line-clamp-2"
            title={entry.title}
          >
            {entry.title}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {entry.authors} · {entry.year}
          </p>
        </div>
        {entry.answer_completed ? (
          <Badge className="shrink-0 text-[10px] bg-primary/10 text-primary border-primary/20 font-medium">
            <CheckCircle className="w-3 h-3 mr-1" /> Answered
          </Badge>
        ) : (
          <Badge
            variant="outline"
            className="shrink-0 text-[10px] border-amber-300 text-amber-700 bg-amber-50 font-medium"
          >
            Questions Pending
          </Badge>
        )}
      </div>

      {/* Journal */}
      <p className="text-[11px] text-muted-foreground">{entry.journal}</p>

      {/* Relevance note */}
      <p className="text-xs text-foreground/80 italic leading-relaxed border-l-2 border-primary/20 pl-2.5">
        "{entry.relevance_note}"
      </p>

      {/* Footer */}
      <div className="flex items-center justify-between pt-0.5">
        <span className="text-[10px] text-muted-foreground">Logged {date}</span>
        {!entry.answer_completed && (
          <Button
            size="sm"
            variant="outline"
            className="gap-1 text-xs h-7 border-amber-300 text-amber-700 hover:bg-amber-50"
            onClick={onAnswerQuestions}
          >
            Answer Questions <ArrowRight className="w-3 h-3" />
          </Button>
        )}
      </div>
    </div>
  );
}
