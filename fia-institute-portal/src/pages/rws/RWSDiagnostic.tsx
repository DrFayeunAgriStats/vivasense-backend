import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Loader2, CheckCircle, AlertCircle } from "lucide-react";

// ── Constants ─────────────────────────────────────────────────────────────────

const TRACK_LABELS: Record<string, string> = {
  undergraduate_project: "Undergraduate Final-Year Project",
  msc_thesis:            "MSc Thesis Development",
  phd_research:          "PhD Research Writing and Defense",
  research_paper:        "Research Paper Writing",
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

// Maps the AI-returned uppercase level to the DB enum value
const LEVEL_DB_MAP: Record<string, "beginner" | "developing" | "advanced"> = {
  FOUNDATION: "beginner",
  DEVELOPING: "developing",
  ADVANCED:   "advanced",
};

const LEVEL_DISPLAY: Record<string, { label: string; bg: string; text: string }> = {
  FOUNDATION: { label: "Foundation",  bg: "bg-primary/10", text: "text-primary" },
  DEVELOPING: { label: "Developing",  bg: "bg-primary/10", text: "text-primary" },
  ADVANCED:   { label: "Advanced",    bg: "bg-primary/10", text: "text-primary" },
};

const WORD_MIN = 150;
const WORD_MAX = 300;

// ── Helpers ───────────────────────────────────────────────────────────────────

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function sentenceCount(text: string): number {
  return text.trim().split(/[.!?]+\s+/).filter((s) => s.trim().length > 5).length;
}

// ── Types ─────────────────────────────────────────────────────────────────────

type DiagStage = "submission" | "generating" | "answering" | "assessing" | "results";

interface AssessmentResult {
  level: string;
  level_rationale: string;
  three_strengths: string[];
  three_gaps: string[];
  first_priority: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function RWSDiagnostic() {
  const navigate    = useNavigate();
  const { user, profile, refreshProfile } = useAuth() as {
    user: { id: string } | null;
    profile: Record<string, unknown> | null;
    loading: boolean;
    refreshProfile: () => Promise<void>;
  };

  const [stage,         setStage]         = useState<DiagStage>("submission");
  const [writingSample, setWritingSample] = useState("");
  const [questions,     setQuestions]     = useState<string[]>([]);
  const [answers,       setAnswers]       = useState<string[]>(["", "", "", "", ""]);
  const [result,        setResult]        = useState<AssessmentResult | null>(null);
  const [error,         setError]         = useState<string | null>(null);

  useEffect(() => {
    if (!user) navigate("/research-writing/signin");
  }, [user]);

  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
  const authToken   = (() => {
    try {
      // Access session from supabase client directly for the token
      return null; // resolved below via getSession
    } catch { return null; }
  })();

  // Fetch current session token on each API call
  const getToken = useCallback(async (): Promise<string> => {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token || "";
  }, []);

  const wc = wordCount(writingSample);
  const wordCountOk = wc >= WORD_MIN && wc <= WORD_MAX;

  // ── Stage 1 → 2: Submit writing sample, generate questions ────────────────
  const handleSubmitSample = async () => {
    if (!wordCountOk) return;
    setError(null);
    setStage("generating");

    try {
      const token = await getToken();
      const resp  = await fetch(`${supabaseUrl}/functions/v1/writing-diagnostic`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          stage:           "questions",
          writing_sample:  writingSample.trim(),
          track:           profile?.academic_track || "",
          discipline:      profile?.discipline    || "",
          research_stage:  profile?.current_research_stage || "",
        }),
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to generate questions.");

      const qs: string[] = data.questions || [];
      if (qs.length < 5) throw new Error("Received fewer than 5 questions. Please try again.");

      setQuestions(qs);
      setAnswers(["", "", "", "", ""]);
      setStage("answering");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
      setStage("submission");
    }
  };

  // ── Stage 3 → 4: Submit answers, get level assessment ────────────────────
  const handleSubmitAnswers = async () => {
    setError(null);
    setStage("assessing");

    try {
      const token = await getToken();
      const resp  = await fetch(`${supabaseUrl}/functions/v1/writing-diagnostic`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          stage:          "assess",
          writing_sample: writingSample.trim(),
          questions,
          answers,
        }),
      });

      const data: AssessmentResult = await resp.json();
      if (!resp.ok) throw new Error((data as unknown as { error?: string }).error || "Assessment failed.");

      const dbLevel = LEVEL_DB_MAP[data.level] || "developing";

      // Write results to DB in parallel
      await Promise.all([
        supabase.from("profiles").update({
          diagnostic_level:  dbLevel,
          onboarding_completed: true,
        }).eq("id", user!.id),

        supabase.from("writing_diagnostics" as never).insert({
          user_id:        user!.id,
          writing_sample: writingSample.trim(),
          questions:      questions,
          answers:        answers,
          assigned_level: data.level,
          rationale:      data.level_rationale,
          strengths:      data.three_strengths,
          gaps:           data.three_gaps,
          first_priority: data.first_priority,
        }),
      ]);

      await refreshProfile();
      setResult(data);
      setStage("results");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Assessment failed. Please try again.");
      setStage("answering");
    }
  };

  const allAnswersValid = answers.every((a) => sentenceCount(a) >= 2);

  // ── Render helpers ────────────────────────────────────────────────────────

  const profileStrip = profile ? (
    <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground bg-muted/30 border border-border rounded-md px-4 py-2.5 mb-6">
      {profile.academic_track && (
        <span><span className="font-medium text-foreground">Track:</span>{" "}
          {TRACK_LABELS[profile.academic_track as string] || profile.academic_track as string}</span>
      )}
      {profile.discipline && (
        <span><span className="font-medium text-foreground">Discipline:</span>{" "}
          {profile.discipline as string}</span>
      )}
      {profile.current_research_stage && (
        <span><span className="font-medium text-foreground">Stage:</span>{" "}
          {STAGE_LABELS[profile.current_research_stage as string] || profile.current_research_stage as string}</span>
      )}
    </div>
  ) : null;

  // ── Stage: submission ─────────────────────────────────────────────────────
  if (stage === "submission") {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-2xl space-y-6">

          {/* Header */}
          <div className="text-center space-y-2">
            <div className="mx-auto w-12 h-12 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground text-xl font-serif font-bold">D</span>
            </div>
            <h1 className="font-serif text-2xl font-bold text-foreground">
              Writing Diagnostic
            </h1>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              This short diagnostic helps the FIA Research Writing Mentor understand your current
              thinking level so it can guide you more effectively.
            </p>
          </div>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium">Submit Your Writing Sample</CardTitle>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Paste 150–300 words of your own academic writing below. This can be a paragraph
                from your proposal draft, a literature review attempt, or any academic paragraph
                you have written. Do not use AI-generated text.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {profileStrip}

              <Textarea
                value={writingSample}
                onChange={(e) => setWritingSample(e.target.value)}
                placeholder="Paste your writing here…"
                rows={10}
                className="resize-none text-sm leading-relaxed"
              />

              {/* Word count */}
              <div className="flex items-center justify-between">
                <span
                  className={`text-xs font-medium tabular-nums ${
                    wc === 0
                      ? "text-muted-foreground"
                      : wordCountOk
                      ? "text-primary"
                      : wc < WORD_MIN
                      ? "text-amber-600"
                      : "text-destructive"
                  }`}
                >
                  {wc} / {WORD_MAX} words
                  {wc > 0 && !wordCountOk && wc < WORD_MIN && ` — need at least ${WORD_MIN}`}
                  {wc > WORD_MAX && ` — over the ${WORD_MAX}-word limit`}
                </span>
                {wordCountOk && (
                  <span className="text-xs text-primary flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Good length
                  </span>
                )}
              </div>

              {error && (
                <div className="flex items-start gap-2 rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  {error}
                </div>
              )}

              <Button
                className="w-full gap-2"
                onClick={handleSubmitSample}
                disabled={!wordCountOk}
              >
                Submit for Assessment <ArrowRight className="w-4 h-4" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  // ── Stage: generating questions ───────────────────────────────────────────
  if (stage === "generating") {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="text-center space-y-4">
          <Loader2 className="w-10 h-10 animate-spin text-primary mx-auto" />
          <p className="font-medium text-foreground">Generating your diagnostic questions…</p>
          <p className="text-sm text-muted-foreground">This takes a few seconds.</p>
        </div>
      </div>
    );
  }

  // ── Stage: answering ──────────────────────────────────────────────────────
  if (stage === "answering") {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-2xl space-y-6">

          <div className="text-center space-y-1">
            <h1 className="font-serif text-2xl font-bold text-foreground">
              Diagnostic Questions
            </h1>
            <p className="text-sm text-muted-foreground">
              Answer each question in your own words. Write at least 2 sentences per answer.
            </p>
          </div>

          {profileStrip}

          {/* Writing sample reference */}
          <div className="rounded-lg border border-border bg-muted/20 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Your Writing Sample
            </p>
            <p className="text-sm text-foreground leading-relaxed line-clamp-4 font-serif">
              {writingSample}
            </p>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-xs text-destructive">
              <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              {error}
            </div>
          )}

          <div className="space-y-5">
            {questions.map((q, i) => {
              const asc = sentenceCount(answers[i]);
              const isValid = asc >= 2;
              return (
                <div key={i} className="space-y-2">
                  <div className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center mt-0.5">
                      {i + 1}
                    </span>
                    <p className="text-sm font-medium text-foreground leading-relaxed">{q}</p>
                  </div>
                  <div className="ml-9">
                    <Textarea
                      value={answers[i]}
                      onChange={(e) => {
                        const next = [...answers];
                        next[i] = e.target.value;
                        setAnswers(next);
                      }}
                      placeholder="Write your answer here (at least 2 sentences)…"
                      rows={4}
                      className="resize-none text-sm"
                    />
                    <p className={`text-[11px] mt-1 ${isValid ? "text-primary" : "text-muted-foreground"}`}>
                      {answers[i].trim().length === 0
                        ? "At least 2 sentences required"
                        : isValid
                        ? "✓ Good"
                        : "Keep writing — need at least 2 sentences"}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          <Button
            className="w-full gap-2"
            onClick={handleSubmitAnswers}
            disabled={!allAnswersValid}
          >
            Submit Answers <ArrowRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
    );
  }

  // ── Stage: assessing ──────────────────────────────────────────────────────
  if (stage === "assessing") {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="text-center space-y-4">
          <Loader2 className="w-10 h-10 animate-spin text-primary mx-auto" />
          <p className="font-medium text-foreground">Analysing your responses…</p>
          <p className="text-sm text-muted-foreground">Assigning your writing level. One moment.</p>
        </div>
      </div>
    );
  }

  // ── Stage: results ────────────────────────────────────────────────────────
  if (stage === "results" && result) {
    const display = LEVEL_DISPLAY[result.level] || LEVEL_DISPLAY.DEVELOPING;

    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-2xl space-y-6">

          {/* Header */}
          <div className="text-center space-y-3">
            <div className="mx-auto w-14 h-14 rounded-full bg-primary flex items-center justify-center">
              <CheckCircle className="w-7 h-7 text-primary-foreground" />
            </div>
            <h1 className="font-serif text-2xl font-bold text-foreground">
              Diagnostic Complete
            </h1>
          </div>

          {/* Level badge */}
          <div className={`rounded-xl ${display.bg} border border-primary/20 px-6 py-5 text-center`}>
            <p className="text-xs font-semibold uppercase tracking-widest text-primary/70 mb-1">
              Your Writing Level
            </p>
            <p className={`font-serif text-3xl font-bold ${display.text}`}>
              {display.label}
            </p>
          </div>

          {/* Rationale */}
          <Card>
            <CardContent className="pt-5 pb-5">
              <p className="text-sm text-foreground leading-relaxed font-serif">
                {result.level_rationale}
              </p>
            </CardContent>
          </Card>

          {/* Strengths */}
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
              What you are doing well
            </p>
            <div className="space-y-2">
              {result.three_strengths.map((s, i) => (
                <div key={i} className="flex items-start gap-2.5 rounded-lg bg-primary/5 border border-primary/15 px-3 py-2.5">
                  <CheckCircle className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                  <p className="text-sm text-foreground">{s}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Gaps */}
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
              Areas to develop
            </p>
            <div className="space-y-2">
              {result.three_gaps.map((g, i) => (
                <div key={i} className="flex items-start gap-2.5 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2.5">
                  <span className="w-3.5 h-3.5 rounded-full bg-amber-400 shrink-0 mt-0.5 flex-none" />
                  <p className="text-sm text-foreground">{g}</p>
                </div>
              ))}
            </div>
          </div>

          {/* First priority */}
          <div className="rounded-lg border-l-4 border-l-primary bg-primary/5 px-4 py-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-primary mb-1.5">
              Your First Priority
            </p>
            <p className="text-sm text-foreground leading-relaxed">{result.first_priority}</p>
          </div>

          {/* CTA */}
          <Button
            className="w-full gap-2"
            onClick={() => navigate("/research-writing/dashboard")}
          >
            Go to My Dashboard <ArrowRight className="w-4 h-4" />
          </Button>

        </div>
      </div>
    );
  }

  return null;
}
