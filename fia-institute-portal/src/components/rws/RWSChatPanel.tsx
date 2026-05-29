import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Send, Loader2, Trash2, FileText, X, Download, Mail, Save, CheckCircle,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { rwsStreamChat, type RWSMode, type RWSChatMessage, type RWSContext } from "@/lib/rwsStreamChat";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

const MODE_OPTIONS: { value: RWSMode; label: string; desc: string }[] = [
  { value: "guide",      label: "Guide Mode",       desc: "Socratic reasoning through research problems" },
  { value: "explain",    label: "Explain Mode",      desc: "Clear explanations of research concepts" },
  { value: "review",     label: "Review Mode",       desc: "Feedback on your writing" },
  { value: "upgrade",    label: "Upgrade Mode",      desc: "Improve writing structure" },
  { value: "supervisor", label: "Supervisor Lens",   desc: "Simulated supervisor feedback" },
  { value: "defense",    label: "Defense Examiner",  desc: "Viva voce practice questions" },
];

const STAGE_LABELS: Record<string, string> = {
  topic_proposal:      "Topic / Proposal",
  literature_review:   "Literature Review",
  methodology:         "Methodology",
  data_analysis:       "Data Analysis",
  results_writing:     "Results Writing",
  discussion:          "Discussion",
  defense_preparation: "Defense Preparation",
};

interface Props {
  defaultMode?: RWSMode;
  context?: RWSContext;
  showModeSelector?: boolean;
  placeholder?: string;
}

interface BriefingData {
  briefing_text: string;
  conversation_id: string;
  mode: string;
  stage: string;
  topic: string;
  exchange_count: number;
  student_name: string;
  generated_at: string;
}

export function RWSChatPanel({
  defaultMode = "guide",
  context,
  showModeSelector = true,
  placeholder,
}: Props) {
  const { session, user, profile } = useAuth() as {
    session: { access_token: string } | null;
    user: { id: string } | null;
    profile: Record<string, unknown> | null;
    loading: boolean;
  };

  const [mode, setMode]                     = useState<RWSMode>(defaultMode);
  const [messages, setMessages]             = useState<RWSChatMessage[]>([]);
  const [input, setInput]                   = useState("");
  const [isStreaming, setIsStreaming]        = useState(false);
  const [streamingText, setStreamingText]   = useState("");
  const [conversationId, setConversationId] = useState<string | undefined>();

  // Briefing state
  const [generatingBriefing, setGeneratingBriefing] = useState(false);
  const [briefingData, setBriefingData]             = useState<BriefingData | null>(null);
  const [showBriefingModal, setShowBriefingModal]   = useState(false);
  const [briefingSaved, setBriefingSaved]           = useState(false);
  const [savingBriefing, setSavingBriefing]         = useState(false);
  const [endSessionError, setEndSessionError]       = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamingText]);

  // ── Send message ────────────────────────────────────────────────────────────
  const handleSend = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const newMessages: RWSChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(newMessages);
    setInput("");
    setIsStreaming(true);
    setStreamingText("");

    let accumulated = "";
    await rwsStreamChat({
      mode,
      messages: newMessages,
      context,
      conversationId,
      authToken: session?.access_token,
      onDelta: (delta) => {
        accumulated += delta;
        setStreamingText(accumulated);
      },
      onDone: (convId) => {
        setMessages((prev) => [...prev, { role: "assistant", content: accumulated }]);
        setStreamingText("");
        setIsStreaming(false);
        if (convId) setConversationId(convId);
      },
      onError: (err) => {
        setMessages((prev) => [...prev, { role: "assistant", content: `⚠️ ${err}` }]);
        setStreamingText("");
        setIsStreaming(false);
      },
    });
  };

  const handleClear = () => {
    setMessages([]);
    setConversationId(undefined);
    setStreamingText("");
    setBriefingData(null);
    setBriefingSaved(false);
    setEndSessionError(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── End session & generate briefing ────────────────────────────────────────
  const userMsgCount = messages.filter((m) => m.role === "user").length;
  const canEndSession =
    userMsgCount >= 3 && !isStreaming && !!conversationId && !!session?.access_token;

  const handleEndSession = async () => {
    if (!canEndSession) return;
    setGeneratingBriefing(true);
    setEndSessionError(null);
    try {
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
      const resp = await fetch(`${supabaseUrl}/functions/v1/generate-briefing`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session!.access_token}`,
        },
        body: JSON.stringify({ conversation_id: conversationId }),
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to generate briefing.");

      setBriefingData(data as BriefingData);
      setBriefingSaved(false);
      setShowBriefingModal(true);
    } catch (err: unknown) {
      setEndSessionError(
        err instanceof Error ? err.message : "Could not generate briefing. Please try again."
      );
    } finally {
      setGeneratingBriefing(false);
    }
  };

  // ── Save briefing to Supabase ───────────────────────────────────────────────
  const handleSaveBriefing = async () => {
    if (!briefingData || !user) return;
    setSavingBriefing(true);
    try {
      const { error } = await supabase.from("supervisor_briefings" as never).insert({
        user_id:         user.id,
        conversation_id: briefingData.conversation_id,
        briefing_text:   briefingData.briefing_text,
        mode:            briefingData.mode,
        stage:           briefingData.stage,
        topic:           briefingData.topic,
        exchange_count:  briefingData.exchange_count,
      });
      if (error) throw error;
      setBriefingSaved(true);
    } catch {
      setEndSessionError("Could not save briefing. Please try again.");
    } finally {
      setSavingBriefing(false);
    }
  };

  // ── PDF download via new window print ──────────────────────────────────────
  const handleDownloadPDF = () => {
    if (!briefingData) return;
    const w = window.open("", "_blank");
    if (!w) return;

    const date = new Date(briefingData.generated_at).toLocaleDateString("en-GB", {
      day: "numeric", month: "long", year: "numeric",
    });
    const stageLabel = STAGE_LABELS[briefingData.stage] || briefingData.stage.replace(/_/g, " ");
    const safeText = briefingData.briefing_text
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    w.document.write(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Supervisor Briefing — ${date}</title>
  <style>
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 680px; margin: 48px auto; line-height: 1.8; color: #1a1a1a; font-size: 13px; }
    .header { border-bottom: 2px solid #1a5c38; padding-bottom: 14px; margin-bottom: 22px; }
    h1 { font-size: 17px; color: #1a5c38; margin: 0 0 3px; letter-spacing: 0.01em; }
    .institution { font-size: 11px; color: #555; margin: 0; }
    .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 28px; font-size: 12px; color: #333; margin-bottom: 24px; }
    .meta-grid strong { color: #111; }
    .body { font-size: 13px; line-height: 1.85; white-space: pre-wrap; }
    .footer { margin-top: 44px; padding-top: 10px; border-top: 1px solid #ccc; font-size: 10px; color: #888; }
    @media print { body { margin: 28px 36px; } button { display: none; } }
  </style>
</head>
<body>
  <div class="header">
    <h1>Supervisor Briefing Note</h1>
    <p class="institution">Field-to-Insight Academy — Research Writing Mentor</p>
  </div>
  <div class="meta-grid">
    <div><strong>Student:</strong> ${briefingData.student_name}</div>
    <div><strong>Date:</strong> ${date}</div>
    <div><strong>Research Stage:</strong> ${stageLabel}</div>
    <div><strong>Session Mode:</strong> ${briefingData.mode}</div>
    <div><strong>Exchanges:</strong> ${briefingData.exchange_count}</div>
  </div>
  <div class="body">${safeText}</div>
  <div class="footer">
    Generated by the FIA Research Writing Mentor &nbsp;·&nbsp;
    AI-assisted professional summary — NOT the AI conversation text &nbsp;·&nbsp;
    fieldtoinsightacademy.com.ng
  </div>
  <script>window.onload = () => { window.print(); }<\/script>
</body>
</html>`);
    w.document.close();
  };

  // ── Email via mailto: ────────────────────────────────────────────────────────
  const handleCopyEmail = () => {
    if (!briefingData) return;
    const supervisorEmail = (profile?.supervisor_email as string) || "";
    const date = new Date(briefingData.generated_at).toLocaleDateString("en-GB", {
      day: "numeric", month: "long", year: "numeric",
    });
    const subject = encodeURIComponent(
      `Research Progress Briefing — ${briefingData.student_name} — ${date}`
    );
    const body = encodeURIComponent(
      `Dear Supervisor,\n\nPlease find below a progress briefing from ${briefingData.student_name}'s recent FIA Research Writing Mentor session (${date}).\n\n` +
      `─────────────────────────────────────────\n\n${briefingData.briefing_text}\n\n` +
      `─────────────────────────────────────────\n\n` +
      `This is an AI-assisted professional summary — not a transcript of the AI conversation.\n\n` +
      `Field-to-Insight Academy\nfieldtoinsightacademy.com.ng`
    );
    window.open(`mailto:${supervisorEmail}?subject=${subject}&body=${body}`, "_blank");
  };

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <>
      {/* ── Chat Panel ─────────────────────────────────────────────────────── */}
      <div className="flex flex-col h-full border border-border rounded-lg bg-card overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 p-3 border-b border-border bg-muted/30">
          {showModeSelector && (
            <Select value={mode} onValueChange={(v) => setMode(v as RWSMode)}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODE_OPTIONS.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    <div>
                      <span className="font-medium">{m.label}</span>
                      <span className="text-xs text-muted-foreground ml-2 hidden sm:inline">
                        {m.desc}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              className="ml-auto text-muted-foreground"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>

        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px] max-h-[500px]"
        >
          {messages.length === 0 && !isStreaming && (
            <div className="text-center text-muted-foreground text-sm py-12">
              <p className="font-medium">
                {MODE_OPTIONS.find((m) => m.value === mode)?.label || "AI Mentor"}
              </p>
              <p className="text-xs mt-1">
                {MODE_OPTIONS.find((m) => m.value === mode)?.desc}
              </p>
              <p className="text-xs mt-4 text-muted-foreground/70">
                Type your question or paste your writing below.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-4 py-3 text-sm ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                }`}
              >
                {msg.role === "assistant" ? (
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {isStreaming && streamingText && (
            <div className="flex justify-start">
              <div className="max-w-[85%] rounded-lg px-4 py-3 text-sm bg-muted text-foreground">
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <ReactMarkdown>{streamingText}</ReactMarkdown>
                </div>
              </div>
            </div>
          )}

          {isStreaming && !streamingText && (
            <div className="flex justify-start">
              <div className="rounded-lg px-4 py-3 bg-muted">
                <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
              </div>
            </div>
          )}
        </div>

        {/* End Session Banner — only visible after ≥3 user exchanges */}
        {canEndSession && (
          <div className="border-t border-border bg-primary/5 px-3 py-2 space-y-1">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground">
                Ready to wrap up? Generate a professional briefing note for your supervisor.
              </p>
              <Button
                size="sm"
                variant="outline"
                className="gap-1.5 text-xs border-primary/50 text-primary hover:bg-primary hover:text-primary-foreground shrink-0"
                onClick={handleEndSession}
                disabled={generatingBriefing}
              >
                {generatingBriefing ? (
                  <><Loader2 className="w-3 h-3 animate-spin" /> Generating…</>
                ) : (
                  <><FileText className="w-3 h-3" /> End Session &amp; Brief Supervisor</>
                )}
              </Button>
            </div>
            {endSessionError && (
              <p className="text-xs text-destructive">{endSessionError}</p>
            )}
          </div>
        )}

        {/* Input */}
        <div className="border-t border-border p-3">
          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder || "Ask a question or paste your writing..."}
              rows={2}
              className="resize-none text-sm"
              disabled={isStreaming}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              size="icon"
              className="shrink-0 self-end"
            >
              {isStreaming
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <Send className="w-4 h-4" />}
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1.5 flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
            FIA AI tools guide research thinking but do not generate thesis or research paper text.
          </p>
        </div>
      </div>

      {/* ── Briefing Modal ───────────────────────────────────────────────────── */}
      {showBriefingModal && briefingData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">

            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-border">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center">
                  <FileText className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <h2 className="font-serif text-base font-semibold text-foreground leading-tight">
                    Supervisor Briefing Note
                  </h2>
                  <p className="text-[11px] text-muted-foreground">
                    Field-to-Insight Academy — Research Writing Mentor
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setShowBriefingModal(false)}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Student notice */}
            <div className="px-5 pt-4 pb-2">
              <p className="text-[11px] italic text-muted-foreground leading-relaxed border-l-2 border-primary/30 pl-3">
                This briefing summarises your session for your supervisor. Review it before saving.
                It does not contain your exact conversation — only a professional summary of your
                progress and gaps.
              </p>
            </div>

            {/* Session metadata */}
            <div className="px-5 pt-2 pb-1 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
              <span>
                <strong className="text-foreground font-medium">Stage:</strong>{" "}
                {STAGE_LABELS[briefingData.stage] || briefingData.stage.replace(/_/g, " ")}
              </span>
              <span>
                <strong className="text-foreground font-medium">Mode:</strong> {briefingData.mode}
              </span>
              <span>
                <strong className="text-foreground font-medium">Exchanges:</strong>{" "}
                {briefingData.exchange_count}
              </span>
              <span>
                <strong className="text-foreground font-medium">Generated:</strong>{" "}
                {new Date(briefingData.generated_at).toLocaleDateString("en-GB", {
                  day: "numeric", month: "short", year: "numeric",
                })}
              </span>
            </div>

            {/* Briefing text */}
            <div className="px-5 pt-3 pb-4">
              <div className="rounded-lg border border-border bg-muted/20 p-4">
                <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap font-serif">
                  {briefingData.briefing_text}
                </p>
              </div>
            </div>

            {/* Save confirmation */}
            {briefingSaved && (
              <div className="mx-5 mb-3 flex items-center gap-2 rounded-md bg-primary/10 border border-primary/20 px-3 py-2 text-xs text-primary">
                <CheckCircle className="w-3.5 h-3.5 shrink-0" />
                Briefing saved. Find it in the Supervisor Briefings section on your dashboard.
              </div>
            )}

            {/* Error in modal */}
            {endSessionError && briefingSaved === false && (
              <p className="mx-5 mb-2 text-xs text-destructive">{endSessionError}</p>
            )}

            {/* Action buttons */}
            <div className="flex flex-wrap items-center gap-2 px-5 pb-5 pt-3 border-t border-border">
              {!briefingSaved && (
                <Button
                  size="sm"
                  className="gap-1.5 text-xs"
                  onClick={handleSaveBriefing}
                  disabled={savingBriefing}
                >
                  {savingBriefing ? (
                    <><Loader2 className="w-3 h-3 animate-spin" /> Saving…</>
                  ) : (
                    <><Save className="w-3 h-3" /> Save &amp; Make Available for Download</>
                  )}
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5 text-xs"
                onClick={handleDownloadPDF}
              >
                <Download className="w-3 h-3" /> Download PDF
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5 text-xs"
                onClick={handleCopyEmail}
              >
                <Mail className="w-3 h-3" /> Copy to Email
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-muted-foreground ml-auto"
                onClick={() => setShowBriefingModal(false)}
              >
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
