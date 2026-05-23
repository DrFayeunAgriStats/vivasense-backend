import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { Send, Bot, Loader2, MessageSquare, Sparkles, AlertCircle, Download, ShieldCheck, ChevronDown } from "lucide-react";
import { downloadResultsAsText } from "@/lib/downloadResults";
import ReactMarkdown from "react-markdown";
import { FastAPIResultsData } from "./VivaSenseFastAPIResults";
import type { GroundingCheck } from "./VivaSenseMultiTraitInterpretation";
const INTERPRET_URL = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/vivasense-interpret`;

interface Props {
  analysisType: string;
  results: FastAPIResultsData;
}

type ChatMessage = { role: "user" | "assistant"; content: string };

function parseSSEChunk(line: string): string {
  if (!line.startsWith("data: ")) return "";
  const jsonStr = line.slice(6).trim();
  if (jsonStr === "[DONE]") return "";
  try {
    const parsed = JSON.parse(jsonStr);
    // Anthropic content_block_delta: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
    if (parsed.type === "content_block_delta" && parsed.delta?.text) {
      return parsed.delta.text;
    }
    // OpenAI-style format
    if (parsed.choices?.[0]?.delta?.content) {
      return parsed.choices[0].delta.content;
    }
    // Skip non-content events (message_start, content_block_start, ping, etc.)
    if (parsed.type && parsed.type !== "content_block_delta") {
      return "";
    }
    return parsed.content || parsed.text || "";
  } catch {
    return jsonStr;
  }
}

export function VivaSenseInterpretation({ analysisType, results }: Props) {
  const [interpretation, setInterpretation] = useState("");
  const [groundingCheck, setGroundingCheck] = useState<GroundingCheck | undefined>();
  const [isInterpreting, setIsInterpreting] = useState(false);
  const [interpretError, setInterpretError] = useState<string | null>(null);

  const [followUpMessages, setFollowUpMessages] = useState<ChatMessage[]>([]);
  const [followUpInput, setFollowUpInput] = useState("");
  const [isFollowUpLoading, setIsFollowUpLoading] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: "smooth" });
  }, [followUpMessages]);

  const streamFromEdgeFunction = async (
    body: Record<string, unknown>,
    onDelta: (text: string) => void,
    signal?: AbortSignal,
  ): Promise<string> => {
    const resp = await fetch(INTERPRET_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      throw new Error(errBody.error || `Server error (${resp.status})`);
    }

    if (!resp.body) {
      const data = await resp.json();
      const text = data.interpretation || data.response || JSON.stringify(data);
      onDelta(text);
      return text;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newlineIdx: number;
      while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
        let line = buffer.slice(0, newlineIdx).replace(/\r$/, "");
        buffer = buffer.slice(newlineIdx + 1);
        if (!line || line.startsWith(":") || line.trim() === "") continue;
        const content = parseSSEChunk(line);
        if (content) {
          full += content;
          onDelta(full);
        }
      }
    }

    // Flush remaining buffer
    if (buffer.trim()) {
      const content = parseSSEChunk(buffer.trim());
      if (content) {
        full += content;
        onDelta(full);
      }
    }

    return full;
  };

  const runInterpretation = async (signal?: AbortSignal) => {
    setIsInterpreting(true);
    setInterpretError(null);
    setInterpretation("");
    setGroundingCheck(undefined);

    try {
      // Strip large redundant fields to stay within AI token limits
      const { html_tables, plots, ...rest } = results;
      const slimResults = { ...rest } as Record<string, unknown>;
      delete slimResults["publication_tables"];
      const result = await streamFromEdgeFunction(
        { analysis_type: analysisType, results: slimResults },
        (text) => setInterpretation(text),
        signal,
      );

      if (!result.trim()) {
        setInterpretation("No interpretation was returned.");
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setInterpretError(err.message || "Failed to get interpretation. Please try again.");
      }
    } finally {
      setIsInterpreting(false);
    }
  };

  // Auto-interpret on mount
  useEffect(() => {
    const controller = new AbortController();
    runInterpretation(controller.signal);
    return () => controller.abort();
  }, [analysisType, results]);

  const sendFollowUp = async () => {
    const text = followUpInput.trim();
    if (!text || isFollowUpLoading) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setFollowUpInput("");
    setFollowUpMessages((prev) => [...prev, userMsg]);
    setIsFollowUpLoading(true);

    try {
      const formattedMessages = [...followUpMessages, userMsg].map(m => ({
        role: m.role,
        content: m.content,
      }));

      let assistantAdded = false;
      await streamFromEdgeFunction(
        {
          analysis_type: analysisType,
          results,
          interpretation,
          messages: formattedMessages,
          mode: "followup",
        },
        (soFar) => {
          if (!assistantAdded) {
            assistantAdded = true;
            setFollowUpMessages((prev) => [...prev, { role: "assistant", content: soFar }]);
          } else {
            setFollowUpMessages((prev) =>
              prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: soFar } : m))
            );
          }
        },
      );
    } catch (err: any) {
      const errMsg = err instanceof Error ? err.message : String(err);
      setFollowUpMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errMsg}` },
      ]);
    } finally {
      setIsFollowUpLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendFollowUp();
    }
  };

  return (
    <div className="space-y-6">
      {/* Dr. Fayeun's Interpretation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-primary" />
            Dr. Fayeun's Interpretation
            {groundingCheck?.passed && (
              <Badge variant="outline" className="text-green-600 border-green-300 dark:text-green-400 dark:border-green-700 text-xs ml-auto">
                <ShieldCheck className="w-3.5 h-3.5 mr-1" />
                Grounded ✓
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isInterpreting && !interpretation && (
            <div className="flex items-center gap-3 text-muted-foreground">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Generating interpretation…</span>
            </div>
          )}
          {interpretError && (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-3 text-destructive text-sm bg-destructive/10 rounded-lg p-4">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{interpretError}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => runInterpretation()}
                className="self-start"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Regenerate Interpretation
              </Button>
            </div>
          )}
          {!interpretation && !isInterpreting && !interpretError && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => runInterpretation()}
            >
              <Sparkles className="w-4 h-4 mr-2" />
              Regenerate Interpretation
            </Button>
          )}
          {interpretation && (
            <>
              {/* Grounding check warning */}
              {groundingCheck && !groundingCheck.passed && (
                <Collapsible defaultOpen className="mb-4">
                  <div className="rounded-lg border border-yellow-400 bg-yellow-50 dark:bg-yellow-950/30 dark:border-yellow-600 p-4">
                    <CollapsibleTrigger className="w-full flex items-center justify-between text-sm font-semibold text-yellow-800 dark:text-yellow-200">
                      <span className="flex items-center gap-2">
                        <AlertCircle className="w-4 h-4" />
                        Statistical Review Alerts ({groundingCheck.warning_count} warning{groundingCheck.warning_count !== 1 ? "s" : ""})
                      </span>
                      <ChevronDown className="w-4 h-4 transition-transform duration-200 [[data-state=open]>&]:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <ul className="list-disc list-inside space-y-1 mt-2">
                        {groundingCheck.warnings.map((w, i) => (
                          <li key={i} className="text-sm text-yellow-700 dark:text-yellow-300">{w}</li>
                        ))}
                      </ul>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              )}
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown>{interpretation}</ReactMarkdown>
              </div>
              <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  ⚠️ <strong>Academic Integrity Reminder:</strong> This interpretation is a starting point for your own analysis. Verify all numbers against the tables above. Adapt any suggested text to your own words and field context. Discuss with your supervisor before submitting.
                </p>
              </div>
            </>
          )}
          {isInterpreting && interpretation && (
            <Loader2 className="w-4 h-4 animate-spin text-muted-foreground mt-2" />
          )}
          {interpretation && !isInterpreting && (
            <div className="mt-6 pt-4 border-t border-border">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  try {
                    downloadResultsAsText(analysisType, results as any, interpretation, followUpMessages);
                  } catch (err) {
                    console.error("Download failed:", err);
                  }
                }}
              >
                <Download className="w-4 h-4 mr-2" />
                Download Results
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Follow-up Chat */}
      {(interpretation || interpretError) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <MessageSquare className="w-6 h-6 text-primary" />
              Follow-up Questions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {followUpMessages.length > 0 && (
              <div ref={chatScrollRef} className="max-h-[400px] overflow-y-auto space-y-3 mb-4">
                {followUpMessages.map((msg, i) => (
                  <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    {msg.role === "assistant" && (
                      <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                        <Bot className="w-3.5 h-3.5 text-primary-foreground" />
                      </div>
                    )}
                    <div className={`max-w-[80%] rounded-xl px-4 py-3 text-sm ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-foreground"
                    }`}>
                      <div className="prose prose-sm max-w-none dark:prose-invert">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ))}
                {isFollowUpLoading && followUpMessages[followUpMessages.length - 1]?.role === "user" && (
                  <div className="flex gap-3">
                    <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                      <Bot className="w-3.5 h-3.5 text-primary-foreground" />
                    </div>
                    <div className="bg-muted rounded-xl px-4 py-3">
                      <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    </div>
                  </div>
                )}
              </div>
            )}
            <div className="flex gap-2">
              <Textarea
                value={followUpInput}
                onChange={(e) => setFollowUpInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a follow-up question about these results…"
                className="resize-none min-h-[44px] max-h-[100px]"
                rows={1}
                disabled={isFollowUpLoading}
              />
              <Button
                onClick={sendFollowUp}
                disabled={isFollowUpLoading || !followUpInput.trim()}
                size="icon"
                className="flex-shrink-0"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
