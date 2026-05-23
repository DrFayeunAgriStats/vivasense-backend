import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Send, Loader2, Trash2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { rwsStreamChat, type RWSMode, type RWSChatMessage, type RWSContext } from "@/lib/rwsStreamChat";
import { useAuth } from "@/contexts/AuthContext";

const MODE_OPTIONS: { value: RWSMode; label: string; desc: string }[] = [
  { value: "guide", label: "Guide Mode", desc: "Socratic reasoning through research problems" },
  { value: "explain", label: "Explain Mode", desc: "Clear explanations of research concepts" },
  { value: "review", label: "Review Mode", desc: "Feedback on your writing" },
  { value: "upgrade", label: "Upgrade Mode", desc: "Improve writing structure" },
  { value: "supervisor", label: "Supervisor Lens", desc: "Simulated supervisor feedback" },
  { value: "defense", label: "Defense Examiner", desc: "Viva voce practice questions" },
];

interface Props {
  defaultMode?: RWSMode;
  context?: RWSContext;
  showModeSelector?: boolean;
  placeholder?: string;
}

export function RWSChatPanel({ defaultMode = "guide", context, showModeSelector = true, placeholder }: Props) {
  const { session } = useAuth();
  const [mode, setMode] = useState<RWSMode>(defaultMode);
  const [messages, setMessages] = useState<RWSChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [conversationId, setConversationId] = useState<string>();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamingText]);

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
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
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
                    <span className="text-xs text-muted-foreground ml-2 hidden sm:inline">{m.desc}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {messages.length > 0 && (
          <Button variant="ghost" size="sm" onClick={handleClear} className="ml-auto text-muted-foreground">
            <Trash2 className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px] max-h-[500px]">
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
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
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
          <Button onClick={handleSend} disabled={!input.trim() || isStreaming} size="icon" className="shrink-0 self-end">
            {isStreaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </div>
        <p className="text-[10px] text-muted-foreground mt-1.5 flex items-center gap-1">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
          FIA AI tools guide research thinking but do not generate thesis or research paper text.
        </p>
      </div>
    </div>
  );
}
