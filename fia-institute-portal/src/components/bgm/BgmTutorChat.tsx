import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, ArrowLeft, Dna } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useToast } from "@/hooks/use-toast";
import type { BgmStudent } from "@/hooks/useBgmSession";

type ChatMessage = { role: "user" | "assistant"; content: string };

const CHAT_URL = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/csp811-tutor`;

const TOPICS = [
  "Module 1 – Foundations (P=G+E)",
  "Module 2 – Experimental Designs",
  "Module 3 – Variance Components",
  "Module 4 – Heritability",
  "Module 5 – Generation Mean Analysis",
  "Module 6 – Diallel & Combining Ability",
  "Module 7 – Regression & Path Analysis",
  "Module 8 – G×E & Stability",
  "Module 9 – Selection Indices",
  "Module 10 – Genomic & Molecular Prediction",
  "Module 11 – ML in Plant Breeding",
];

const QUICK_QUESTIONS = [
  "Begin Module 1",
  "Show my progress dashboard",
  "Derive narrow-sense heritability",
  "What is the predictability ratio in diallel?",
  "Explain AMMI model step by step",
];

function makeWelcome(student: BgmStudent): ChatMessage {
  return {
    role: "assistant",
    content: `Welcome, **${student.full_name}**.\n\nI am the **Biometrical Genetics Mastery Tutor** — your postgraduate research mentor for this mastery-based programme covering 11 modules.\n\nYou are currently on **Module ${student.current_module}**: *${TOPICS[student.current_module - 1]?.split(" – ")[1] || ""}*.\n\n**How this works:**\n- Strict module-by-module progression\n- Each module: teaching → derivations → worked examples → 10-question assessment\n- ≥ 7/10 required to advance\n- All 11 modules → completion token for certification\n\nType **"Begin Module ${student.current_module}"** to continue.\n\n*Variance is the raw material of selection.* — Dr. Fayeun`,
  };
}

async function streamChat({
  messages, onDelta, onDone, onError,
}: {
  messages: ChatMessage[];
  onDelta: (text: string) => void;
  onDone: () => void;
  onError: (err: string) => void;
}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  const resp = await fetch(CHAT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
    },
    body: JSON.stringify({ messages }),
    signal: controller.signal,
  });

  clearTimeout(timeout);

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    onError(data.error || "Something went wrong. Please try again.");
    return;
  }
  if (!resp.body) { onError("No response stream."); return; }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex: number;
    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
      let line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (line.startsWith(":") || line.trim() === "") continue;
      if (!line.startsWith("data: ")) continue;
      const jsonStr = line.slice(6).trim();
      if (jsonStr === "[DONE]") { onDone(); return; }
      try {
        const parsed = JSON.parse(jsonStr);
        if (parsed.type === "content_block_delta" && parsed.delta?.text) {
          onDelta(parsed.delta.text);
          continue;
        }
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) onDelta(content);
      } catch {
        buffer = line + "\n" + buffer;
        break;
      }
    }
  }

  if (buffer.trim()) {
    for (let raw of buffer.split("\n")) {
      if (!raw) continue;
      if (raw.endsWith("\r")) raw = raw.slice(0, -1);
      if (!raw.startsWith("data: ")) continue;
      const jsonStr = raw.slice(6).trim();
      if (jsonStr === "[DONE]") continue;
      try {
        const parsed = JSON.parse(jsonStr);
        if (parsed.type === "content_block_delta" && parsed.delta?.text) {
          onDelta(parsed.delta.text);
          continue;
        }
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) onDelta(content);
      } catch { /* ignore */ }
    }
  }
  onDone();
}

type Props = {
  student: BgmStudent;
  onBack: () => void;
};

export function BgmTutorChat({ student, onBack }: Props) {
  const welcome = makeWelcome(student);
  const [messages, setMessages] = useState<ChatMessage[]>([welcome]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [topic, setTopic] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || isLoading) return;

    const prefix = topic ? `[Topic: ${topic}] ` : "";
    const userMsg: ChatMessage = { role: "user", content: prefix + msg };
    setInput("");
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    let assistantSoFar = "";
    const upsert = (chunk: string) => {
      assistantSoFar += chunk;
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && prev.length > 1 && prev[prev.length - 2]?.role === "user") {
          return prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: assistantSoFar } : m));
        }
        return [...prev, { role: "assistant", content: assistantSoFar }];
      });
    };

    try {
      const history = messages.filter((m) => m !== welcome);
      await streamChat({
        messages: [...history, userMsg],
        onDelta: upsert,
        onDone: () => setIsLoading(false),
        onError: (err) => {
          toast({ title: "Error", description: err, variant: "destructive" });
          setIsLoading(false);
        },
      });
    } catch {
      toast({ title: "Error", description: "Connection failed. Please try again.", variant: "destructive" });
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <div className="bg-primary text-primary-foreground py-3">
        <div className="container-wide flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}
            className="text-primary-foreground hover:bg-primary-foreground/10">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <Dna className="w-5 h-5" />
          <div>
            <h1 className="font-serif text-lg font-bold">Biometrical Genetics Mastery Tutor</h1>
            <p className="text-primary-foreground/70 text-[11px]">Module {student.current_module} · {student.full_name}</p>
          </div>
        </div>
      </div>

      <div className="flex-1 container-wide py-4 flex flex-col max-w-3xl mx-auto">
        {/* Topic Selector */}
        <div className="mb-3">
          <Select value={topic} onValueChange={setTopic}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Choose a topic to focus your questions..." />
            </SelectTrigger>
            <SelectContent>
              {TOPICS.map((t) => (
                <SelectItem key={t} value={t}>{t}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Chat card */}
        <div className="border border-border rounded-2xl bg-card overflow-hidden flex flex-col flex-1 min-h-0">
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                {msg.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-primary-foreground" />
                  </div>
                )}
                <div className={`max-w-[80%] rounded-xl px-4 py-3 text-sm ${
                  msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
                }`}>
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{msg.content}</ReactMarkdown>
                  </div>
                </div>
                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-accent-foreground" />
                  </div>
                )}
              </div>
            ))}
            {isLoading && messages[messages.length - 1]?.role === "user" && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-primary-foreground" />
                </div>
                <div className="bg-muted rounded-xl px-4 py-3">
                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                </div>
              </div>
            )}
          </div>

          {/* Quick questions */}
          <div className="px-4 pt-2 flex flex-wrap gap-2">
            {QUICK_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                disabled={isLoading}
                className="text-xs px-3 py-1.5 rounded-full border border-border bg-background hover:bg-muted text-foreground transition-colors disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>

          {/* Input */}
          <div className="border-t border-border p-4">
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about biometrical genetics..."
                className="resize-none min-h-[44px] max-h-[120px]"
                rows={1}
                disabled={isLoading}
              />
              <Button onClick={() => send()} disabled={isLoading || !input.trim()} size="icon" className="flex-shrink-0">
                <Send className="w-4 h-4" />
              </Button>
            </div>
            <p className="text-[10px] text-muted-foreground mt-2 text-center">
              Powered by FIA | AI-assisted learning tool · © Dr. Fayeun Lawrence Stephen
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
