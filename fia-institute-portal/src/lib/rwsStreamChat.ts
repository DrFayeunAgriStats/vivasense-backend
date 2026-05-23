export type RWSMode = "guide" | "explain" | "review" | "upgrade" | "supervisor" | "defense" | "interpret";

export interface RWSChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface RWSContext {
  track?: string;
  discipline?: string;
  stage?: string;
  title?: string;
  analysis_type?: string;
  data_summary?: string;
  drill_type?: string;
}

const RWS_URL = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/rws-ai`;

export async function rwsStreamChat({
  mode,
  messages,
  context,
  conversationId,
  authToken,
  onDelta,
  onDone,
  onError,
}: {
  mode: RWSMode;
  messages: RWSChatMessage[];
  context?: RWSContext;
  conversationId?: string;
  authToken?: string;
  onDelta: (text: string) => void;
  onDone: (convId?: string) => void;
  onError: (err: string) => void;
}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${authToken || import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
    };

    const resp = await fetch(RWS_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({
        mode,
        messages,
        context,
        conversation_id: conversationId,
        stream: true,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      onError(data.error || "Something went wrong. Please try again.");
      return;
    }

    const convId = resp.headers.get("X-Conversation-Id") || undefined;

    if (!resp.body) {
      onError("No response stream.");
      return;
    }

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
        if (jsonStr === "[DONE]") {
          onDone(convId);
          return;
        }
        try {
          const parsed = JSON.parse(jsonStr);
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) onDelta(content);
        } catch {
          buffer = line + "\n" + buffer;
          break;
        }
      }
    }

    onDone(convId);
  } catch (e: any) {
    clearTimeout(timeout);
    if (e.name === "AbortError") {
      onError("Request timed out. Please try again.");
    } else {
      onError(e.message || "Connection error.");
    }
  }
}
