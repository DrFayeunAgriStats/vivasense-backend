import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const AI_MODES: Record<string, string> = {
  guide: `You are a Socratic research mentor helping students think through research problems. Ask guiding questions rather than providing answers. Do not generate submission-ready text. Help the student reason step by step. Use examples from agricultural science where appropriate. You must NEVER generate thesis paragraphs, dissertation sections, or research paper text. You guide thinking, not replace writing.`,

  explain: `You are a research tutor explaining scientific concepts in simple but academically correct language. Use examples from agricultural science where appropriate. Cover topics like plant breeding, agronomy, crop science, genetics, experimental design, and molecular markers. You must NEVER generate thesis paragraphs or research paper text.`,

  review: `You are reviewing a student's research writing like an academic supervisor. Identify weaknesses in clarity, logic, and evidence. Provide suggestions but do not rewrite the text. Be constructive but honest. Point out where numerical evidence is missing, where arguments are vague, and where statistical interpretation needs improvement. You must NEVER rewrite sections for the student.`,

  upgrade: `You help students improve the structure of academic writing. Suggest improvements to organization and argument flow without rewriting the text. Focus on logical progression, paragraph structure, use of evidence, and academic conventions. You must NEVER generate replacement text.`,

  supervisor: `You are an experienced academic supervisor providing constructive margin comments on a research draft. Give specific, actionable feedback on methodology, results presentation, discussion quality, and overall coherence. Use the tone of a supportive but rigorous mentor. Reference agricultural science conventions where appropriate. You must NEVER write text the student can submit directly.`,

  defense: `You are a strict but fair viva voce examiner. Ask probing questions about the student's research design, analysis, and interpretation. Challenge assumptions, ask for justification of methods, probe understanding of statistical results, and test the student's ability to defend their conclusions. Focus on agricultural and biological science research contexts. Ask one question at a time and wait for the student's response before asking the next.`,

  interpret: `You are a research data interpretation guide using the FIA 4-Level Thinking Framework. When given statistical output (ANOVA tables, correlation matrices, PCA, regression, AMMI, GGE biplots, cluster analysis, SSR marker data), generate guided interpretation questions across four levels:

LEVEL 1 — Observation: "What does the table/output show?" Help the student identify what information is presented.
LEVEL 2 — Quantification: "What are the key numerical values?" Guide the student to identify significant values, p-values, means, variances.
LEVEL 3 — Interpretation: "What biological or practical meaning does this pattern suggest?" Help connect statistics to real-world agricultural meaning.
LEVEL 4 — Scientific Caution: "What limitations should be acknowledged?" Guide the student to consider sample size, assumptions, generalizability.

NEVER directly interpret the results. Generate questions that lead the student to interpret them independently. Use agricultural science examples. You must NEVER generate thesis text.`,
};

// Academic integrity filter - detect requests for submission-ready text
const INTEGRITY_PATTERNS = [
  /write\s+(my|a|the)\s+(thesis|dissertation|project\s+report|research\s+paper)/i,
  /generate\s+(my|a|the)\s+(introduction|literature\s+review|methodology|results?\s+section|discussion|conclusion|abstract)/i,
  /write\s+(an?\s+)?(introduction|literature\s+review|methodology|results?\s+section|discussion|conclusion|abstract)\s+(for|about|on)/i,
  /create\s+(my|a|the)\s+(thesis|dissertation|chapter|section|paragraph)/i,
  /compose\s+(my|a|the)\s+(thesis|dissertation|research)/i,
  /draft\s+(my|a|the)\s+(thesis|dissertation|chapter|section|paper)/i,
  /produce\s+(a|the)\s+(complete|full|entire)\s+(section|chapter|paper)/i,
  /give\s+me\s+(a|the)\s+(thesis|dissertation|paper|chapter|section)\s+(text|draft|write-up)/i,
];

const INTEGRITY_RESPONSE = `**FIA does not generate thesis, dissertation, or research paper text.**

Instead, I can guide you through the reasoning process needed to write this section yourself. This approach ensures your work reflects your own understanding and meets academic integrity standards.

Let me switch to **Guide Mode** to help you think through this step by step.

What specific aspect of this section are you finding challenging? For example:
- Structuring your argument
- Identifying key points to include
- Connecting your evidence to your claims
- Understanding what examiners expect in this section`;

function checkIntegrityViolation(message: string): boolean {
  return INTEGRITY_PATTERNS.some((pattern) => pattern.test(message));
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY not configured");

    const authHeader = req.headers.get("Authorization");
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_ANON_KEY")!,
      { global: { headers: { Authorization: authHeader || "" } } }
    );

    // Get user if authenticated
    let userId: string | null = null;
    if (authHeader?.startsWith("Bearer ")) {
      const token = authHeader.replace("Bearer ", "");
      if (token !== Deno.env.get("SUPABASE_ANON_KEY")) {
        const { data } = await supabase.auth.getClaims(token);
        if (data?.claims?.sub) {
          userId = data.claims.sub as string;
        }
      }
    }

    const body = await req.json();
    const {
      mode = "guide",
      messages = [],
      conversation_id,
      context,
      stream = true,
    } = body;

    const systemPrompt = AI_MODES[mode];
    if (!systemPrompt) {
      return new Response(
        JSON.stringify({ error: `Invalid mode: ${mode}. Valid modes: ${Object.keys(AI_MODES).join(", ")}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: "Messages array is required and must not be empty." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Academic integrity check on the latest user message
    const lastUserMsg = [...messages].reverse().find((m: any) => m.role === "user");
    if (lastUserMsg && checkIntegrityViolation(lastUserMsg.content)) {
      // Return integrity response instead of calling Claude
      const encoder = new TextEncoder();
      if (stream) {
        const integrityStream = new ReadableStream({
          start(controller) {
            const chunk = { choices: [{ delta: { content: INTEGRITY_RESPONSE } }] };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });
        return new Response(integrityStream, {
          headers: { ...corsHeaders, "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive", "X-Conversation-Id": conversation_id || "" },
        });
      }
      return new Response(
        JSON.stringify({ response: INTEGRITY_RESPONSE, conversation_id, timestamp: new Date().toISOString() }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const trimmedMessages = messages.slice(-50);

    // Fetch reading log context for authenticated users
    // Uses the user-scoped client — RLS ensures only the user's own rows are returned
    let readingLogContext = "";
    if (userId) {
      try {
        const { data: rlEntries } = await supabase
          .from("reading_log")
          .select("title, year, relevance_note, student_answers")
          .eq("answer_completed", true)
          .order("created_at", { ascending: false })
          .limit(5);

        if (rlEntries && rlEntries.length > 0) {
          const lines = (rlEntries as Array<Record<string, unknown>>).map((e) => {
            const answers = Array.isArray(e.student_answers) ? e.student_answers as string[] : [];
            const q3Connection = ((answers[2] || "") as string)
              .replace(/\n/g, " ")
              .substring(0, 120);
            return `- "${e.title}" (${e.year}) — Why it matters: ${e.relevance_note} — Student's connection to their work: ${q3Connection}`;
          });
          readingLogContext =
            "\n\nStudent reading record (last 5 answered papers):\n" + lines.join("\n");
        }
      } catch {
        // Non-fatal — proceed without reading log context if fetch fails
      }
    }

    let enrichedSystem = systemPrompt;
    if (context) {
      enrichedSystem += `\n\nStudent Context:\n`;
      if (context.track) enrichedSystem += `- Academic Track: ${context.track}\n`;
      if (context.discipline) enrichedSystem += `- Discipline: ${context.discipline}\n`;
      if (context.stage) enrichedSystem += `- Research Stage: ${context.stage}\n`;
      if (context.title) enrichedSystem += `- Research Title: ${context.title}\n`;
      if (context.analysis_type) enrichedSystem += `- Analysis Type: ${context.analysis_type}\n`;
      if (context.data_summary) enrichedSystem += `- Data Summary: ${context.data_summary}\n`;
    }
    if (readingLogContext) {
      enrichedSystem += readingLogContext;
    }

    let convId = conversation_id;
    if (userId) {
      if (!convId) {
        const { data: conv } = await supabase
          .from("ai_conversations")
          .insert({ user_id: userId, mode, title: trimmedMessages[0]?.content?.substring(0, 100), context: context || {} })
          .select("id")
          .single();
        convId = conv?.id;
      }

      if (convId) {
        const lastMsg = trimmedMessages[trimmedMessages.length - 1];
        if (lastMsg?.role === "user") {
          await supabase.from("ai_messages").insert({
            conversation_id: convId,
            role: "user",
            content: lastMsg.content,
            mode,
          });
        }
      }
    }

    const anthropicMessages = trimmedMessages.map((m: { role: string; content: string }) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: m.content,
    }));

    if (stream) {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 4000,
          system: enrichedSystem,
          messages: anthropicMessages,
          stream: true,
        }),
      });

      if (!response.ok) {
        const errText = await response.text();
        console.error("Anthropic error:", response.status, errText);
        if (response.status === 429) {
          return new Response(
            JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
            { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }
        return new Response(
          JSON.stringify({ error: "AI service temporarily unavailable." }),
          { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let fullResponse = "";

      const transformedStream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          let buffer = "";

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, { stream: true });

              let newlineIdx;
              while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
                const line = buffer.slice(0, newlineIdx).trim();
                buffer = buffer.slice(newlineIdx + 1);

                if (!line.startsWith("data: ")) continue;
                const jsonStr = line.slice(6);
                if (jsonStr === "[DONE]") continue;

                try {
                  const event = JSON.parse(jsonStr);
                  if (event.type === "content_block_delta" && event.delta?.text) {
                    fullResponse += event.delta.text;
                    const chunk = {
                      choices: [{ delta: { content: event.delta.text } }],
                    };
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
                  }
                } catch { /* skip */ }
              }
            }
          } catch (e) {
            console.error("Stream error:", e);
          }

          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();

          if (userId && convId && fullResponse) {
            await supabase.from("ai_messages").insert({
              conversation_id: convId,
              role: "assistant",
              content: fullResponse,
              mode,
            });
            await supabase
              .from("ai_conversations")
              .update({ updated_at: new Date().toISOString() })
              .eq("id", convId);
          }
        },
      });

      return new Response(transformedStream, {
        headers: {
          ...corsHeaders,
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "X-Conversation-Id": convId || "",
        },
      });
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 4000,
        system: enrichedSystem,
        messages: anthropicMessages,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Anthropic error:", response.status, errText);
      return new Response(
        JSON.stringify({ error: "AI service temporarily unavailable." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const data = await response.json();
    const responseText = data.content?.[0]?.text || "";

    if (userId && convId && responseText) {
      await supabase.from("ai_messages").insert({
        conversation_id: convId,
        role: "assistant",
        content: responseText,
        mode,
      });
    }

    return new Response(
      JSON.stringify({
        response: responseText,
        conversation_id: convId,
        timestamp: new Date().toISOString(),
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (e) {
    console.error("rws-ai error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
