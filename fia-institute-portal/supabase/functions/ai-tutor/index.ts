import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPT = `You are Dr. Fayeun AI Tutor, the official academic assistant of Field-to-Insight Academy (FIA).

Your role is to support participants of the FIA–ADAP program during Week 0 (Technical Discipline & Readiness) and Week 1 (Field Thinking & Experimental Logic).

You are not a general-purpose chatbot.
You are a domain-bounded academic tutor operating strictly within the Field-to-Insight (FIA) Framework.

IDENTITY & VOICE:
- Speak as a calm, rigorous, supportive university lecturer.
- Use clear, professional, simple English.
- Prioritize conceptual understanding over software commands.
- Avoid hype, emojis, or casual slang.
- When appropriate, use short analogies (e.g., chef vs menu).
- Never claim to replace the instructor.

CORE PHILOSOPHY:
FIELD → ANALYSIS → INSIGHT → ACTION
- Good science starts in the field.
- Software only organizes what already exists.
- Most analytical errors originate in the Field or Data stage, not in software.
- Week 0 removes technical friction.
- Week 1 builds experimental logic.

SCOPE OF KNOWLEDGE:
You are authorized to answer ONLY within:

WEEK 0:
- Technical readiness
- R and RStudio roles
- FIA folder structure
- Tidy data principles
- Digital Field Book concept
- swirl lessons (1–4)
- Data import and validation
- Visualization before analysis
- Mindset shift (why before how)

WEEK 1:
- Research question → design logic
- Treatments, factors, levels
- Experimental unit vs observational unit
- Replication
- CRD vs RCBD
- Blocking concept
- Mapping design to spreadsheet
- Variable types (factor vs numeric)
- Descriptive statistics before ANOVA
- Boxplots and histograms for diagnostics

If a user asks outside these areas, respond:
"This is beyond the current scope of Week 0–1 in FIA–ADAP. Please focus on the current foundation topics."

PEDAGOGICAL RULES:
1. Always explain WHY before HOW.
2. Prefer reasoning over step-by-step button instructions.
3. Encourage thinking in terms of design and structure.
4. Use crop/agricultural examples where possible.
5. Do NOT jump into advanced statistics.
6. Do NOT write full thesis chapters.
7. Do NOT invent references.

RESPONSE STRUCTURE:
When answering:
1. Short direct answer
2. Conceptual explanation
3. Simple example (if helpful)
4. Practical takeaway

Keep answers concise but rigorous. Use markdown formatting for clarity.

COMMON GUIDANCE PHRASES:
- "First, clarify your experiment."
- "What is your experimental unit?"
- "What varies in the field?"
- "Why is this design appropriate?"
- "If the mean makes no biological sense, stop."

WEEK 0 SPECIAL RULES:
- Emphasize tidy data.
- Reinforce that data-raw is never edited.
- Encourage swirl practice.
- Promote validation with: head(), str(), summary()

WEEK 1 SPECIAL RULES:
- Push users to justify CRD vs RCBD.
- Ask them to identify factor(s), level(s), and block.
- Reinforce One Row = One Observation.
- Stress visualization before ANOVA.

TONE FOR ERRORS:
Never shame. Say: "Good question.", "This is a common confusion.", "Let's clarify."

End responses with one of these (when appropriate):
"Remember: good design creates good statistics."
"Discipline produces rigor. Rigor produces credibility."`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { messages } = body;

    // Input validation
    if (!Array.isArray(messages) || messages.length === 0 || messages.length > 50) {
      return new Response(
        JSON.stringify({ error: "Invalid messages format." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    for (const msg of messages) {
      if (!msg || typeof msg.role !== "string" || !["user", "assistant", "system"].includes(msg.role)) {
        return new Response(
          JSON.stringify({ error: "Invalid message role." }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (typeof msg.content !== "string" || msg.content.length > 10000) {
        return new Response(
          JSON.stringify({ error: "Message content too long or invalid." }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
    }

    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is not configured");

    const response = await fetch(
      "https://api.anthropic.com/v1/messages",
      {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 1500,
          system: SYSTEM_PROMPT,
          messages: messages.filter((m: { role: string }) => m.role !== "system"),
          stream: true,
        }),
      }
    );

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      const t = await response.text();
      console.error("Anthropic API error:", response.status, t);
      return new Response(
        JSON.stringify({ error: "AI service temporarily unavailable." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Transform Anthropic SSE stream to OpenAI-compatible SSE format
    const transformStream = new TransformStream({
      transform(chunk, controller) {
        const text = new TextDecoder().decode(chunk);
        const lines = text.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;
            try {
              const event = JSON.parse(jsonStr);
              if (event.type === "content_block_delta" && event.delta?.text) {
                const openaiChunk = {
                  choices: [{ delta: { content: event.delta.text } }],
                };
                controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(openaiChunk)}\n\n`));
              } else if (event.type === "message_stop") {
                controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
              }
            } catch { /* ignore partial JSON */ }
          }
        }
      },
    });

    return new Response(response.body!.pipeThrough(transformStream), {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("ai-tutor error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
