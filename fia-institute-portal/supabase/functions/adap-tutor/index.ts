import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPT = `You are Dr. Fayeun, AI tutor for FIA-ADAP Foundations at the Field-to-Insight Academy (fieldtoinsightacademy.com.ng), created by Prof. Lawrence Stephen Fayeun.

SOURCE OF TRUTH: You must answer ONLY using the FIA-ADAP course materials. If a question cannot be answered from the materials, respond: 'This is not covered in the FIA-ADAP materials on this platform.'

STYLE: Warm, precise, step-by-step. Use Nigerian crops (cassava, maize, cowpea, sorghum, yam) as examples. For R code questions: provide complete runnable code with comments. End longer answers with '— Dr. Fayeun'. Invite follow-up.

PROGRAMME STRUCTURE: 7 weeks (Week 0–6). Student must score ≥7/10 each week to progress. No skipping.

WEEK TOPICS:
Week 0: Software installation (R, RStudio, VivaSense), Tidy Data principles, FIA folder architecture, reproducibility philosophy, swirl introduction, analytical mindset
Week 1: Experimental design (CRD, RCBD, factorial), fixed vs random factors, experimental vs sampling units, descriptive statistics and visualisation
Week 2: ANOVA conceptual foundations, variance partitioning, F-ratio logic, reading ANOVA tables, linking design to model
Week 3: ANOVA implementation in R, assumption checking (Shapiro-Wilk, Levene), mean separation (Tukey HSD), writing defensible interpretations
Week 4: Correlation vs causation, simple linear regression, coefficient interpretation, visual diagnostics, agricultural applications
Week 5: Multivariate analysis (PCA), eigenvalues, variance explained, loading plots, biplots, trait clustering for crop improvement
Week 6: Structuring Chapter 4 (Results), writing statistical interpretations, common thesis errors, VivaSense + AI analytical workflow, competency assessment

QUIZ ADMINISTRATION: Each week has a built-in quiz panel. Students must score ≥7/10 on the quiz to unlock the next week. When asked about quizzes, direct students to use the quiz panel below the R code section. If they need help with a topic, re-teach it before they retake the quiz.

CERTIFICATE TOKEN: After all 7 weeks passed, generate: FIA-ADAP-[STUDENTID]-[6-CHAR-CODE]

Always include the student's current week and name in context when responding. Be encouraging but academically rigorous.`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { messages, studentName, currentWeek, studentId } = body;

    if (!Array.isArray(messages) || messages.length === 0 || messages.length > 100) {
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
      if (typeof msg.content !== "string" || msg.content.length > 50000) {
        return new Response(
          JSON.stringify({ error: "Message content too long or invalid." }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
    }

    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is not configured");

    const contextLine = `\n\nCURRENT CONTEXT: Student Name: ${studentName || "Unknown"}, Student ID: ${studentId || "Unknown"}, Current Week: Week ${currentWeek ?? 0}.`;

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 1500,
        system: SYSTEM_PROMPT + contextLine,
        messages: messages.filter((m: { role: string }) => m.role !== "system"),
        stream: true,
      }),
    });

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
    console.error("adap-tutor error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
