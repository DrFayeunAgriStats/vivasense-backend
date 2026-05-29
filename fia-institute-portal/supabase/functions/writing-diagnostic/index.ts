import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

// ── Prompts ────────────────────────────────────────────────────────────────────

function buildQuestionsSystemPrompt(): string {
  return `You are a postgraduate research writing diagnostic assessor for the Field-to-Insight Academy. A student has submitted a writing sample. Your task is NOT to grade it — it is to ask five probing questions that reveal the student's actual thinking level.

Rules:
- Ask exactly five questions, numbered 1–5
- Each question must be answerable only if the student genuinely understands what they wrote
- Questions must progress from surface (Q1) to deep (Q5)
- Q1–Q2: test whether the student can explain their own argument in plain language
- Q3–Q4: test whether the student understands the evidence behind their claims
- Q5: test whether the student can identify the weakness or limitation in their own writing
- Never ask yes/no questions
- Never ask questions answerable by googling
- Do not praise the writing
- Do not suggest improvements yet
- Address the student as 'you' not by name
- End with exactly this line: Answer each question below. Take your time.`;
}

function buildAssessSystemPrompt(): string {
  return `You are assessing a postgraduate research writing student for the Field-to-Insight Academy. Based on their writing sample and their answers to five diagnostic questions, assign them to exactly one of three levels.

Level definitions:
- FOUNDATION: Student cannot yet explain their own argument. Answers are vague, circular, or restate the original text. Confuses description with analysis.
- DEVELOPING: Student understands their argument at surface level but cannot justify evidence choices or identify limitations. Writing shows structure but lacks critical reasoning.
- ADVANCED: Student can explain their argument, justify evidence, and identify limitations unprompted. Shows genuine critical thinking beyond the text.

Return a JSON object only — no other text:
{
  "level": "FOUNDATION" | "DEVELOPING" | "ADVANCED",
  "level_rationale": "2–3 sentences explaining the assignment for the student to read",
  "three_strengths": ["string", "string", "string"],
  "three_gaps": ["string", "string", "string"],
  "first_priority": "One sentence: the single most important thing to work on first"
}`;
}

// ── Question parser ────────────────────────────────────────────────────────────

function parseQuestions(text: string): string[] {
  const questions: string[] = [];
  const lines = text.split("\n");
  let current = "";

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.toLowerCase().startsWith("answer each question")) {
      if (current) questions.push(current.trim());
      break;
    }
    const numbered = line.match(/^([1-5])[.)]\s+(.+)/);
    if (numbered) {
      if (current) questions.push(current.trim());
      current = numbered[2];
    } else if (current) {
      current += " " + line;
    }
  }
  if (current && questions.length < 5) questions.push(current.trim());

  return questions.slice(0, 5);
}

// ── Assessment fallback ────────────────────────────────────────────────────────

const FALLBACK_ASSESSMENT = {
  level: "DEVELOPING",
  level_rationale:
    "Your writing sample showed a developing level of research thinking. You are building a foundation in academic argumentation. Working through the FIA Research Writing Mentor modules will help you strengthen your critical reasoning and evidence use.",
  three_strengths: [
    "You engaged fully with the diagnostic process",
    "You submitted a substantive writing sample",
    "You are actively seeking to develop your research writing skills",
  ],
  three_gaps: [
    "Continue developing how you explain your argument in plain language",
    "Work on connecting your claims to specific evidence",
    "Practise identifying the limitations of your own writing before submitting",
  ],
  first_priority:
    "Begin with Guide Mode in the AI Research Writing Mentor to work through your current chapter and practise explaining your argument step by step.",
};

// ── Main handler ──────────────────────────────────────────────────────────────

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) {
      return new Response(
        JSON.stringify({ error: "AI service not configured on this deployment." }),
        { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Authenticate user
    const authHeader = req.headers.get("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
      return new Response(
        JSON.stringify({ error: "Unauthorized" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const supabaseUser = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_ANON_KEY")!,
      { global: { headers: { Authorization: authHeader } } }
    );

    const { data: { user }, error: authErr } = await supabaseUser.auth.getUser();
    if (authErr || !user) {
      return new Response(
        JSON.stringify({ error: "Unauthorized" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const body = await req.json() as Record<string, unknown>;
    const { stage } = body;

    // ── STAGE 1: Generate questions ──────────────────────────────────────────
    if (stage === "questions") {
      const writingSample = (body.writing_sample as string || "").trim();
      const track         = (body.track         as string || "").replace(/_/g, " ");
      const discipline    = (body.discipline    as string || "Not specified");
      const researchStage = (body.research_stage as string || "").replace(/_/g, " ");

      if (!writingSample || writingSample.split(/\s+/).length < 100) {
        return new Response(
          JSON.stringify({ error: "Writing sample is too short." }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const userMessage =
        `Student track: ${track}\n` +
        `Student discipline: ${discipline}\n` +
        `Research stage: ${researchStage}\n` +
        `Writing sample:\n\n${writingSample}`;

      const claudeRes = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 800,
          system: buildQuestionsSystemPrompt(),
          messages: [{ role: "user", content: userMessage }],
        }),
      });

      if (!claudeRes.ok) {
        const errText = await claudeRes.text();
        console.error("Anthropic error (questions):", claudeRes.status, errText);
        return new Response(
          JSON.stringify({ error: "AI service temporarily unavailable." }),
          { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const claudeData = await claudeRes.json() as Record<string, unknown>;
      const rawText = ((claudeData.content as Array<{text: string}>)?.[0]?.text || "").trim();
      const questions = parseQuestions(rawText);

      if (questions.length < 5) {
        console.error("Question parsing yielded fewer than 5:", questions);
        return new Response(
          JSON.stringify({ error: "Could not generate 5 questions. Please try again." }),
          { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      return new Response(
        JSON.stringify({ questions }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // ── STAGE 2: Assess answers and assign level ─────────────────────────────
    if (stage === "assess") {
      const writingSample = (body.writing_sample as string || "").trim();
      const questions     = (body.questions as string[]) || [];
      const answers       = (body.answers   as string[]) || [];

      if (questions.length !== 5 || answers.length !== 5) {
        return new Response(
          JSON.stringify({ error: "Expected 5 questions and 5 answers." }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const qaLines = questions
        .map((q, i) => `Q${i + 1}: ${q}\nAnswer: ${answers[i] || "(no answer)"}`)
        .join("\n\n");

      const userMessage =
        `Writing sample:\n\n${writingSample}\n\n` +
        `${qaLines}`;

      const claudeRes = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 600,
          system: buildAssessSystemPrompt(),
          messages: [{ role: "user", content: userMessage }],
        }),
      });

      if (!claudeRes.ok) {
        console.error("Anthropic error (assess):", claudeRes.status);
        return new Response(
          JSON.stringify(FALLBACK_ASSESSMENT),
          { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const claudeData = await claudeRes.json() as Record<string, unknown>;
      const rawText = ((claudeData.content as Array<{text: string}>)?.[0]?.text || "").trim();

      let assessment: Record<string, unknown>;
      try {
        // Strip any markdown code fences Claude might add
        const cleaned = rawText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "");
        assessment = JSON.parse(cleaned);

        // Validate required fields
        if (!["FOUNDATION", "DEVELOPING", "ADVANCED"].includes(assessment.level as string)) {
          throw new Error("Invalid level value");
        }
      } catch (e) {
        console.error("JSON parse failed — using fallback:", e, "Raw:", rawText);
        assessment = FALLBACK_ASSESSMENT;
      }

      return new Response(
        JSON.stringify(assessment),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({ error: `Unknown stage: ${stage}` }),
      { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (e) {
    console.error("writing-diagnostic error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
