import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
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

const TRACK_LABELS: Record<string, string> = {
  undergraduate_project: "Undergraduate Final-Year Project",
  msc_thesis:            "MSc Thesis Development",
  phd_research:          "PhD Research Writing and Defense",
  research_paper:        "Research Paper Writing",
};

// Fallback when Claude fails or response cannot be parsed
const FALLBACK_QUESTIONS = [
  "What is the central argument or main finding of this paper, and what specific evidence does the author use to support it?",
  "What are the key methodological choices in this study, and what limitations do they introduce for interpreting the results?",
  "How does the specific contribution of this paper address a gap or question that is directly relevant to your own research?",
];

function buildSystemPrompt(
  track: string,
  discipline: string,
  thesisTitle: string,
  stage: string,
  title: string,
  authors: string,
  journal: string,
  year: string | number,
  relevanceNote: string,
): string {
  return `You are a postgraduate research supervisor at a Nigerian agricultural university. A student has logged a paper they have read. Your task is to ask exactly three questions that reveal whether they genuinely understood the paper and can connect it to their own research.

Rules:
- Ask exactly 3 questions, numbered 1–3
- Q1: Test whether the student understood the paper's main argument or finding (not just its topic)
- Q2: Test whether the student can evaluate the methodology or evidence quality
- Q3: Test whether the student can connect this paper specifically to their own research problem or gap
- Never ask yes/no questions
- Never ask questions answerable from the abstract alone
- Do not ask for a summary
- Questions must be specific to THIS paper, not generic
- End with exactly this line: Answer these when you are ready — there is no time limit.

Student research context:
Track: ${track}
Discipline: ${discipline}
Thesis title: ${thesisTitle}
Research stage: ${stage}

Paper details:
Title: ${title}
Authors: ${authors}
Journal: ${journal}
Year: ${year}
Student's relevance note: ${relevanceNote}`;
}

function parseQuestions(text: string): string[] {
  const questions: string[] = [];
  const lines = text.split("\n");
  let current = "";

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (/answer these when you are ready/i.test(line)) {
      if (current) questions.push(current.trim());
      break;
    }
    const numbered = line.match(/^([1-3])[.)]\s+(.+)/);
    if (numbered) {
      if (current) questions.push(current.trim());
      current = numbered[2];
    } else if (current) {
      current += " " + line;
    }
  }
  if (current && questions.length < 3) questions.push(current.trim());

  // Pad with fallbacks if parsing yielded fewer than 3
  while (questions.length < 3) {
    questions.push(FALLBACK_QUESTIONS[questions.length]);
  }

  return questions.slice(0, 3);
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) {
      return new Response(
        JSON.stringify({ questions: FALLBACK_QUESTIONS }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
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
    const {
      title        = "",
      authors      = "",
      journal      = "",
      year         = new Date().getFullYear(),
      relevance_note = "",
      track        = "",
      discipline   = "",
      thesis_title = "not yet set",
      stage        = "",
    } = body as Record<string, string | number>;

    if (!title || !authors || !journal || !relevance_note) {
      return new Response(
        JSON.stringify({ questions: FALLBACK_QUESTIONS }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const trackLabel  = TRACK_LABELS[track as string]   || (track  as string).replace(/_/g, " ") || "Not specified";
    const stageLabel  = STAGE_LABELS[stage as string]   || (stage  as string).replace(/_/g, " ") || "Not specified";

    const systemPrompt = buildSystemPrompt(
      trackLabel,
      discipline as string || "Not specified",
      thesis_title as string || "not yet set",
      stageLabel,
      title as string,
      authors as string,
      journal as string,
      year,
      relevance_note as string,
    );

    let questions = FALLBACK_QUESTIONS;

    try {
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
          system: systemPrompt,
          messages: [{ role: "user", content: "Generate the three diagnostic questions for this paper now." }],
        }),
      });

      if (claudeRes.ok) {
        const data = await claudeRes.json() as Record<string, unknown>;
        const rawText = ((data.content as Array<{ text: string }>)?.[0]?.text || "").trim();
        if (rawText) {
          questions = parseQuestions(rawText);
        }
      } else {
        console.error("Anthropic error:", claudeRes.status);
        // Silently use fallback — never fail the paper-add flow
      }
    } catch (e) {
      console.error("Claude call failed, using fallback:", e);
      // Silently use fallback
    }

    return new Response(
      JSON.stringify({ questions }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (e) {
    console.error("reading-log-questions error:", e);
    // Return fallback even on unexpected error — never block the user
    return new Response(
      JSON.stringify({ questions: FALLBACK_QUESTIONS }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
