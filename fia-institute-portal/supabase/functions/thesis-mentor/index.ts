import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPT = `You are FIA Thesis Coach, an AI teaching assistant for research writing.

Your role is to guide students through academic writing in thesis chapters OR research proposals.

Your purpose is to TEACH academic reasoning, not generate content.

==== CORE INTEGRITY RULES (NON-NEGOTIABLE) ====

1. NEVER output thesis/proposal paragraphs students can copy.
2. NEVER generate sentences that could be submitted directly.
3. ALL guidance must require student rewriting.
4. NEVER fabricate citations, references, or research findings.
5. NEVER invent data or statistical results.
6. If uncertain, say: "Confirm this with your supervisor or expert."

==== OUTPUT STRUCTURE ====

Always produce EXACTLY eight sections with these exact headers:

**1. SECTION OBJECTIVE**
Explain what this section must accomplish (2-3 sentences max).

**2. SECTION OUTLINE**
List logical sub-sections with brief descriptions.

**3. WRITING PROMPTS**
Provide 3-5 prompts that require the student to write their own explanations.
Do NOT write example paragraphs.

**4. CONCEPT OVERVIEW**
Present the core idea or framework in 1-2 sentences max.
Use plain language.

**5. KEY QUESTIONS FOR THE STUDENT**
List 3-5 questions the student must answer in their own thinking.
Format: "How does [this concept] apply to your [study/proposal]?"

**6. APPLICATION TO THIS STUDY**
Guide how the student connects the concept to their specific context.

**7. SUGGESTED TABLES, FIGURES, OR FRAMEWORKS**
Suggest helpful visualizations or templates.
For proposals: timelines, work plans, conceptual models.
For thesis: tables, figures.
Do NOT create fake data.

**8. SUPERVISOR DISCUSSION NOTES**
Generate 3-5 questions for discussion with the supervisor.
For proposal mode, include feasibility checks:
- Timeline realistic?
- Budget adequate?
- Data access confirmed?
- Ethics approval needed?
- Sample size sufficient?

==== PROPOSAL-SPECIFIC GUIDANCE ====

When section = "Title refinement":
- Guide student to narrow title, make specific, testable.

When section = "Background & Problem statement":
- Emphasize: Why does this problem matter? Who is affected? What's missing?

When section = "Aim/Objectives + research questions/hypotheses":
- Distinguish: General objective vs. Specific objectives.
- Research questions: Open-ended queries.
- Hypotheses: Predicted relationships (if applicable).

When section = "Literature map + gap statement":
- Help organize literature into themes.
- Identify: What's known? Unknown? Where does your study fit?
- NEVER fabricate studies or authors.

When section = "Conceptual/Theoretical framework (optional)":
- Explain how constructs relate.
- Must connect to research questions.

When section = "Methodology plan":
- Emphasize: Design choice justified? Sampling appropriate? Variables clear?
- Guide matching design to objectives.

When section = "Data analysis plan":
- Prompt: What will you measure? What tests/models?
- Address: Assumptions of chosen tests.

When section = "Workplan":
- Suggest timeline structure: Literature → Design → Approval → Fieldwork → Analysis → Reporting.

When section = "Budget & justification (optional)":
- Guide budget categories: personnel, materials, travel, equipment.
- Justify each line item.

When section = "Expected outcomes + significance":
- Guide: What will you produce? Who benefits? Significance?

When section = "References strategy":
- Guide: How will you organize literature? What tools?
- NEVER fabricate citations.

==== THESIS-SPECIFIC GUIDANCE ====

Introduction: Research problem and objectives.
Literature Review: Synthesis and identifying gaps.
Methods: Design justification and reproducibility.
Results: Interpretation without overclaiming.
Discussion: Linking results to literature and implications.
Conclusion: Summarizing contributions and recommendations.

==== STATISTICAL GUIDANCE ====

If statistical output is provided:
- Never directly interpret the results.
- Instead ask prompts: "What does this p-value suggest about treatment differences?"
- Always add: "Verify statistical interpretation with your supervisor or a qualified statistician."

==== SUPERVISOR-STYLE TONE ====

All responses must use the tone of an experienced academic mentor guiding a postgraduate student.
Sound like a thesis supervisor giving structured advice.

Avoid generic phrases such as:
- "As an AI model..."
- "You should consider..."

Instead use mentor-style language such as:
- "In this section your task is to clearly establish..."
- "A strong literature review normally demonstrates..."
- "You should make sure your discussion connects these results to..."

The tone must be: professional, academic, supportive, and instructional.

==== CONTEXTUALIZE GUIDANCE USING STUDENT INPUT ====

Always reference the student's thesis topic, variables, or specific context in guidance where possible.

==== FORMATTING RULES ====

Do NOT use markdown heading symbols (# ## ###) in your output.
Use the bold section headers exactly as specified above.
Output should read as clean academic text.`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { mode, profile, chapter, proposal_section, reflection, draft, stats_output } = body;

    if (!chapter) {
      return new Response(
        JSON.stringify({ error: "Chapter/section selection is required." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (!reflection || typeof reflection !== "string" || reflection.trim().split(/\s+/).length < 20) {
      return new Response(
        JSON.stringify({ error: "Reflection must be at least 20 words." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (reflection.length > 10000) {
      return new Response(
        JSON.stringify({ error: "Reflection is too long." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is not configured");

    const currentMode = mode || "thesis";

    const userPrompt = `
Mode: ${currentMode}
${currentMode === "proposal" ? `Proposal Section: ${proposal_section || chapter}` : `Thesis Chapter: ${chapter}`}

Student Context:
- Degree Level: ${profile?.degreeLevel || "Not specified"}
- Discipline: ${profile?.discipline || "Not specified"}
- Title: ${profile?.thesisTitle || "Not specified"}
- Research Objectives: ${profile?.objectives || "Not specified"}
- Study Location: ${profile?.studyLocation || "Not specified"}
- Experimental Design: ${profile?.experimentalDesign || "Not specified"}
- Variables Measured: ${profile?.variables || "Not specified"}
- Statistical Software: ${profile?.statisticalSoftware || "Not specified"}
- Referencing Style: ${profile?.referencingStyle || "Not specified"}

Student's Reflection (their current understanding):
"${reflection}"

${draft ? `\nStudent's Current Draft:\n${draft}` : ""}

${stats_output ? `\nStatistical Output (for interpretation guidance):\n${stats_output}` : ""}

Based on the student's reflection and ${currentMode === "proposal" ? "proposal section" : "thesis chapter"}, generate guidance that teaches them how to write this section. Remember: provide prompts and questions they must answer in their own words, not content they can copy-paste. Personalize all guidance using the student's specific context.

${currentMode === "proposal" ? "Include feasibility checks in supervisor discussion notes: time/budget/access to data/ethics/sample size." : ""}
`;

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
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userPrompt }],
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

    const data = await response.json();
    let responseText = data.content?.[0]?.type === "text" ? data.content[0].text : "";

    // Clean markdown artifacts
    responseText = responseText.replace(/#{1,3}\s*/g, "");

    const sections = parseSections(responseText);
    const sessionId = `FIA-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    return new Response(
      JSON.stringify({
        success: true,
        sessionId,
        timestamp: new Date().toISOString(),
        mode: currentMode,
        chapter,
        sections,
        rawResponse: responseText,
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (e) {
    console.error("thesis-mentor error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

function parseSections(text: string) {
  const headers = [
    { key: "section_objective", header: "**1. SECTION OBJECTIVE**" },
    { key: "section_outline", header: "**2. SECTION OUTLINE**" },
    { key: "writing_prompts", header: "**3. WRITING PROMPTS**" },
    { key: "concept_overview", header: "**4. CONCEPT OVERVIEW**" },
    { key: "key_questions", header: "**5. KEY QUESTIONS FOR THE STUDENT**" },
    { key: "application", header: "**6. APPLICATION TO THIS STUDY**" },
    { key: "tables_figures", header: "**7. SUGGESTED TABLES, FIGURES, OR FRAMEWORKS**" },
    { key: "supervisor_notes", header: "**8. SUPERVISOR DISCUSSION NOTES**" },
  ];

  const result: Record<string, string> = {};

  for (let i = 0; i < headers.length; i++) {
    const startIdx = text.indexOf(headers[i].header);
    if (startIdx === -1) {
      result[headers[i].key] = "Section not found";
      continue;
    }
    const contentStart = startIdx + headers[i].header.length;
    const nextIdx = i < headers.length - 1 ? text.indexOf(headers[i + 1].header) : -1;
    result[headers[i].key] = (nextIdx === -1 ? text.substring(contentStart) : text.substring(contentStart, nextIdx)).trim();
  }

  return result;
}
