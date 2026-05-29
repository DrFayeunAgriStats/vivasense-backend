import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

const STAGE_LABELS: Record<string, string> = {
  topic_proposal: "Topic / Proposal",
  literature_review: "Literature Review",
  methodology: "Methodology",
  data_analysis: "Data Analysis",
  results_writing: "Results Writing",
  discussion: "Discussion",
  defense_preparation: "Defense Preparation",
};

const MODE_LABELS: Record<string, string> = {
  guide: "Socratic Guide",
  explain: "Concept Explanation",
  review: "Writing Review",
  upgrade: "Structure Upgrade",
  supervisor: "Supervisor Lens",
  defense: "Defense Examiner",
  interpret: "Data Interpretation",
};

const SYSTEM_PROMPT = `You are generating a Supervisor Briefing Note for a Nigerian postgraduate supervisor. Based on the session transcript provided, write a half-page plain-English briefing covering exactly five points in flowing prose:

1. What the student worked on in this session — be specific about the topic or chapter
2. Where the student demonstrated good understanding or made progress
3. Where the student struggled, showed gaps, or needed repeated guidance
4. One specific, focused question the supervisor should ask at their next meeting — make it directly connected to what was discussed
5. One concrete action the student should complete before that meeting — be specific, not generic

Rules:
- Write in professional academic tone appropriate for a Nigerian university supervisor
- Be specific to the actual content discussed — never generic or templated
- Do not reproduce any part of the AI conversation verbatim
- Write as a knowledgeable colleague who observed the student working and is now briefing the supervisor
- Use flowing paragraphs only — no bullet points or numbered lists in the output
- Length: 200–260 words
- Do not include a heading or title — begin directly with the briefing content`;

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

    // Authenticate the requesting user
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

    const { data: { user }, error: authError } = await supabaseUser.auth.getUser();
    if (authError || !user) {
      return new Response(
        JSON.stringify({ error: "Unauthorized" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Service client to bypass RLS for reading messages + profile
    const supabaseService = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );

    const body = await req.json();
    const { conversation_id } = body as { conversation_id: string };

    if (!conversation_id) {
      return new Response(
        JSON.stringify({ error: "conversation_id is required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Verify the conversation belongs to this user
    const { data: conversation, error: convError } = await supabaseService
      .from("ai_conversations")
      .select("id, mode, user_id")
      .eq("id", conversation_id)
      .single();

    if (convError || !conversation || conversation.user_id !== user.id) {
      return new Response(
        JSON.stringify({ error: "Conversation not found or access denied" }),
        { status: 403, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Fetch messages — cap at 40 (20 exchanges) to keep context manageable
    const { data: messages, error: messagesError } = await supabaseService
      .from("ai_messages")
      .select("role, content")
      .eq("conversation_id", conversation_id)
      .order("created_at", { ascending: true })
      .limit(40);

    if (messagesError || !messages || messages.length < 4) {
      return new Response(
        JSON.stringify({ error: "Session too short to generate a meaningful briefing. Complete at least 3 exchanges first." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Fetch student profile
    const { data: profile } = await supabaseService
      .from("profiles")
      .select("full_name, academic_track, discipline, current_research_stage, thesis_title")
      .eq("id", user.id)
      .single();

    const studentName = (profile as Record<string, unknown>)?.full_name as string || "The student";
    const track = ((profile as Record<string, unknown>)?.academic_track as string || "").replace(/_/g, " ");
    const discipline = (profile as Record<string, unknown>)?.discipline as string || "Not specified";
    const stage = (profile as Record<string, unknown>)?.current_research_stage as string || "";
    const stageLabel = STAGE_LABELS[stage] || stage.replace(/_/g, " ") || "Not specified";
    const thesisTitle = (profile as Record<string, unknown>)?.thesis_title as string || "Not specified";
    const modeLabel = MODE_LABELS[conversation.mode] || conversation.mode || "General";

    // Count user-side exchanges
    const userMsgCount = (messages as Array<{role: string; content: string}>)
      .filter((m) => m.role === "user").length;

    // Build a clean transcript (not verbatim — labelled for context)
    const transcript = (messages as Array<{role: string; content: string}>)
      .map((m) => `[${m.role === "user" ? "Student" : "AI Mentor"}]: ${m.content}`)
      .join("\n\n");

    const today = new Date().toLocaleDateString("en-GB", {
      day: "numeric",
      month: "long",
      year: "numeric",
    });

    const userMessage = `Student: ${studentName}
Academic Track: ${track}
Discipline: ${discipline}
Current Research Stage: ${stageLabel}
Thesis / Project Title: ${thesisTitle}
AI Session Mode: ${modeLabel}
Session Date: ${today}
Number of student exchanges: ${userMsgCount}

SESSION TRANSCRIPT:
${transcript}

Generate the supervisor briefing note now.`;

    // Call Claude Haiku
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
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userMessage }],
      }),
    });

    if (!claudeRes.ok) {
      const errText = await claudeRes.text();
      console.error("Anthropic error:", claudeRes.status, errText);
      if (claudeRes.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit reached. Please try again in a moment." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      return new Response(
        JSON.stringify({ error: "AI service temporarily unavailable." }),
        { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const claudeData = await claudeRes.json();
    const briefingText = (claudeData.content?.[0]?.text as string || "").trim();

    if (!briefingText) {
      return new Response(
        JSON.stringify({ error: "Briefing generation returned empty content." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Return briefing to frontend — storage is user-triggered from the modal
    return new Response(
      JSON.stringify({
        briefing_text: briefingText,
        conversation_id,
        mode: conversation.mode,
        stage,
        topic: (messages as Array<{role: string; content: string}>)
          .find((m) => m.role === "user")?.content?.substring(0, 80) || "",
        exchange_count: userMsgCount,
        student_name: studentName,
        generated_at: new Date().toISOString(),
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (e) {
    console.error("generate-briefing error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
