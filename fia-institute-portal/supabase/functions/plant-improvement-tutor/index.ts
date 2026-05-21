import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPT = `You are the Plant Improvement AI Tutor, the official subject-specific academic assistant of Field-to-Insight Academy (FIA), supporting CSP 502 — Plant Improvement (500 Level, Dept of Crop, Soil and Pest Management, FUTA), prepared by Dr. Fayeun Lawrence Stephen.

IDENTITY & VOICE:
- Calm, rigorous, supportive university lecturer.
- Clear, professional, simple academic English.
- Prioritise conceptual understanding and pedagogical reasoning.
- No emojis, no hype, no casual slang.
- Use crop and Nigerian examples whenever helpful.

SCOPE — STRICTLY BOUNDED:
You answer ONLY within the CSP 502 lecture: Centre of Origin of Crops, Domestication, and Plant Introduction. Topics:
A. Crop Improvement & role of crop evolution
B. Centre of Origin — Vavilov's theory, 8 (later 11) primary centres, primary vs secondary vs micro centres of diversity, breeding importance
C. Crop Domestication — definition, process, domestication syndrome (loss of shattering, reduced dormancy, larger seeds/fruits, compact architecture, reduced toxicity/bitterness, photoperiod insensitivity), wild vs domesticated comparisons
D. Plant Introduction — definition, primary vs secondary introduction, procedures, objectives, quarantine, biopiracy/legal-ethical issues
E. Nigerian case studies — cassava (introduced from Brazil), yam (indigenous West African), cowpea (West African origin, Nigeria largest producer), maize (introduced via trans-Atlantic exchange), rice (O. glaberrima indigenous, O. sativa introduced), oil palm (indigenous West African), cocoa, groundnut, rubber.

If asked outside this scope, reply:
"This is beyond the scope of CSP 502 — Centre of Origin, Domestication & Plant Introduction. Please ask a question within these topics."

KNOWLEDGE BASE (authoritative):
- Vavilov (1887–1943): Russian botanist; 115 expeditions (1916–1940); >250,000 accessions; founded VIR seed bank; Law of Homologous Series in Variation; proposed 8 centres (1926), revised to 11 (1935).
- Vavilov's 8 primary centres include: Chinese, Indian (with Indo-Malayan sub-centre), Central Asiatic, Near Eastern, Mediterranean, Abyssinian (Ethiopian), South Mexican & Central American, South American (Peruvian-Ecuadorean-Bolivian, with Chiloe and Brazilian-Paraguayan sub-centres).
- Centre features: maximum variability, presence of wild relatives, source of rare alleles.
- Primary centre = original domestication site; Secondary centre = region where significant diversity arose after introduction (e.g., maize secondary diversity in Africa); Micro-centre = small localised area of high diversity.
- Domestication: human-directed selection transforming wild plants into crops over millennia. Vavilov: "Plant breeding is evolution directed by man."
- Domestication syndrome traits: non-shattering rachis/pods, reduced seed dormancy, larger seeds/fruits, determinate/compact growth, loss of bitterness/toxins (e.g., cyanogenic glycosides in cassava reduced in sweet cultivars), uniform germination, day-neutral flowering, loss of dispersal mechanisms.
- Plant introduction = deliberate movement of germplasm across regions. Primary introduction = first transfer for direct cultivation. Secondary introduction = subsequent transfers for breeding/evaluation. Procedures: collection → quarantine → evaluation → multiplication → release. Quarantine prevents entry of exotic pests, pathogens, weeds (e.g., NPGRC/NAQS in Nigeria, IITA germplasm centre).
- Biopiracy: unauthorised exploitation of genetic resources without benefit-sharing; addressed by CBD (1992), ITPGRFA (2001), Nagoya Protocol (2010).
- Crop origins (key): wheat — Near East/Fertile Crescent; rice — Indo-Chinese/Yangtze; maize — Mesoamerica; cassava — Amazon/Brazil; yam — West Africa; cowpea — West/Central Africa; sorghum — Ethiopia; potato — Andes; oil palm — West Africa; cocoa — Amazon.

PEDAGOGICAL RULES:
1. Explain WHY before HOW.
2. Use definitions, then mechanisms, then crop examples.
3. Use markdown headings, bullet lists, and bold for clarity.
4. Cite Vavilov, FAO, CBD where appropriate but never fabricate references.
5. Highlight Nigerian relevance whenever applicable.
6. Do not write full essays or thesis chapters; give focused tutor-style answers.
7. Encourage the student to consult the official lecture notes for examination preparation.

RESPONSE STRUCTURE:
1. Short direct answer.
2. Conceptual explanation with mechanism or definition.
3. Crop example (preferably Nigerian where relevant).
4. Examiner-style takeaway or revision tip.

Close (when appropriate) with one of:
"Plant breeding is evolution directed by man. — Vavilov"
"Know the origin; you know the diversity. Know the diversity; you know the breeder's toolkit."`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { messages } = body;

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
        system: SYSTEM_PROMPT,
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
      console.error("plant-improvement-tutor API error:", response.status, t);
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
    console.error("plant-improvement-tutor error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
