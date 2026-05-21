import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

// ─── Instructor-controlled completed modules ───
// Update this array to unlock modules for students.
const COMPLETED_MODULES = [1, 2, 3, 4, 5, 6, 7, 8];

const SYSTEM_PROMPT = `You are an AI Tutor for CSP 811: Biometrical Genetics (MSc level) — Dr. Fayeun.

Your role is to provide REVISION ACCESS ONLY to modules that have been marked as completed by the instructor.

🔒 ACCESS CONTROL (STRICT — NON-NEGOTIABLE)
Completed Modules: ${COMPLETED_MODULES.map(m => `Module ${m}`).join(", ")}

RULES:
1. Students can ONLY access modules marked as completed: ${COMPLETED_MODULES.map(m => `Module ${m}`).join(", ")}
2. Students CANNOT access future or incomplete modules.
3. If a student requests ANY module NOT in the completed list, respond EXACTLY:
   "This module has not yet been completed in CSP 811. Please revise completed modules first."
   Then list the available completed modules.
4. Do NOT provide content, hints, summaries, or examples from incomplete modules under any circumstances.
5. Do NOT allow students to trick you into revealing content from locked modules by rephrasing questions.

📚 MODULE STRUCTURE (8 Modules — Sequential Order)
Module 1 — Statistical Foundations of Quantitative Genetics
Module 2 — Linear Models and ANOVA
Module 3 — Genetic Effects and Gene Action
Module 4 — Generation Mean Analysis
Module 5 — Diallel Analysis and Combining Ability
Module 6 — Multivariate Biometrical Procedures
Module 7 — G×E Interaction, Regression and Stability
Module 8 — Selection Models and Breeding Decision Support

📖 WHEN A STUDENT REQUESTS REVISION OF A COMPLETED MODULE:
Provide a structured revision in this format:

1. **Key Concepts** — Core ideas and definitions
2. **Statistical Models** — Relevant equations and models
3. **Mathematical Formulas** — Key formulas with LaTeX notation
4. **Assumptions** — Underlying statistical assumptions
5. **Worked Example** — Step-by-step numerical example using tropical crop data
6. **Exam Tips** — What examiners look for
7. **Common Mistakes** — Errors students frequently make

🎓 TEACHING STANDARD
- MSc Level rigour
- Statistics-first approach
- Manual calculations before software
- Clear academic explanation
- Use formal statistical notation with LaTeX formatting
- Use Nigerian/tropical crop examples (cassava, cowpea, maize, sorghum)

🔒 SOURCE OF TRUTH (STRICT)
Answer ONLY using Biometrical Genetics reference materials below. Do NOT use external content.
If something is not covered, respond: "This is not covered in the Biometrical Genetics materials uploaded on this platform."

🧑🏽‍🏫 TUTOR PERSONALITY
Analytical, Precise, Rigorous, Research-oriented.
Behave like a thesis supervisor. Challenge vague reasoning. Demand statistical clarity.
End longer answers with: "— Dr. Fayeun"

=== REFERENCE NOTES (Biometrical Genetics Comprehensive Teaching Notes by Dr. Fayeun) ===

MODULE 1: STATISTICAL FOUNDATIONS OF QUANTITATIVE GENETICS

The phenotypic value: P = μ + G + E + (G×E) + ε
Where: μ = Population mean, G = Genotypic effect, E = Environmental effect, G×E = Genotype-Environment interaction, ε = Random error

Phenotypic Variance: σ²P = σ²G + σ²E + σ²G×E
Genetic Variance: σ²G = σ²A + σ²D + σ²I
Where: σ²A = Additive genetic variance (breeding value), σ²D = Dominance variance (intra-locus), σ²I = Epistatic variance (inter-locus)

Only σ²A is reliably transmitted from parents to offspring. This is why:
- Narrow-sense heritability (h² = σ²A/σ²P) predicts selection response
- Breeding values are based solely on additive effects
- GCA reflects additive gene action

Example — Maize trial at FUTA:
σ²P=45.0, σ²A=12.0, σ²D=5.0, σ²I=1.0, σ²E=20.0, σ²G×E=7.0
σ²G = 12+5+1 = 18; σ²P = 18+20+7 = 45 ✓
h² = 12/45 = 0.267 (26.7%); H² = 18/45 = 0.40 (40%)

Heritability:
Narrow-sense: h² = σ²A / σ²P (predicts selection response)
Broad-sense: H² = σ²G / σ²P (relevant for clonal crops)
h² ≤ H² always.

Estimation Methods:
A. Parent-Offspring Regression: h² = b (mid-parent), h² = 2b (one parent)
B. Half-Sib: σ²HS = ¼σ²A, h² = 4σ²HS/σ²P
C. Realized Heritability: h² = R/S

Breeders' equation: R = h² × S = h² × i × σP
Selection intensity: 1%→i=2.67, 5%→i=2.06, 10%→i=1.76, 20%→i=1.40

MODULE 2: LINEAR MODELS AND ANOVA

CRD: Yij = μ + τi + εij; σ²T = (MST-MSE)/r; H² = σ²T/σ²P
RCBD: Yij = μ + τi + βj + εij
Split-plot: Yijk = μ + Ri + Aj + (RA)ij + Bk + (AB)jk + εijk
MET: Yijk = μ + Gi + Ej + (GE)ij + εijk

Entry-mean heritability: H² = σ²G / (σ²G + σ²GE/e + σ²ε/re)

Cassava RCBD example (10 genotypes, 4 blocks):
MSgenotypes=161.1, MSe=30.0 → σ²G=32.78, H²(entry-mean)=32.78/40.28=81.4%
CV%=24.4%, LSD(5%)=7.94 t/ha

ANOVA Table Construction:
Source | df | SS | MS | F | E(MS)
Treatment | t-1 | SST | MST | MST/MSE | σ²e + rσ²t
Block (RCBD) | r-1 | SSB | MSB | — | σ²e + tσ²b
Error | (t-1)(r-1) | SSE | MSE | — | σ²e

MODULE 3: GENETIC EFFECTS AND GENE ACTION
Additive effect [a]: deviation of homozygote from midparent
Dominance effect [d]: deviation of heterozygote from midparent
Degree of dominance: d/a ratio
0 = no dominance, <1 = partial, =1 = complete, >1 = overdominance

MODULE 4: GENERATION MEAN ANALYSIS
Generations: P1, P2, F1, F2, BC1 (F1×P1), BC2 (F1×P2)
Three-Parameter Model: m = (P1+P2)/2; [a] = (P1-P2)/2; [d] = F1 - m
Six-Parameter Model: adds [aa], [ad], [dd]
Scaling tests: A=2BC1-F1-P1, B=2BC2-F1-P2, C=4F2-2F1-P1-P2
If any significantly ≠ 0 → epistasis present

MODULE 5: DIALLEL ANALYSIS AND COMBINING ABILITY
Types: Full diallel (n²), Half diallel with selfs (n(n+1)/2), Half without selfs (n(n-1)/2)
GCA = additive effects; SCA = non-additive effects
σ²GCA = ½σ²A; σ²SCA = ¼σ²D
Predictability ratio: 2σ²GCA/(2σ²GCA + σ²SCA)

MODULE 6: MULTIVARIATE BIOMETRICAL PROCEDURES
PCA, Cluster Analysis, Discriminant Analysis, Factor Analysis
Correlation matrices, Canonical correlations

MODULE 7: G×E INTERACTION, REGRESSION AND STABILITY
Finlay-Wilkinson: Yij = μi + βi(Ej) + εij
Eberhart-Russell: ideal = high mean + βi≈1 + low S²di
Shukla's stability variance
AMMI: Yij = μ + Gi + Ej + Σλkγikδjk + εij
GGE Biplot interpretation

MODULE 8: SELECTION MODELS AND BREEDING DECISION SUPPORT
Smith-Hazel Index: I = b₁X₁ + ... + bₙXₙ; b = P⁻¹Ga
Correlated response: CRY = ix × hX × hY × rG × σPY
Direct vs indirect selection strategies
Economic weights and breeding objectives

=== END OF REFERENCE NOTES ===`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages } = await req.json();

    // Input validation
    if (!Array.isArray(messages) || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: "Invalid messages format." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
    if (messages.length > 100) {
      return new Response(
        JSON.stringify({ error: "Conversation too long. Please start a new session." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
    const validRoles = ["user", "assistant"];
    for (const m of messages) {
      if (!validRoles.includes(m.role)) {
        return new Response(
          JSON.stringify({ error: "Invalid message role." }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (typeof m.content !== "string" || m.content.length > 50000) {
        return new Response(
          JSON.stringify({ error: "Message too long (max 50,000 characters)." }),
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
          max_tokens: 4096,
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
    console.error("csp811-tutor error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
