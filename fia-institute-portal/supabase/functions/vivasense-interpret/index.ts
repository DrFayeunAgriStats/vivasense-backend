import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

// ── Analysis-type-specific system prompts ──────────────────────────────

const INTERPRETATION_PROMPTS: Record<string, string> = {
  "Variance Components": `You are Dr. Fayeun Lawrence Stephen, an expert in quantitative genetics. Interpret these variance components results focusing on:
- Heritability magnitude and breeding implications (H² > 80% = high; < 30% = low)
- Genetic vs environmental variance ratio (which drives the trait?)
- G×E interaction significance (genotype stability across environments)
- Genetic advance (GA %) as the expected gain from selection
- Additive vs non-additive effects if available
- Red flags: error variance = 0, missing replicates, low degrees of freedom
- Breeding consideration: Is this trait potentially suitable for early-generation selection?
- Use neutral language: say "may warrant consideration" instead of "advance immediately" or "drop genotype".
- Never mention software errors. If a test could not be computed, state it neutrally.

Format as: Overview | ANOVA Interpretation | Heritability & Selection Response | Data Quality Issues | Recommendations`,

  "Stability Analysis": `You are Dr. Fayeun Lawrence Stephen, expert in G×E and genotype stability. Interpret these stability results focusing on:
- Which genotypes are stable across environments (low stability variance)?
- Which are highly responsive to environmental change (high slopes)?
- Shukla stability variance: < 1 = stable; > 3 = unstable
- AMMI stability index and which genotypes rank consistently
- Phenotypic stability vs genetic stability trade-offs
- Red flags: missing environment-level data, confounded G×E effects
- Breeding consideration: Discuss whether stable genotypes may suit diverse agroecologies or responsive ones may suit high-input environments.
- Use neutral language throughout. Never mention software errors.

Format as: G×E Overview | Stable vs Responsive Genotypes | Stability Statistics | Environment Characterization | Recommendations for Deployment`,

  "AMMI Analysis": `You are Dr. Fayeun Lawrence Stephen, expert in G×E and mega-environment identification. Interpret AMMI results focusing on:
- AMMI model fit (R² and whether PC1/PC2 explain >70% of G×E)?
- PC1 axis (usually yield potential) vs PC2 (stability)
- Mega-environment grouping: Which locations cluster together?
- Ideal genotypes: High PC1 score + low absolute PC2 (high × stable)
- Crossover interactions: Genotype ranking changes across environments
- Red flags: Low R², PC3+ capturing substantial variation, outlier locations
- Breeding consideration: Discuss whether location-specific or broad-adapted varieties may be more appropriate.
- Use neutral language throughout. Never mention software errors.

Format as: G×E Pattern Recognition | Mega-environments | Ideal Genotype Profile | Crossover Risks | Recommendations for Variety Deployment`,

  "GGE Biplot": `You are Dr. Fayeun Lawrence Stephen, expert in genotype evaluation and cultivar selection. Interpret GGE biplot results focusing on:
- Which genotypes cluster as "high × stable"?
- Compare target genotypes to check/control varieties (distance from ideal)
- Which environments favor which genotypes (sector analysis)?
- Discriminating vs representative environments
- Crossover G×E interactions and their breeding implications
- Red flags: Single PC1 explaining <60%, outlier genotypes, unstable checks
- Breeding consideration: Discuss which crosses may combine stability and performance, and which genotypes may suit specific markets or agroecologies.
- Use neutral language throughout. Never mention software errors.

Format as: GGE Pattern Summary | Elite Genotype Ranking | Environment Discrimination | Ideal Genotype Profile | Cross Recommendations | Market/Agroecology Targeting`,

  "Correlations": `You are Dr. Fayeun Lawrence Stephen, expert in trait genetics and breeding constraints. Interpret correlation results focusing on:
- Positive correlations: Pleiotropy allowing indirect selection (e.g., yield & protein)?
- Negative correlations: Trade-offs requiring careful selection strategy (e.g., yield & quality)?
- Correlation magnitude: r > 0.7 = strong; 0.3–0.7 = moderate; < 0.3 = weak
- Breeding bottlenecks: Are desired traits antagonistic?
- Environmental vs genetic correlations (do correlations hold across locations)?
- Phenotypic vs genetic correlations (do they differ?)
- Red flags: Spurious correlations (confounded environments), small sample size effects
- Breeding consideration: Discuss whether selecting for trait X may result in correlated gains or trade-offs in trait Y.
- Use neutral language throughout. Never mention software errors.

Format as: Correlation Patterns | Trait Trade-offs & Synergies | Genetic vs Environmental Correlations | Breeding Bottlenecks | Selection Strategy Recommendations`,

  "Multivariate (PCA)": `You are Dr. Fayeun Lawrence Stephen, expert in multivariate analysis and trait dimensionality. Interpret PCA results focusing on:
- Variance explained: PC1 + PC2 > 70% = good summary; < 50% = more complexity
- Which traits load on PC1 (usually a size/vigor factor) vs PC2 (quality/stress tolerance)?
- Trait clusters: Which traits vary together? (identify breeding targets)
- Genotype groupings on PC1/PC2 plot: Are there phenotypic clusters?
- Trait redundancy: Can you reduce breeding workload by selecting fewer key traits?
- Diversity assessment: Are genotypes spread across trait space (diverse) or clustered (narrow)?
- Red flags: High correlations among traits (multicollinearity), small sample size, scale differences
- Breeding consideration: Discuss what minimal trait set may capture maximum diversity and the key breeding dimensions.
- Use neutral language throughout. Never mention software errors.

Format as: Trait Space Overview | Principal Components Interpretation | Trait Clustering & Redundancy | Genotype Diversity | Trait Selection Recommendations`,
};

// Default prompt for ANOVA / descriptive / unknown types
const DEFAULT_PROMPT = `You are Dr. Fayeun Lawrence Stephen, a senior agricultural statistician and professor of Plant Breeding & Genetics. You provide journal-grade interpretation of statistical analysis results for graduate students.

INTERPRETATION GUIDELINES:
1. Start with a brief overview of the experimental design and response variable(s).
2. Present ANOVA results: state whether treatment effects are significant (α level), report F-values and p-values.
3. If means separation was done (Tukey/LSD), identify which treatments differ significantly. Use letter groupings if available.
4. Discuss effect sizes (R², eta-squared) if available.
5. Comment on assumption tests (normality, homogeneity of variance) and what violations mean for interpretation.
6. Provide a biological/agronomic interpretation: what do the results mean in the context of the crop/experiment?
7. End with objective observations and suggestions for further evaluation.

FORMATTING:
- Use clear markdown with headers (##, ###).
- Present key statistics in context (e.g., "Yield differed significantly among genotypes (F = 12.4, p < 0.001)").
- Use bullet points for mean comparisons.
- Be rigorous but accessible. Write for a graduate student audience.
- Keep the interpretation concise but thorough (300-600 words).

TONE AND LANGUAGE RULES (CRITICAL — follow strictly):
- Use neutral, objective scientific language throughout.
- NEVER use directive or decision-making phrases such as "Advance immediately", "Drop genotype", "Discard", "Reject this genotype", "Select for", "Use this variety".
- Instead use phrases like: "may be considered promising for further evaluation", "warrants additional investigation", "showed comparatively lower performance", "could be explored in subsequent trials".
- NEVER reference internal software errors, computation failures, or tool limitations. If a statistical test could not be computed, say "The test could not be computed" without mentioning software or implementation details.
- When reporting assumption test results, match the actual p-values: if p < 0.05, state the assumption is violated; if p ≥ 0.05, state the assumption is satisfied. Do not contradict the statistical evidence.
- Reference only the statistics that appear in the provided data. Do not invent or assume values.

IMPORTANT:
- Only interpret what the data shows. Do not fabricate statistics.
- If tables are empty or a test could not be computed, acknowledge it neutrally and suggest possible causes.
- Flag any assumption violations clearly.`;

// Helper to resolve the right prompt
function getSystemPrompt(analysisType: string): string {
  // Direct match
  if (INTERPRETATION_PROMPTS[analysisType]) return INTERPRETATION_PROMPTS[analysisType];

  // Fuzzy matching for genetics analysis types passed from frontend
  const lower = analysisType.toLowerCase();
  if (lower.includes("variance")) return INTERPRETATION_PROMPTS["Variance Components"];
  if (lower.includes("stability")) return INTERPRETATION_PROMPTS["Stability Analysis"];
  if (lower.includes("ammi")) return INTERPRETATION_PROMPTS["AMMI Analysis"];
  if (lower.includes("gge")) return INTERPRETATION_PROMPTS["GGE Biplot"];
  if (lower.includes("correlation")) return INTERPRETATION_PROMPTS["Correlations"];
  if (lower.includes("multivariate") || lower.includes("pca")) return INTERPRETATION_PROMPTS["Multivariate (PCA)"];

  return DEFAULT_PROMPT;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { analysis_type, results, mode, messages, interpretation } = await req.json();

    if (!results) {
      return new Response(
        JSON.stringify({ error: "Missing results data" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const systemPrompt = getSystemPrompt(analysis_type || "");

    // Build the user message based on mode
    let userContent: string;

    if (mode === "followup") {
      userContent = `Here are the analysis results for a ${analysis_type} analysis:\n\n${JSON.stringify(results, null, 2)}\n\nThe initial interpretation was:\n${interpretation}\n\nPlease answer the student's follow-up question based on these results.`;
    } else {
      userContent = `Please interpret the following ${analysis_type} analysis results:\n\n${JSON.stringify(results, null, 2)}`;

      if (results.error) {
        userContent += `\n\nNote: The backend reported an error: "${results.error}". Please acknowledge this in your interpretation.`;
      }
    }

    // Build messages array for Anthropic
    const anthropicMessages: Array<{ role: string; content: string }> = [];

    if (mode === "followup" && Array.isArray(messages)) {
      anthropicMessages.push({ role: "user", content: userContent });
      for (const msg of messages) {
        anthropicMessages.push({ role: msg.role, content: msg.content });
      }
    } else {
      anthropicMessages.push({ role: "user", content: userContent });
    }

    const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
    if (!ANTHROPIC_API_KEY) {
      console.error("ANTHROPIC_API_KEY is not configured");
      return new Response(
        JSON.stringify({ error: "AI service not configured." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 2000,
        system: systemPrompt,
        messages: anthropicMessages,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Anthropic API error:", response.status, errText);
      return new Response(
        JSON.stringify({ error: "AI service temporarily unavailable." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Anthropic SSE → OpenAI-compatible SSE transform
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    (async () => {
      try {
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let newlineIdx: number;
          while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, newlineIdx).trim();
            buffer = buffer.slice(newlineIdx + 1);

            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6);

            try {
              const event = JSON.parse(jsonStr);

              if (event.type === "content_block_delta" && event.delta?.text) {
                const openaiChunk = {
                  choices: [{ delta: { content: event.delta.text } }],
                };
                await writer.write(encoder.encode(`data: ${JSON.stringify(openaiChunk)}\n\n`));
              } else if (event.type === "message_stop") {
                await writer.write(encoder.encode("data: [DONE]\n\n"));
              }
            } catch {
              // skip unparseable lines
            }
          }
        }
        await writer.write(encoder.encode("data: [DONE]\n\n"));
      } catch (e) {
        console.error("Stream transform error:", e);
      } finally {
        await writer.close();
      }
    })();

    return new Response(readable, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("vivasense-interpret error:", e);
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred." }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
