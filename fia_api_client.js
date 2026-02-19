/**
 * FIA AI Proxy — Lovable Frontend Integration
 * =============================================
 * Replace direct Anthropic calls in your Lovable project with these functions.
 * Set VITE_API_BASE_URL in Lovable → Settings → Environment Variables
 *
 * VITE_API_BASE_URL = https://your-render-app.onrender.com
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL;

// ── 1. CSP 811 TUTOR CHAT (streaming) ─────────────────────────────────────
/**
 * Call this from your CSP 811 tutor page instead of hitting Anthropic directly.
 * Streams the response token by token.
 *
 * @param {Array} messages  - [{role: "user", content: "..."}, ...]
 * @param {string} topic    - e.g. "Topic 2: Heritability" (optional)
 * @param {function} onChunk - callback receives each text chunk as it streams
 * @param {function} onDone  - callback when stream completes
 */
export async function streamTutorChat(messages, topic = null, onChunk, onDone) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, topic, stream: true }),
  });

  if (!response.ok) {
    throw new Error(`Proxy error: ${response.status}`);
  }

  await readStream(response, onChunk, onDone);
}


// ── 2. VIVASENSE AI INTERPRETATION (streaming) ────────────────────────────
/**
 * Call this automatically after VivaSense completes an analysis.
 * Streams Dr. Fayeun's interpretation of the results.
 *
 * @param {string} analysisType  - e.g. "One-way ANOVA"
 * @param {object} results       - the full results JSON from your analysis engine
 * @param {string} crop          - e.g. "cowpea", "cassava" (optional)
 * @param {function} onChunk     - callback for each streamed chunk
 * @param {function} onDone      - callback when complete
 *
 * EXAMPLE RESULTS OBJECT:
 * {
 *   summary: { design: "CRD one-way", factor: "treatment", n: 3, p_value: 0.003 },
 *   anova_table: { ... },
 *   means: { A: 125, B: 102, C: 88 },
 *   tukey: { ... },
 *   assumptions: { shapiro_p: 0.43, levene_p: 0.21 }
 * }
 */
export async function streamVivaSenseInterpretation(analysisType, results, crop = null, onChunk, onDone) {
  const response = await fetch(`${API_BASE}/api/interpret`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis_type: analysisType,
      results,
      crop,
      stream: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`Interpretation error: ${response.status}`);
  }

  await readStream(response, onChunk, onDone);
}


// ── 3. VIVASENSE POST-ANALYSIS FOLLOW-UP CHAT (streaming) ─────────────────
/**
 * Powers the "Ask Dr. Fayeun about your results" chat box.
 * Always passes the analysis results so replies reference actual data.
 *
 * @param {Array}  messages        - full conversation history
 * @param {object} analysisResults - the results object (same as above)
 * @param {function} onChunk       - callback for each streamed chunk
 * @param {function} onDone        - callback when complete
 */
export async function streamFollowupChat(messages, analysisResults, onChunk, onDone) {
  const response = await fetch(`${API_BASE}/api/followup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      analysis_results: analysisResults,
      stream: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`Follow-up error: ${response.status}`);
  }

  await readStream(response, onChunk, onDone);
}


// ── SHARED STREAM READER ──────────────────────────────────────────────────
async function readStream(response, onChunk, onDone) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    for (const line of chunk.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;
      try {
        const parsed = JSON.parse(data);
        if (parsed.type === "content_block_delta" && parsed.delta?.text) {
          fullContent += parsed.delta.text;
          onChunk(parsed.delta.text, fullContent);
        }
      } catch {}
    }
  }

  if (onDone) onDone(fullContent);
}


// ── HEALTH CHECK ──────────────────────────────────────────────────────────
export async function checkProxyHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return await res.json();
  } catch {
    return { status: "unreachable" };
  }
}


/* ═══════════════════════════════════════════════════════════════
   LOVABLE USAGE EXAMPLES
   ═══════════════════════════════════════════════════════════════

   -- CSP 811 Tutor --
   Replace your existing fetch() call with:

   let response = "";
   await streamTutorChat(
     messages,
     selectedTopic?.title,
     (chunk, full) => setStreamingContent(full),   // update UI as it types
     (full) => {
       setMessages(prev => [...prev, { role: "assistant", content: full }]);
       setStreamingContent("");
     }
   );


   -- VivaSense: auto-interpret after analysis --
   After your analysis API returns results:

   let interpretation = "";
   setInterpretationLoading(true);
   await streamVivaSenseInterpretation(
     analysisType,          // "One-way ANOVA"
     analysisResults,       // your results JSON
     cropName,              // "cowpea" (optional)
     (chunk, full) => setInterpretationText(full),
     (full) => setInterpretationLoading(false)
   );


   -- VivaSense: follow-up chat --
   In your chat component:

   await streamFollowupChat(
     chatMessages,
     analysisResults,
     (chunk, full) => setStreamingReply(full),
     (full) => {
       setChatMessages(prev => [...prev, { role: "assistant", content: full }]);
       setStreamingReply("");
     }
   );
*/
