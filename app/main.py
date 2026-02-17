import React, { useMemo, useState } from "react";

/**
 * VivaSense V1 Frontend (Lovable-ready)
 *
 * IMPORTANT (Backend compatibility):
 * This frontend sends requests as multipart/form-data with fields like:
 *  - file (CSV)
 *  - factor, trait, alpha, block (optional), traits (repeated), columns (repeated), method, x, y
 *
 * Your FastAPI endpoints MUST accept these as Form(...) + File(...).
 * (If your backend expects JSON body models, switch backend to Form inputs.)
 */

export default function VivaSenseFrontend() {
  // ---- CONFIG
  const [apiBaseUrl, setApiBaseUrl] = useState("https://vivasense-backend.onrender.com");

  // ---- File + columns
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [numericCols, setNumericCols] = useState([]);
  const [nonNumericCols, setNonNumericCols] = useState([]);
  const [fileName, setFileName] = useState("");

  // ---- UI mode
  const MODES = [
    { key: "anova_oneway", label: "One-way ANOVA (Single Trait)" },
    { key: "anova_multitrait", label: "One-way ANOVA (Multi-trait)" },
    { key: "correlation", label: "Correlation + Heatmap" },
    { key: "regression", label: "Simple Regression" },
  ];
  const [mode, setMode] = useState("anova_oneway");

  // ---- Common params
  const [alpha, setAlpha] = useState(0.05);

  // ---- ANOVA params
  const [factor, setFactor] = useState("");
  const [block, setBlock] = useState(""); // optional (RCBD)
  const [trait, setTrait] = useState("");
  const [traitsMulti, setTraitsMulti] = useState([]);

  // ---- Correlation params
  const [corrMethod, setCorrMethod] = useState("pearson");
  const [corrColumns, setCorrColumns] = useState([]);

  // ---- Regression params
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");

  // ---- Results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // ----------------------------
  // CSV header + numeric inference
  // ----------------------------
  async function readCsvColumnsAndTypes(fileObj) {
    const text = await fileObj.text();
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);

    if (lines.length < 2) return { cols: [], numericCols: [], nonNumericCols: [] };

    const headerLine = lines[0];

    // Parse header (simple quote-aware)
    const cols = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < headerLine.length; i++) {
      const ch = headerLine[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
        continue;
      }
      if (ch === "," && !inQuotes) {
        cols.push(current.trim());
        current = "";
      } else {
        current += ch;
      }
    }
    if (current.length) cols.push(current.trim());

    const cleanCols = cols.filter((c) => c.length > 0);
    if (!cleanCols.length) return { cols: [], numericCols: [], nonNumericCols: [] };

    // Sample up to 40 data rows
    const sampleLines = lines.slice(1, Math.min(lines.length, 41));

    const counts = {};
    const numericCounts = {};
    cleanCols.forEach((c) => {
      counts[c] = 0;
      numericCounts[c] = 0;
    });

    const isNum = (v) => v !== null && v !== "" && !Number.isNaN(Number(v));

    for (const line of sampleLines) {
      // naive split (works well for typical experiment CSVs)
      const parts = line.split(",");
      for (let i = 0; i < cleanCols.length; i++) {
        const col = cleanCols[i];
        const v = (parts[i] ?? "").replaceAll('"', "").trim();
        if (!v) continue;
        counts[col] += 1;
        if (isNum(v)) numericCounts[col] += 1;
      }
    }

    const numeric = [];
    const nonNumeric = [];
    for (const c of cleanCols) {
      const seen = counts[c];
      const num = numericCounts[c];
      const ratio = seen === 0 ? 0 : num / seen;
      if (ratio >= 0.8) numeric.push(c);
      else nonNumeric.push(c);
    }

    return { cols: cleanCols, numericCols: numeric, nonNumericCols: nonNumeric };
  }

  function resetResults() {
    setError("");
    setResult(null);
  }

  function onPickFile(e) {
    resetResults();
    const f = e.target.files?.[0] || null;

    setFile(f);
    setFileName(f ? f.name : "");

    setColumns([]);
    setNumericCols([]);
    setNonNumericCols([]);

    setFactor("");
    setBlock("");
    setTrait("");
    setTraitsMulti([]);

    setCorrColumns([]);
    setXCol("");
    setYCol("");

    if (!f) return;

    readCsvColumnsAndTypes(f)
      .then(({ cols, numericCols, nonNumericCols }) => {
        setColumns(cols);
        setNumericCols(numericCols);
        setNonNumericCols(nonNumericCols);

        // Defaults
        const defaultFactor = nonNumericCols[0] || cols[0] || "";
        setFactor(defaultFactor);

        const blockCandidate =
          cols.find((c) => c.toLowerCase() === "block") ||
          cols.find((c) => c.toLowerCase() === "rep") ||
          cols.find((c) => c.toLowerCase() === "replicate") ||
          "";
        setBlock(blockCandidate);

        const defaultTrait = numericCols[0] || "";
        setTrait(defaultTrait);

        setTraitsMulti(numericCols.slice(0, Math.min(numericCols.length, 4)));
        setCorrColumns(numericCols.slice(0, Math.min(numericCols.length, 6)));

        setXCol(numericCols[0] || "");
        setYCol(numericCols[1] || "");
      })
      .catch(() => {
        setColumns([]);
        setNumericCols([]);
        setNonNumericCols([]);
      });
  }

  function validate() {
    if (!apiBaseUrl?.trim()) return "Please set your API base URL.";
    if (!file) return "Please upload a CSV file.";
    if (!columns.length) return "Could not detect CSV headers. Ensure the first row contains column names.";

    if (mode === "anova_oneway") {
      if (!factor) return "Please select a factor column.";
      if (!trait) return "Please select a trait column (numeric).";
      if (!numericCols.includes(trait)) return "Trait must be numeric (e.g., Yield, Height).";
      if (block && block === factor) return "Block cannot be the same as Factor.";
      if (block && numericCols.includes(block)) return "Block should be categorical (non-numeric).";
    }

    if (mode === "anova_multitrait") {
      if (!factor) return "Please select a factor column.";
      if (!traitsMulti.length) return "Please select one or more traits.";
      for (const t of traitsMulti) {
        if (!numericCols.includes(t)) return `Trait '${t}' must be numeric.`;
      }
      if (block && block === factor) return "Block cannot be the same as Factor.";
      if (block && numericCols.includes(block)) return "Block should be categorical (non-numeric).";
    }

    if (mode === "correlation") {
      if (!corrColumns.length) return "Please select numeric columns for correlation.";
      for (const c of corrColumns) {
        if (!numericCols.includes(c)) return `Correlation column '${c}' must be numeric.`;
      }
      if (!["pearson", "spearman"].includes(corrMethod)) return "Correlation method must be pearson or spearman.";
    }

    if (mode === "regression") {
      if (!xCol || !yCol) return "Please select x and y columns.";
      if (xCol === yCol) return "x and y must be different columns.";
      if (!numericCols.includes(xCol) || !numericCols.includes(yCol)) return "Regression x and y must be numeric columns.";
    }

    return "";
  }

  async function postMultipart(url, formData) {
    const res = await fetch(url, { method: "POST", body: formData });
    const contentType = res.headers.get("content-type") || "";
    let payload = null;

    if (contentType.includes("application/json")) {
      payload = await res.json().catch(() => null);
    } else {
      const txt = await res.text().catch(() => "");
      payload = { detail: txt };
    }

    if (!res.ok) {
      const msg = payload?.detail || payload?.error || "Request failed.";
      throw new Error(msg);
    }
    return payload;
  }

  async function runAnalysis() {
    resetResults();
    const v = validate();
    if (v) {
      setError(v);
      return;
    }

    setLoading(true);
    try {
      if (mode === "anova_oneway") {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("factor", factor);
        fd.append("trait", trait);
        fd.append("alpha", String(alpha));
        if (block) fd.append("block", block);

        const data = await postMultipart(`${apiBaseUrl}/analyze/anova/oneway`, fd);
        setResult(data);
      }

      if (mode === "anova_multitrait") {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("factor", factor);
        traitsMulti.forEach((t) => fd.append("traits", t));
        fd.append("alpha", String(alpha));
        if (block) fd.append("block", block);

        const data = await postMultipart(`${apiBaseUrl}/analyze/anova/multitrait`, fd);
        setResult(data);
      }

      if (mode === "correlation") {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("method", corrMethod);
        corrColumns.forEach((c) => fd.append("columns", c));

        const data = await postMultipart(`${apiBaseUrl}/analyze/correlation`, fd);
        setResult(data);
      }

      if (mode === "regression") {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("x", xCol);
        fd.append("y", yCol);

        const data = await postMultipart(`${apiBaseUrl}/analyze/regression/simple`, fd);
        setResult(data);
      }
    } catch (e) {
      setError(e?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function Pill({ text }) {
    return <span style={styles.pill}>{text}</span>;
  }

  const detectedCols = useMemo(() => columns || [], [columns]);

  // ----------------------------
  // UI
  // ----------------------------
  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>VivaSense V1</div>
          <div style={styles.subtitle}>Upload CSV → Analyze → Get tables, plots, and interpretation</div>
        </div>
      </div>

      <div style={styles.card}>
        <div style={styles.grid2}>
          <div>
            <label style={styles.label}>Backend API Base URL</label>
            <input
              style={styles.input}
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
              placeholder="https://vivasense-backend.onrender.com"
            />
            <div style={styles.hint}>
              Example: <code>http://127.0.0.1:8000</code> or <code>https://vivasense-backend.onrender.com</code>
            </div>
          </div>

          <div>
            <label style={styles.label}>Upload CSV</label>
            <input style={styles.input} type="file" accept=".csv" onChange={onPickFile} />
            <div style={styles.hint}>
              {fileName ? (
                <>
                  Loaded: <b>{fileName}</b>
                </>
              ) : (
                "Choose a CSV file with a header row (column names)."
              )}
            </div>
          </div>
        </div>

        <div style={styles.hr} />

        <div style={styles.grid2}>
          <div>
            <label style={styles.label}>Analysis Type</label>
            <select
              style={styles.select}
              value={mode}
              onChange={(e) => {
                resetResults();
                setMode(e.target.value);
              }}
            >
              {MODES.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>

            <div style={styles.hint}>
              Detected columns:{" "}
              {detectedCols.length ? (
                detectedCols.slice(0, 12).map((c) => <Pill key={c} text={c} />)
              ) : (
                <span style={styles.muted}>Upload a CSV to detect columns.</span>
              )}
              {detectedCols.length > 12 && <span style={styles.muted}> +{detectedCols.length - 12} more</span>}
              <div style={{ marginTop: 8 }}>
                <span style={styles.muted}>Numeric:</span>{" "}
                {numericCols.length ? numericCols.slice(0, 10).map((c) => <Pill key={c} text={c} />) : <span style={styles.muted}>none</span>}
              </div>
              <div style={{ marginTop: 6 }}>
                <span style={styles.muted}>Categorical:</span>{" "}
                {nonNumericCols.length ? nonNumericCols.slice(0, 10).map((c) => <Pill key={c} text={c} />) : <span style={styles.muted}>none</span>}
              </div>
            </div>
          </div>

          {(mode === "anova_oneway" || mode === "anova_multitrait") && (
            <div>
              <label style={styles.label}>Alpha (significance level)</label>
              <input
                style={styles.input}
                type="number"
                step="0.01"
                min="0.0001"
                max="0.2"
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
              />
              <div style={styles.hint}>Default is 0.05</div>
            </div>
          )}
        </div>

        {/* Mode-specific controls */}
        {mode === "anova_oneway" && (
          <div style={styles.cardInner}>
            <div style={styles.sectionTitle}>One-way ANOVA (Single Trait)</div>
            <div style={styles.grid4}>
              <div>
                <label style={styles.label}>Factor column</label>
                <select style={styles.select} value={factor} onChange={(e) => setFactor(e.target.value)}>
                  <option value="">-- select --</option>
                  {nonNumericCols.length
                    ? nonNumericCols.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))
                    : detectedCols.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                </select>
              </div>

              <div>
                <label style={styles.label}>Block column (optional)</label>
                <select style={styles.select} value={block} onChange={(e) => setBlock(e.target.value)}>
                  <option value="">-- none --</option>
                  {nonNumericCols
                    .filter((c) => c !== factor)
                    .map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                </select>
                <div style={styles.hint}>Use for RCBD (e.g., Block/Replicate). Leave blank for CRD.</div>
              </div>

              <div>
                <label style={styles.label}>Trait column (numeric)</label>
                <select style={styles.select} value={trait} onChange={(e) => setTrait(e.target.value)}>
                  <option value="">-- select --</option>
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ display: "flex", alignItems: "end" }}>
                <button style={styles.button} onClick={runAnalysis} disabled={loading}>
                  {loading ? "Running..." : "Run ANOVA"}
                </button>
              </div>
            </div>
          </div>
        )}

        {mode === "anova_multitrait" && (
          <div style={styles.cardInner}>
            <div style={styles.sectionTitle}>One-way ANOVA (Multi-trait)</div>
            <div style={styles.grid4}>
              <div>
                <label style={styles.label}>Factor column</label>
                <select style={styles.select} value={factor} onChange={(e) => setFactor(e.target.value)}>
                  <option value="">-- select --</option>
                  {nonNumericCols.length
                    ? nonNumericCols.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))
                    : detectedCols.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                </select>
              </div>

              <div>
                <label style={styles.label}>Block column (optional)</label>
                <select style={styles.select} value={block} onChange={(e) => setBlock(e.target.value)}>
                  <option value="">-- none --</option>
                  {nonNumericCols
                    .filter((c) => c !== factor)
                    .map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                </select>
                <div style={styles.hint}>Use for RCBD (e.g., Block/Replicate). Leave blank for CRD.</div>
              </div>

              <div>
                <label style={styles.label}>Traits (numeric, multi-select)</label>
                <select
                  style={styles.select}
                  multiple
                  size={Math.min(8, Math.max(4, numericCols.length))}
                  value={traitsMulti}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions).map((o) => o.value);
                    setTraitsMulti(selected);
                  }}
                >
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                <div style={styles.hint}>Hold Ctrl/Command to select multiple traits.</div>
              </div>

              <div style={{ display: "flex", alignItems: "end" }}>
                <button style={styles.button} onClick={runAnalysis} disabled={loading}>
                  {loading ? "Running..." : "Run Multi-trait ANOVA"}
                </button>
              </div>
            </div>
          </div>
        )}

        {mode === "correlation" && (
          <div style={styles.cardInner}>
            <div style={styles.sectionTitle}>Correlation + Heatmap</div>
            <div style={styles.grid3}>
              <div>
                <label style={styles.label}>Method</label>
                <select style={styles.select} value={corrMethod} onChange={(e) => setCorrMethod(e.target.value)}>
                  <option value="pearson">pearson</option>
                  <option value="spearman">spearman</option>
                </select>
              </div>

              <div>
                <label style={styles.label}>Columns (numeric, multi-select)</label>
                <select
                  style={styles.select}
                  multiple
                  size={Math.min(8, Math.max(4, numericCols.length))}
                  value={corrColumns}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions).map((o) => o.value);
                    setCorrColumns(selected);
                  }}
                >
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ display: "flex", alignItems: "end" }}>
                <button style={styles.button} onClick={runAnalysis} disabled={loading}>
                  {loading ? "Running..." : "Run Correlation"}
                </button>
              </div>
            </div>
          </div>
        )}

        {mode === "regression" && (
          <div style={styles.cardInner}>
            <div style={styles.sectionTitle}>Simple Regression</div>
            <div style={styles.grid3}>
              <div>
                <label style={styles.label}>X (predictor, numeric)</label>
                <select style={styles.select} value={xCol} onChange={(e) => setXCol(e.target.value)}>
                  <option value="">-- select --</option>
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={styles.label}>Y (response, numeric)</label>
                <select style={styles.select} value={yCol} onChange={(e) => setYCol(e.target.value)}>
                  <option value="">-- select --</option>
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ display: "flex", alignItems: "end" }}>
                <button style={styles.button} onClick={runAnalysis} disabled={loading}>
                  {loading ? "Running..." : "Run Regression"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Errors */}
        {error && (
          <div style={styles.errorBox}>
            <b>Error:</b> {error}
          </div>
        )}
      </div>

      {/* RESULTS */}
      {result && (
        <div style={styles.card}>
          <div style={styles.sectionTitle}>Results</div>

          {/* Multi-trait wrapper */}
          {result?.meta?.analysis === "multi_trait_oneway_anova" ? (
            <div>
              <div style={styles.hint}>
                Factor: <b>{result?.meta?.factor}</b> | Traits: <b>{result?.meta?.n_traits}</b> | Alpha:{" "}
                <b>{result?.meta?.alpha}</b>
                {result?.meta?.block ? (
                  <>
                    {" "}
                    | Block: <b>{result.meta.block}</b>
                  </>
                ) : null}
              </div>

              {(result.results || []).map((r, idx) => (
                <div key={idx} style={styles.resultBlock}>
                  <div style={styles.resultBlockHeader}>
                    Trait: <b>{r?.meta?.trait}</b>
                    {r?.error && <span style={{ ...styles.pill, marginLeft: 10, background: "#ffd7d7" }}>Error</span>}
                  </div>

                  {r?.error ? <div style={styles.errorBox}>{r.error}</div> : <OneWayAnovaResultView data={r} />}
                </div>
              ))}
            </div>
          ) : (
            <>
              {result?.meta?.analysis === "one_way_anova" && <OneWayAnovaResultView data={result} />}
              {result?.meta?.analysis === "correlation" && <CorrelationResultView data={result} />}
              {result?.meta?.analysis === "simple_regression" && <RegressionResultView data={result} />}
              {!["one_way_anova", "correlation", "simple_regression"].includes(result?.meta?.analysis) && (
                <div style={styles.muted}>Unknown result format.</div>
              )}
            </>
          )}
        </div>
      )}

      <div style={styles.footer}>
        <div style={styles.muted}>
          Tip: If you get a CORS error on Render, ensure your FastAPI CORS middleware allows your Lovable domain.
        </div>
      </div>
    </div>
  );
}

// ----------------------------
// Result Views
// ----------------------------
function OneWayAnovaResultView({ data }) {
  const b64ToImgSrc = (b64) => `data:image/png;base64,${b64}`;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
        <MetaPill label="Analysis" value={data?.meta?.analysis} />
        {data?.meta?.design && <MetaPill label="Design" value={data.meta.design} />}
        <MetaPill label="Factor" value={data?.meta?.factor} />
        {data?.meta?.block && <MetaPill label="Block" value={data.meta.block} />}
        <MetaPill label="Trait" value={data?.meta?.trait} />
        <MetaPill label="Alpha" value={String(data?.meta?.alpha)} />
        <MetaPill label="N used" value={String(data?.meta?.n_rows_used)} />
        {data?.meta?.cv_percent != null && <MetaPill label="CV%" value={Number(data.meta.cv_percent).toFixed(2)} />}
      </div>

      {data?.interpretation && (
        <div style={styles.interpretation}>
          <b>Interpretation:</b> {data.interpretation}
        </div>
      )}

      {data?.assumptions?.notes?.length ? (
        <div style={styles.noteBox}>
          <b>Assumptions:</b>
          <ul style={{ marginTop: 8 }}>
            {data.assumptions.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div style={styles.grid2}>
        <div>
          <div style={styles.subTitle}>ANOVA Table</div>
          <Table rows={data?.anova_table || []} />
        </div>
        <div>
          <div style={styles.subTitle}>Group Summary (Mean, SE, Letters)</div>
          <Table rows={data?.group_summary || []} />
        </div>
      </div>

      <div style={styles.subTitle}>Tukey HSD</div>
      <Table rows={data?.tukey_hsd || []} />

      <div style={styles.grid2}>
        <div>
          <div style={styles.subTitle}>Mean Plot (±SE)</div>
          {data?.plots?.mean_plot_png_b64 ? (
            <img style={styles.img} src={b64ToImgSrc(data.plots.mean_plot_png_b64)} alt="Mean plot" />
          ) : (
            <div style={styles.muted}>No plot</div>
          )}
        </div>
        <div>
          <div style={styles.subTitle}>Boxplot</div>
          {data?.plots?.boxplot_png_b64 ? (
            <img style={styles.img} src={b64ToImgSrc(data.plots.boxplot_png_b64)} alt="Boxplot" />
          ) : (
            <div style={styles.muted}>No plot</div>
          )}
        </div>
      </div>
    </div>
  );
}

function CorrelationResultView({ data }) {
  const b64ToImgSrc = (b64) => `data:image/png;base64,${b64}`;
  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
        <MetaPill label="Analysis" value={data?.meta?.analysis} />
        <MetaPill label="Method" value={data?.meta?.method} />
        <MetaPill label="N used" value={String(data?.meta?.n_rows_used)} />
      </div>

      {data?.interpretation && (
        <div style={styles.interpretation}>
          <b>Interpretation:</b> {data.interpretation}
        </div>
      )}

      <div style={styles.subTitle}>Correlation Matrix</div>
      <Table rows={data?.correlation_matrix || []} />

      <div style={styles.subTitle}>Heatmap</div>
      {data?.plots?.heatmap_png_b64 ? (
        <img style={styles.img} src={b64ToImgSrc(data.plots.heatmap_png_b64)} alt="Heatmap" />
      ) : (
        <div style={styles.muted}>No heatmap</div>
      )}
    </div>
  );
}

function RegressionResultView({ data }) {
  const b64ToImgSrc = (b64) => `data:image/png;base64,${b64}`;
  const m = data?.model || {};
  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
        <MetaPill label="Analysis" value={data?.meta?.analysis} />
        <MetaPill label="X" value={data?.meta?.x} />
        <MetaPill label="Y" value={data?.meta?.y} />
        <MetaPill label="N used" value={String(data?.meta?.n_rows_used)} />
      </div>

      {data?.interpretation && (
        <div style={styles.interpretation}>
          <b>Interpretation:</b> {data.interpretation}
        </div>
      )}

      <div style={styles.subTitle}>Model Summary</div>
      <Table
        rows={[
          {
            intercept: m.intercept,
            slope: m.slope,
            r_squared: m.r_squared,
            p_value_slope: m.p_value_slope,
            stderr_slope: m.stderr_slope,
          },
        ]}
      />

      <div style={styles.subTitle}>Scatter + Fitted Line</div>
      {data?.plots?.scatter_fit_png_b64 ? (
        <img style={styles.img} src={b64ToImgSrc(data.plots.scatter_fit_png_b64)} alt="Regression plot" />
      ) : (
        <div style={styles.muted}>No plot</div>
      )}
    </div>
  );
}

function MetaPill({ label, value }) {
  return (
    <span style={styles.metaPill}>
      <b>{label}:</b> {value ?? ""}
    </span>
  );
}

function Table({ rows }) {
  if (!rows || !rows.length) return <div style={styles.muted}>No data</div>;
  const cols = Object.keys(rows[0] || {});
  return (
    <div style={styles.tableWrap}>
      <table style={styles.table}>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c} style={styles.th}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c} style={styles.td}>
                  {formatCell(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(3);
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

// ----------------------------
// Inline styles (Lovable-friendly)
// ----------------------------
const styles = {
  page: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: 18,
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"',
    color: "#111",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 14,
  },
  title: { fontSize: 26, fontWeight: 800, letterSpacing: -0.3 },
  subtitle: { fontSize: 13, color: "#444", marginTop: 4 },
  card: {
    background: "#fff",
    border: "1px solid #e6e6e6",
    borderRadius: 12,
    padding: 14,
    boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
    marginBottom: 14,
  },
  cardInner: {
    marginTop: 14,
    padding: 12,
    background: "#fafafa",
    border: "1px solid #eee",
    borderRadius: 12,
  },
  sectionTitle: { fontSize: 16, fontWeight: 800, marginBottom: 10 },
  subTitle: { fontSize: 14, fontWeight: 700, marginTop: 12, marginBottom: 6 },
  label: { display: "block", fontSize: 12, fontWeight: 700, marginBottom: 6 },
  hint: { fontSize: 12, color: "#555", marginTop: 6, lineHeight: 1.35 },
  muted: { color: "#666", fontSize: 12 },
  input: {
    width: "100%",
    padding: "10px 10px",
    borderRadius: 10,
    border: "1px solid #ddd",
    outline: "none",
  },
  select: {
    width: "100%",
    padding: "10px 10px",
    borderRadius: 10,
    border: "1px solid #ddd",
    outline: "none",
    background: "#fff",
  },
  button: {
    width: "100%",
    padding: "11px 12px",
    borderRadius: 10,
    border: "1px solid #111",
    background: "#111",
    color: "#fff",
    fontWeight: 800,
    cursor: "pointer",
  },
  hr: { height: 1, background: "#eee", margin: "14px 0" },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
  grid3: { display: "grid", gridTemplateColumns: "1fr 1fr 220px", gap: 12 },
  grid4: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr 220px", gap: 12 },
  tableWrap: {
    overflowX: "auto",
    border: "1px solid #eee",
    borderRadius: 10,
    background: "#fff",
  },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
  th: {
    textAlign: "left",
    padding: "10px 10px",
    borderBottom: "1px solid #eee",
    background: "#fafafa",
    position: "sticky",
    top: 0,
  },
  td: { padding: "8px 10px", borderBottom: "1px solid #f0f0f0", whiteSpace: "nowrap" },
  pill: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 999,
    background: "#efefef",
    border: "1px solid #e2e2e2",
    marginRight: 6,
    marginBottom: 6,
    fontSize: 12,
  },
  metaPill: {
    display: "inline-block",
    padding: "6px 10px",
    borderRadius: 999,
    background: "#f7f7f7",
    border: "1px solid #e7e7e7",
    fontSize: 12,
  },
  errorBox: {
    marginTop: 12,
    padding: 10,
    borderRadius: 10,
    background: "#fff1f1",
    border: "1px solid #ffd0d0",
    color: "#7a1111",
    fontSize: 13,
  },
  noteBox: {
    marginTop: 10,
    padding: 10,
    borderRadius: 10,
    background: "#f5fbff",
    border: "1px solid #d6efff",
    color: "#083a57",
    fontSize: 13,
  },
  interpretation: {
    marginTop: 10,
    padding: 10,
    borderRadius: 10,
    background: "#f7fff4",
    border: "1px solid #dbffd0",
    color: "#1e4d16",
    fontSize: 13,
  },
  img: {
    width: "100%",
    maxHeight: 520,
    objectFit: "contain",
    border: "1px solid #eee",
    borderRadius: 12,
    background: "#fff",
    padding: 6,
  },
  resultBlock: {
    marginTop: 12,
    padding: 12,
    border: "1px solid #eee",
    borderRadius: 12,
    background: "#fff",
  },
  resultBlockHeader: {
    fontSize: 14,
    fontWeight: 800,
    marginBottom: 8,
  },
  footer: { marginTop: 10, paddingBottom: 12 },
};
