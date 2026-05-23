import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowLeft, CheckCircle2, Trash2, Upload } from "lucide-react";
import { getVivaSenseMode, subscribeVivaSenseMode, type VivaSenseMode } from "@/lib/vivasenseGating";

export interface TrialMeta {
  trial_id: string;
  trial_name: string;
  design_type: string;
  created_at: string;
  n_plots: number;
  traits: string[];
  is_demo: boolean;
}

export interface PlotObservation {
  plot_id: number;
  rep: number;
  block: number | null;
  treatment: string;
  traits: Record<string, number | null>;
  notes: string;
  timestamp: string;
  is_complete: boolean;
}

type Screen = "trials" | "entry" | "review";

const STORAGE_TRIALS_KEY = "vivasense_trials";
const STORAGE_DATA_PREFIX = "vivasense_data_";

const REQUIRED_FIELDBOOK_COLUMNS = ["plot_id", "rep", "block", "treatment"];

function storageAvailable(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const testKey = "__vivasense_storage_test__";
    window.localStorage.setItem(testKey, "1");
    window.localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

function readJson<T>(key: string, fallback: T): T {
  try {
    if (!storageAvailable()) return fallback;
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as T;
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown): boolean {
  try {
    if (!storageAvailable()) return false;
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function seededShuffle<T>(items: T[], seed: number): T[] {
  let t = seed >>> 0;
  const next = () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };

  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(next() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function csvEscape(value: unknown): string {
  const text = String(value ?? "");
  if (/[,"\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function parseCsvRow(line: string): string[] {
  const out: string[] = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(cell.trim());
      cell = "";
    } else {
      cell += ch;
    }
  }
  out.push(cell.trim());
  return out;
}

function formatLastSaved(lastSaved: Date | null, nowMs: number): string {
  if (!lastSaved) return "Last saved: not yet";
  const diffMs = Math.max(0, nowMs - lastSaved.getTime());
  const sec = Math.floor(diffMs / 1000);
  if (sec < 10) return "Last saved: just now";
  if (sec < 60) return `Last saved: ${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `Last saved: ${min} min${min === 1 ? "" : "s"} ago`;
  const hr = Math.floor(min / 60);
  return `Last saved: ${hr} hr${hr === 1 ? "" : "s"} ago`;
}

function dataKey(trialId: string): string {
  return `${STORAGE_DATA_PREFIX}${trialId}`;
}

export function FieldDataCollection() {
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());
  const isPro = mode === "pro";

  const [screen, setScreen] = useState<Screen>("trials");
  const [isInitializingEntry, setIsInitializingEntry] = useState(false);
  const [trials, setTrials] = useState<TrialMeta[]>([]);
  const [currentTrial, setCurrentTrial] = useState<TrialMeta | null>(null);
  const [observations, setObservations] = useState<PlotObservation[]>([]);
  const [currentPlotIndex, setCurrentPlotIndex] = useState(0);
  const [currentTraitValues, setCurrentTraitValues] = useState<Record<string, string>>({});
  const [currentNotes, setCurrentNotes] = useState("");
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const [storageError, setStorageError] = useState<string | null>(null);
  const [globalMessage, setGlobalMessage] = useState<string>("");
  const [entrySuccess, setEntrySuccess] = useState<string>("");
  const [entryError, setEntryError] = useState<string>("");
  const [traitErrors, setTraitErrors] = useState<Record<string, string>>({});

  const [showNewTrialForm, setShowNewTrialForm] = useState(false);
  const [newTrialName, setNewTrialName] = useState("");
  const [newDesignType, setNewDesignType] = useState("RCBD");
  const [newTraitCount, setNewTraitCount] = useState(3);
  const [newTraits, setNewTraits] = useState<string[]>(["Yield", "Plant_Height", "Days_to_maturity"]);
  const [newCsvText, setNewCsvText] = useState("");
  const [newReps, setNewReps] = useState(3);
  const [newTreatmentsText, setNewTreatmentsText] = useState("Genotype_1, Genotype_2, Genotype_3, Genotype_4");
  const [newTrialError, setNewTrialError] = useState("");

  const [jumpToPlot, setJumpToPlot] = useState("");
  const [nowMs, setNowMs] = useState<number>(() => Date.now());

  useEffect(() => {
    const unsubscribe = subscribeVivaSenseMode(setMode);
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!storageAvailable()) {
      setStorageError("Storage unavailable. Please enable cookies or use a different browser.");
      return;
    }

    try {
      const loaded = getTrials();
      setTrials(loaded);
    } catch {
      setTrials([]);
    }
  }, []);

  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "You have unsaved data. Leave anyway?";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [hasUnsavedChanges]);

  useEffect(() => {
    if (!entrySuccess) return;
    const timer = window.setTimeout(() => setEntrySuccess(""), 2000);
    return () => window.clearTimeout(timer);
  }, [entrySuccess]);

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 30000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!currentTrial || observations.length === 0) {
      setCurrentTraitValues({});
      setCurrentNotes("");
      setTraitErrors({});
      return;
    }

    const obs = observations[currentPlotIndex];
    if (!obs) return;

    const values: Record<string, string> = {};
    currentTrial.traits.forEach((trait) => {
      const v = obs.traits[trait];
      values[trait] = v === null || Number.isNaN(v) ? "" : String(v);
    });

    setCurrentTraitValues(values);
    setCurrentNotes(obs.notes || "");
    setTraitErrors({});
  }, [currentPlotIndex, currentTrial, observations]);

  const currentObservation = observations[currentPlotIndex] ?? null;
  const totalPlots = observations.length;
  const completeCount = useMemo(
    () => observations.filter((o) => o.is_complete).length,
    [observations]
  );

  function createTrial(trial: TrialMeta): void {
    try {
      const existing = getTrials().filter((t) => t.trial_id !== trial.trial_id);
      const next = [trial, ...existing];
      const ok = writeJson(STORAGE_TRIALS_KEY, next);
      if (!ok) {
        setStorageError("Storage unavailable. Please enable cookies or use a different browser.");
        return;
      }
      setTrials(next);
    } catch {
      setStorageError("Storage unavailable. Please enable cookies or use a different browser.");
    }
  }

  function getTrials(): TrialMeta[] {
    try {
      const rows = readJson<TrialMeta[]>(STORAGE_TRIALS_KEY, []);
      if (!Array.isArray(rows)) return [];
      return rows.filter((row) => row && typeof row.trial_id === "string" && Array.isArray(row.traits));
    } catch {
      return [];
    }
  }

  function deleteTrial(trial_id: string): void {
    try {
      const next = getTrials().filter((t) => t.trial_id !== trial_id);
      writeJson(STORAGE_TRIALS_KEY, next);
      if (storageAvailable()) window.localStorage.removeItem(dataKey(trial_id));
      setTrials(next);
      if (currentTrial?.trial_id === trial_id) {
        setCurrentTrial(null);
        setObservations([]);
        setScreen("trials");
      }
    } catch {
      setStorageError("Storage unavailable. Please enable cookies or use a different browser.");
    }
  }

  function saveObservation(trial_id: string, observation: PlotObservation): void {
    try {
      const key = dataKey(trial_id);
      const rows = getObservations(trial_id);
      const idx = rows.findIndex((r) => r.plot_id === observation.plot_id);
      if (idx >= 0) rows[idx] = observation;
      else rows.push(observation);
      rows.sort((a, b) => a.plot_id - b.plot_id);

      const ok = writeJson(key, rows);
      if (!ok) {
        setStorageError("Storage unavailable. Please enable cookies or use a different browser.");
        return;
      }
      if (currentTrial?.trial_id === trial_id) setObservations(rows);
      setLastSaved(new Date());
    } catch {
      setStorageError("Storage unavailable. Please enable cookies or use a different browser.");
    }
  }

  function getObservations(trial_id: string): PlotObservation[] {
    try {
      const rows = readJson<PlotObservation[]>(dataKey(trial_id), []);
      if (!Array.isArray(rows)) return [];
      return rows
        .filter((row) => row && typeof row.plot_id === "number")
        .sort((a, b) => a.plot_id - b.plot_id);
    } catch {
      return [];
    }
  }

  function parseFieldbookCSV(csvText: string): PlotObservation[] {
    const cleanText = csvText.trim();
    if (!cleanText) return [];

    const lines = cleanText.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (lines.length < 2) {
      throw new Error("CSV appears empty. Include header and at least one row.");
    }

    const header = parseCsvRow(lines[0]).map((h) => h.toLowerCase());
    const missing = REQUIRED_FIELDBOOK_COLUMNS.filter((c) => !header.includes(c));
    if (missing.length > 0) {
      throw new Error(`Missing required CSV headers: ${missing.join(", ")}. Expected: plot_id, rep, block, treatment.`);
    }

    const col = {
      plot_id: header.indexOf("plot_id"),
      rep: header.indexOf("rep"),
      block: header.indexOf("block"),
      treatment: header.indexOf("treatment"),
    };

    const output: PlotObservation[] = [];
    for (let i = 1; i < lines.length; i += 1) {
      const row = parseCsvRow(lines[i]);
      const plot_id = Number(row[col.plot_id]);
      const rep = Number(row[col.rep]);
      const blockRaw = row[col.block];
      const block = blockRaw === "" || blockRaw == null ? null : Number(blockRaw);
      const treatment = String(row[col.treatment] ?? "").trim();

      if (!Number.isFinite(plot_id) || !Number.isFinite(rep) || !treatment) {
        continue;
      }

      output.push({
        plot_id,
        rep,
        block: Number.isFinite(block as number) ? (block as number) : null,
        treatment,
        traits: {},
        notes: "",
        timestamp: "",
        is_complete: false,
      });
    }

    output.sort((a, b) => a.plot_id - b.plot_id);
    return output;
  }

  function exportForAnalysis(trial_id: string): void {
    try {
      const trial = trials.find((t) => t.trial_id === trial_id);
      const rows = getObservations(trial_id);
      if (!trial || rows.length === 0) {
        setGlobalMessage("No data available to export.");
        return;
      }

      const traitColumns = Array.from(
        new Set(rows.flatMap((r) => Object.keys(r.traits || {})))
      );

      const header = ["Genotype", "Replication", "Block", ...traitColumns];
      const csvRows: string[] = [header.join(",")];

      rows.forEach((row) => {
        const values = [
          row.treatment,
          row.rep,
          row.block ?? "",
          ...traitColumns.map((t) => row.traits?.[t] ?? ""),
        ];
        csvRows.push(values.map(csvEscape).join(","));
      });

      const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${trial.trial_name.replace(/\s+/g, "_")}_analysis_ready.csv`;
      a.click();
      URL.revokeObjectURL(url);

      setGlobalMessage(
        "✓ CSV exported. Upload this file to VivaSense Statistical Analysis → Genetic Parameters for heritability, or → ANOVA for treatment comparisons."
      );
    } catch {
      setGlobalMessage("Could not export analysis CSV. Please try again.");
    }
  }

  function exportRawCSV(trial_id: string): void {
    try {
      const trial = trials.find((t) => t.trial_id === trial_id);
      const rows = getObservations(trial_id);
      if (!trial || rows.length === 0) {
        setGlobalMessage("No raw data available to export.");
        return;
      }

      const traitColumns = Array.from(
        new Set(rows.flatMap((r) => Object.keys(r.traits || {})))
      );

      const header = [
        "plot_id",
        "rep",
        "block",
        "treatment",
        ...traitColumns,
        "notes",
        "timestamp",
        "is_complete",
      ];

      const csvRows: string[] = [header.join(",")];
      rows.forEach((row) => {
        const values = [
          row.plot_id,
          row.rep,
          row.block ?? "",
          row.treatment,
          ...traitColumns.map((t) => row.traits?.[t] ?? ""),
          row.notes,
          row.timestamp,
          row.is_complete ? "1" : "0",
        ];
        csvRows.push(values.map(csvEscape).join(","));
      });

      const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${trial.trial_name.replace(/\s+/g, "_")}_raw.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setGlobalMessage("Could not export raw CSV. Please try again.");
    }
  }

  async function syncToCloud(trial_id: string): Promise<{ status: string }> {
    console.log("Cloud sync coming in VivaSense v2.", trial_id);
    return { status: "pending" };
  }

  function makeEmptyTraits(traits: string[]): Record<string, number | null> {
    return traits.reduce<Record<string, number | null>>((acc, trait) => {
      acc[trait] = null;
      return acc;
    }, {});
  }

  function loadTrial(trial: TrialMeta, target: Screen): void {
    const rows = getObservations(trial.trial_id);
    if (target === "entry") {
      setIsInitializingEntry(true);
    }
    setCurrentTrial(trial);
    setObservations(rows);
    setCurrentPlotIndex(0);
    setCurrentTraitValues({});
    setCurrentNotes("");
    setScreen(target);
    setGlobalMessage("");
    setEntryError("");
    if (target === "entry") {
      // Allow the entry UI a tick to mount, then clear the skeleton
      window.setTimeout(() => setIsInitializingEntry(false), 350);
    } else {
      setIsInitializingEntry(false);
    }
  }

  function confirmLeaveIfNeeded(): boolean {
    if (!hasUnsavedChanges) return true;
    return window.confirm("You have unsaved data. Leave anyway?");
  }

  function createDemoTrial(): void {
    const trial: TrialMeta = {
      trial_id: "demo_001",
      trial_name: "Cowpea Yield Trial (Demo)",
      design_type: "RCBD",
      created_at: new Date().toISOString(),
      n_plots: 15,
      traits: ["Yield_kg_ha", "Days_to_flowering", "100_seed_weight_g"],
      is_demo: true,
    };

    const genotypes = [
      "IT97K-499-35",
      "IT90K-277-2",
      "Ife-Brown",
      "TVu-7778",
      "SAMPEA-20",
    ];

    const baseTraits = makeEmptyTraits(trial.traits);
    const rows: PlotObservation[] = [];
    let plotId = 1;

    for (let rep = 1; rep <= 3; rep += 1) {
      const ordered = seededShuffle(genotypes, 42 + rep);
      ordered.forEach((treatment) => {
        rows.push({
          plot_id: plotId,
          rep,
          block: rep,
          treatment,
          traits: { ...baseTraits },
          notes: "",
          timestamp: "",
          is_complete: false,
        });
        plotId += 1;
      });
    }

    createTrial(trial);
    writeJson(dataKey(trial.trial_id), rows);
    setGlobalMessage("Demo trial loaded. You can now practice data entry safely.");
    loadTrial(trial, "trials");
    setCurrentTrial(null);
    setObservations([]);
    setScreen("trials");
  }

  function clearDemoData(): void {
    if (!currentTrial || !currentTrial.is_demo) return;
    const cleared = observations.map((obs) => ({
      ...obs,
      traits: makeEmptyTraits(currentTrial.traits),
      notes: "",
      timestamp: "",
      is_complete: false,
    }));
    writeJson(dataKey(currentTrial.trial_id), cleared);
    setObservations(cleared);
    setCurrentPlotIndex(0);
    setEntrySuccess("");
    setGlobalMessage("Demo data cleared. Trial structure preserved.");
  }

  function completeStatusFromValues(values: Record<string, string>): boolean {
    const nums = Object.values(values)
      .map((v) => (v.trim() === "" ? null : Number(v)))
      .filter((v) => v !== null) as number[];
    return nums.length > 0 && nums.every((v) => Number.isFinite(v));
  }

  function persistCurrent(values: Record<string, string>, notes: string): void {
    if (!currentTrial || !currentObservation) return;

    const numericTraits: Record<string, number | null> = {};
    currentTrial.traits.forEach((trait) => {
      const raw = values[trait] ?? "";
      const clean = raw.trim();
      numericTraits[trait] = clean === "" ? null : Number(clean);
      if (!Number.isFinite(numericTraits[trait] as number)) {
        numericTraits[trait] = null;
      }
    });

    const nextObs: PlotObservation = {
      ...currentObservation,
      traits: numericTraits,
      notes,
      timestamp: new Date().toISOString(),
      is_complete: completeStatusFromValues(values),
    };

    saveObservation(currentTrial.trial_id, nextObs);
    setHasUnsavedChanges(false);
  }

  function onTraitChange(trait: string, value: string): void {
    setEntryError("");
    setHasUnsavedChanges(true);

    const isValid = value.trim() === "" || /^-?\d*(\.\d+)?$/.test(value.trim());
    setTraitErrors((prev) => ({
      ...prev,
      [trait]: isValid ? "" : "Please enter a numeric value.",
    }));

    setCurrentTraitValues((prev) => {
      const next = { ...prev, [trait]: value };
      persistCurrent(next, currentNotes);
      return next;
    });
  }

  function onNotesChange(value: string): void {
    setCurrentNotes(value);
    setHasUnsavedChanges(true);
    persistCurrent(currentTraitValues, value);
  }

  function goNextPlot(): void {
    if (!currentTrial || !currentObservation) return;

    const hasAnyValue = Object.values(currentTraitValues).some((v) => v.trim() !== "");
    const hasTraitErrors = Object.values(traitErrors).some((e) => e);

    if (!hasAnyValue) {
      setEntryError("Please enter at least one trait value");
      return;
    }
    if (hasTraitErrors) {
      setEntryError("Please fix numeric validation errors before continuing.");
      return;
    }

    persistCurrent(currentTraitValues, currentNotes);

    const nextIndex = Math.min(currentPlotIndex + 1, Math.max(0, observations.length - 1));
    const nextPlotNo = nextIndex + 1;
    setEntrySuccess(`✓ Plot ${currentObservation.plot_id} saved — Plot ${nextPlotNo} of ${observations.length}`);
    setCurrentPlotIndex(nextIndex);
  }

  function goPrevPlot(): void {
    setCurrentPlotIndex((prev) => Math.max(0, prev - 1));
  }

  function skipPlot(): void {
    setEntryError("");
    setCurrentPlotIndex((prev) => Math.min(observations.length - 1, prev + 1));
  }

  function handleJumpToPlot(): void {
    const plotNo = Number(jumpToPlot);
    if (!Number.isFinite(plotNo) || plotNo < 1 || plotNo > observations.length) {
      setEntryError(`Enter a plot number between 1 and ${observations.length}.`);
      return;
    }
    setCurrentPlotIndex(plotNo - 1);
    setJumpToPlot("");
  }

  function progressForTrial(trial: TrialMeta): { complete: number; total: number; pct: number } {
    const rows = getObservations(trial.trial_id);
    const total = rows.length;
    const complete = rows.filter((r) => r.is_complete).length;
    const pct = total > 0 ? Math.round((complete / total) * 100) : 0;
    return { complete, total, pct };
  }

  function handleTrialDelete(trial: TrialMeta): void {
    if (!window.confirm(`Delete trial "${trial.trial_name}"? This removes all local observations.`)) return;
    deleteTrial(trial.trial_id);
  }

  function handleCreateTrial(): void {
    setNewTrialError("");
    const name = newTrialName.trim();
    if (!name) {
      setNewTrialError("Trial name is required.");
      return;
    }

    if (newTraitCount < 1 || newTraitCount > 10) {
      setNewTrialError("Number of traits must be between 1 and 10.");
      return;
    }

    const traits = newTraits
      .slice(0, newTraitCount)
      .map((t) => t.trim())
      .filter(Boolean);

    if (traits.length !== newTraitCount) {
      setNewTrialError("Provide all trait names.");
      return;
    }

    let importedRows: PlotObservation[] = [];
    const csvProvided = newCsvText.trim().length > 0;
    if (csvProvided) {
      try {
        importedRows = parseFieldbookCSV(newCsvText);
      } catch (error) {
        setNewTrialError(error instanceof Error ? error.message : "Invalid CSV file.");
        return;
      }
    } else {
      const treatments = newTreatmentsText
        .split(/[,\n]/)
        .map((t) => t.trim())
        .filter(Boolean);
      if (treatments.length < 2) {
        setNewTrialError("Provide at least 2 treatments (comma or newline separated), or upload a fieldbook CSV.");
        return;
      }
      if (newReps < 1 || newReps > 20) {
        setNewTrialError("Replications must be between 1 and 20.");
        return;
      }
      let plotId = 1;
      for (let rep = 1; rep <= newReps; rep += 1) {
        const ordered =
          newDesignType === "CRD" ? treatments : seededShuffle(treatments, 17 + rep);
        ordered.forEach((treatment) => {
          importedRows.push({
            plot_id: plotId,
            rep,
            block: newDesignType === "CRD" ? null : rep,
            treatment,
            traits: {},
            notes: "",
            timestamp: "",
            is_complete: false,
          });
          plotId += 1;
        });
      }
    }

    if (importedRows.length === 0) {
      setNewTrialError("No plots were generated. Provide treatments or upload a fieldbook CSV.");
      return;
    }

    const trialId = `trial_${Date.now()}`;
    const trial: TrialMeta = {
      trial_id: trialId,
      trial_name: name,
      design_type: newDesignType,
      created_at: new Date().toISOString(),
      n_plots: importedRows.length,
      traits,
      is_demo: false,
    };

    const template = makeEmptyTraits(traits);
    const rows = importedRows.map((row) => ({
      ...row,
      traits: { ...template },
      notes: "",
      timestamp: "",
      is_complete: false,
    }));

    createTrial(trial);
    writeJson(dataKey(trialId), rows);

    setShowNewTrialForm(false);
    setNewTrialName("");
    setNewTraitCount(3);
    setNewTraits(["Yield", "Plant_Height", "Days_to_maturity"]);
    setNewCsvText("");
    setGlobalMessage(`Trial "${name}" created with ${rows.length} plots.`);
    loadTrial(trial, "entry");
  }

  function handleCsvFileUpload(file: File | null): void {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      setNewCsvText(text);
    };
    reader.onerror = () => {
      setNewTrialError("Could not read CSV file.");
    };
    reader.readAsText(file);
  }

  const completionPct = totalPlots > 0 ? Math.round((completeCount / totalPlots) * 100) : 0;

  return (
    <div className="w-full rounded-2xl border border-[#D7EADF] bg-[#F5FBF8] p-3 sm:p-5 text-[#103328]">
      <div className="mb-4 rounded-xl border border-[#BFDCCD] bg-white p-4">
        <h2 className="text-xl sm:text-2xl font-bold">📋 Field Data Collection</h2>
        <p className="text-sm font-medium text-[#285A4B]">Beta — Offline</p>
        <p className="mt-1 text-xs text-[#285A4B]">Mode: {isPro ? "Pro" : "Free"} · This module is free for all users.</p>
      </div>

      {storageError && (
        <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800">
          {storageError}
        </div>
      )}

      {globalMessage && (
        <div className="mb-4 rounded-lg border border-emerald-300 bg-emerald-50 p-3 text-sm text-emerald-800">
          {globalMessage}
        </div>
      )}

      {screen === "trials" && (
        <section className="space-y-4">
          {trials.length === 0 ? (
            <div className="rounded-xl border border-dashed border-[#A8CDBE] bg-white p-4 text-sm text-[#285A4B]">
              No trials yet. Create a trial or load the demo trial.
            </div>
          ) : (
            <div className="space-y-3">
              {trials.map((trial) => {
                const p = progressForTrial(trial);
                return (
                  <article key={trial.trial_id} className="rounded-xl border border-[#D0E5DA] bg-white p-4">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <h3 className="text-base font-bold text-[#102F25]">{trial.trial_name}</h3>
                        <div className="mt-1 flex flex-wrap items-center gap-2">
                          <span className="inline-flex items-center rounded-full border border-[#0F6E56] bg-[#E8F4F1] px-2 py-0.5 text-xs font-semibold text-[#0F6E56]">
                            {trial.design_type}
                          </span>
                          {trial.is_demo && (
                            <span className="inline-flex items-center rounded-full bg-[#E8A832] px-2 py-0.5 text-xs font-semibold text-[#3D2A00]">
                              DEMO
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="mt-3">
                      <div className="mb-1 text-sm font-medium text-[#204E3F]">
                        Progress: {p.complete} / {p.total} plots recorded
                      </div>
                      <div className="h-2 w-full rounded-full bg-gray-200">
                        <div className="h-2 rounded-full bg-[#0F6E56]" style={{ width: `${p.pct}%` }} />
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
                      <button
                        type="button"
                        onClick={() => loadTrial(trial, "entry")}
                        className="min-h-11 rounded-lg bg-[#0F6E56] px-3 text-sm font-semibold text-white"
                      >
                        Continue →
                      </button>
                      <button
                        type="button"
                        onClick={() => loadTrial(trial, "review")}
                        className="min-h-11 rounded-lg border border-[#0F6E56] bg-white px-3 text-sm font-semibold text-[#0F6E56]"
                      >
                        View Data
                      </button>
                      <button
                        type="button"
                        onClick={() => handleTrialDelete(trial)}
                        className="min-h-11 rounded-lg border border-red-300 bg-red-50 px-3 text-sm font-semibold text-red-700"
                      >
                        🗑 Delete
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => setShowNewTrialForm((v) => !v)}
              className="min-h-11 rounded-lg bg-[#0F6E56] px-4 text-base font-semibold text-white"
            >
              ＋ New Trial
            </button>
            <button
              type="button"
              onClick={createDemoTrial}
              className="min-h-11 rounded-lg border border-[#E8A832] bg-[#FFF8E8] px-4 text-base font-semibold text-[#6A4C00]"
            >
              Load Demo Trial
            </button>
          </div>

          {showNewTrialForm && (
            <div className="rounded-xl border border-[#D0E5DA] bg-white p-4 space-y-3">
              <h3 className="text-base font-bold text-[#113429]">Create Trial</h3>

              <label className="block text-sm font-semibold text-[#1A4638]">
                Trial name
                <input
                  className="mt-1 min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
                  value={newTrialName}
                  onChange={(e) => setNewTrialName(e.target.value)}
                  placeholder="e.g., Maize Nitrogen Trial 2026"
                />
              </label>

              <label className="block text-sm font-semibold text-[#1A4638]">
                Design type
                <select
                  className="mt-1 min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
                  value={newDesignType}
                  onChange={(e) => setNewDesignType(e.target.value)}
                >
                  <option>CRD</option>
                  <option>RCBD</option>
                  <option>Latin Square</option>
                  <option>Split Plot</option>
                  <option>Factorial RCBD</option>
                  <option>Alpha Lattice</option>
                </select>
              </label>

              <label className="block text-sm font-semibold text-[#1A4638]">
                Number of traits (1-10)
                <input
                  type="number"
                  min={1}
                  max={10}
                  className="mt-1 min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
                  value={newTraitCount}
                  onChange={(e) => {
                    const count = Math.max(1, Math.min(10, Number(e.target.value) || 1));
                    setNewTraitCount(count);
                    setNewTraits((prev) => {
                      const next = [...prev];
                      while (next.length < count) next.push(`Trait_${next.length + 1}`);
                      return next.slice(0, count);
                    });
                  }}
                />
              </label>

              <div className="space-y-2">
                <p className="text-sm font-semibold text-[#1A4638]">Trait names</p>
                {Array.from({ length: newTraitCount }).map((_, idx) => (
                  <input
                    key={`trait-${idx}`}
                    className="min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
                    value={newTraits[idx] ?? ""}
                    onChange={(e) => {
                      const value = e.target.value;
                      setNewTraits((prev) => {
                        const next = [...prev];
                        next[idx] = value;
                        return next;
                      });
                    }}
                    placeholder={`Trait ${idx + 1}`}
                  />
                ))}
              </div>

              <label className="block text-sm font-semibold text-[#1A4638]">
                Replications
                <input
                  type="number"
                  min={1}
                  max={20}
                  className="mt-1 min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
                  value={newReps}
                  onChange={(e) => setNewReps(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                />
              </label>

              <label className="block text-sm font-semibold text-[#1A4638]">
                Treatments / Genotypes (comma or newline separated)
                <textarea
                  className="mt-1 min-h-20 w-full rounded-lg border border-[#BFDCCD] px-3 py-2 text-base"
                  value={newTreatmentsText}
                  onChange={(e) => setNewTreatmentsText(e.target.value)}
                  placeholder="e.g., SAMMAZ-15, SAMMAZ-16, SAMMAZ-17, SAMMAZ-18"
                />
                <span className="mt-1 block text-xs font-normal text-[#5A7A6E]">
                  Plots auto-generate as Replications × Treatments. Skip this if uploading a fieldbook CSV below.
                </span>
              </label>

              <div className="space-y-2">

                <label className="inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-lg border border-[#BFDCCD] bg-[#F8FCFA] px-3 text-sm font-semibold text-[#1F4A3B]">
                  <Upload className="h-4 w-4" />
                  Choose CSV file
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    className="hidden"
                    onChange={(e) => handleCsvFileUpload(e.target.files?.[0] ?? null)}
                  />
                </label>
                <textarea
                  className="min-h-24 w-full rounded-lg border border-[#BFDCCD] px-3 py-2 text-base"
                  value={newCsvText}
                  onChange={(e) => setNewCsvText(e.target.value)}
                  placeholder="Or paste CSV with headers: plot_id, rep, block, treatment"
                />
              </div>

              {newTrialError && (
                <div className="rounded-lg border border-red-300 bg-red-50 p-2 text-sm text-red-700">
                  {newTrialError}
                </div>
              )}

              <button
                type="button"
                onClick={handleCreateTrial}
                className="min-h-11 w-full rounded-lg bg-[#0F6E56] px-4 text-base font-semibold text-white"
              >
                Create Trial
              </button>
            </div>
          )}
        </section>
      )}

      {screen === "entry" && (isInitializingEntry || !currentTrial || !currentObservation) && (
        <section className="space-y-4" aria-busy="true" aria-live="polite">
          <div className="rounded-xl border border-[#D0E5DA] bg-white p-4">
            <div className="flex items-center justify-between gap-2">
              <div className="h-10 w-36 animate-pulse rounded-lg bg-[#E8F1EC]" />
              <div className="h-5 w-16 animate-pulse rounded-full bg-[#E8F1EC]" />
            </div>
            <div className="mt-4 h-6 w-2/3 animate-pulse rounded bg-[#E8F1EC]" />
            <div className="mt-2 h-4 w-1/3 animate-pulse rounded bg-[#EEF4F0]" />
            <div className="mt-4 h-2 w-full animate-pulse rounded-full bg-[#EEF4F0]" />
          </div>
          <div className="rounded-xl border border-[#D0E5DA] bg-white p-4 space-y-3">
            <div className="h-5 w-1/3 animate-pulse rounded bg-[#E8F1EC]" />
            {[0, 1, 2].map((i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 w-1/4 animate-pulse rounded bg-[#EEF4F0]" />
                <div className="h-11 w-full animate-pulse rounded-lg bg-[#E8F1EC]" />
              </div>
            ))}
            <div className="flex items-center gap-2 pt-2">
              <div className="h-11 w-28 animate-pulse rounded-lg bg-[#E8F1EC]" />
              <div className="h-11 w-28 animate-pulse rounded-lg bg-[#EEF4F0]" />
            </div>
            <p className="text-center text-xs font-medium text-[#5A7A6E] pt-1">
              Preparing plot-by-plot entry…
            </p>
          </div>
        </section>
      )}

      {screen === "entry" && !isInitializingEntry && currentTrial && currentObservation && (
        <section className="space-y-4">
          <div className="rounded-xl border border-[#D0E5DA] bg-white p-4">
            <div className="flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => {
                  if (!confirmLeaveIfNeeded()) return;
                  setScreen("trials");
                }}
                className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-[#BFDCCD] px-3 text-sm font-semibold text-[#224D3F]"
              >
                <ArrowLeft className="h-4 w-4" />
                Trial Manager
              </button>
              {currentTrial.is_demo && (
                <span className="inline-flex items-center rounded-full bg-[#E8A832] px-2 py-0.5 text-xs font-semibold text-[#3D2A00]">
                  DEMO
                </span>
              )}
            </div>

            <h3 className="mt-3 text-lg font-bold text-[#102F25]">{currentTrial.trial_name}</h3>
            <p className="text-sm font-medium text-[#285A4B]">
              Progress: Plot {currentPlotIndex + 1} of {observations.length}
            </p>
            <div className="mt-2 h-2 w-full rounded-full bg-gray-200">
              <div className="h-2 rounded-full bg-[#0F6E56]" style={{ width: `${completionPct}%` }} />
            </div>
          </div>

          {entrySuccess && (
            <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-3 text-sm font-semibold text-emerald-800">
              {entrySuccess}
            </div>
          )}

          {entryError && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm font-semibold text-red-700">
              {entryError}
            </div>
          )}

          <div className="rounded-xl border-2 border-[#0F6E56] bg-white p-4">
            <p className="text-[24px] leading-8 font-bold text-[#0F6E56]">PLOT {currentObservation.plot_id}</p>
            <p className="mt-1 text-base font-semibold text-[#103328]">
              Rep: {currentObservation.rep} | Block: {currentObservation.block ?? "-"} | Treatment: {currentObservation.treatment}
            </p>
          </div>

          <div className="rounded-xl border border-[#D0E5DA] bg-white p-4 space-y-3">
            {currentTrial.traits.map((trait) => (
              <label key={trait} className="block text-sm font-semibold text-[#173F33]">
                {trait}
                <input
                  type="number"
                  inputMode="decimal"
                  className="mt-1 min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
                  placeholder="Enter value"
                  value={currentTraitValues[trait] ?? ""}
                  onChange={(e) => onTraitChange(trait, e.target.value)}
                />
                {traitErrors[trait] && (
                  <span className="mt-1 block text-xs text-red-700">{traitErrors[trait]}</span>
                )}
              </label>
            ))}

            <label className="block text-sm font-semibold text-[#173F33]">
              Notes
              <textarea
                className="mt-1 min-h-24 w-full rounded-lg border border-[#BFDCCD] px-3 py-2 text-base"
                placeholder="Optional field observations..."
                value={currentNotes}
                onChange={(e) => onNotesChange(e.target.value)}
              />
            </label>
          </div>

          <button
            type="button"
            onClick={goNextPlot}
            className="min-h-11 w-full rounded-lg bg-[#0F6E56] px-4 text-base font-semibold text-white"
          >
            Save & Next Plot →
          </button>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <button
              type="button"
              onClick={goPrevPlot}
              className="min-h-11 rounded-lg border border-[#BFDCCD] bg-white px-4 text-base font-semibold text-[#1F4A3B]"
            >
              ← Back
            </button>
            <button
              type="button"
              onClick={skipPlot}
              className="min-h-11 rounded-lg border border-[#BFDCCD] bg-white px-4 text-base font-semibold text-[#1F4A3B]"
            >
              Skip Plot
            </button>
            <div className="flex gap-2">
              <input
                value={jumpToPlot}
                onChange={(e) => setJumpToPlot(e.target.value)}
                placeholder="Plot #"
                className="min-h-11 w-full rounded-lg border border-[#BFDCCD] px-3 text-base"
              />
              <button
                type="button"
                onClick={handleJumpToPlot}
                className="min-h-11 rounded-lg border border-[#0F6E56] bg-white px-3 text-sm font-semibold text-[#0F6E56]"
              >
                Jump
              </button>
            </div>
          </div>

          <p className="text-xs font-medium text-[#2A5A4B]">{formatLastSaved(lastSaved, nowMs)}</p>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => {
                if (!confirmLeaveIfNeeded()) return;
                setScreen("review");
              }}
              className="min-h-11 rounded-lg border border-[#0F6E56] bg-white px-4 text-base font-semibold text-[#0F6E56]"
            >
              Review & Export
                      <p className="text-xs font-medium text-[#2A5A4B]">{formatLastSaved(lastSaved, nowMs)}</p>
            </button>
            <button
              type="button"
              onClick={() => {
                if (!currentTrial.is_demo) return;
                clearDemoData();
              }}
              className="min-h-11 rounded-lg border border-[#E8A832] bg-[#FFF8E8] px-4 text-base font-semibold text-[#6A4C00]"
            >
              Clear Demo Data
            </button>
          </div>
        </section>
      )}

      {screen === "review" && currentTrial && (
        <section className="space-y-4">
          <div className="rounded-xl border border-[#D0E5DA] bg-white p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => setScreen("trials")}
                className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-[#BFDCCD] px-3 text-sm font-semibold text-[#224D3F]"
              >
                <ArrowLeft className="h-4 w-4" />
                Trial Manager
              </button>
              {currentTrial.is_demo && (
                <span className="inline-flex items-center rounded-full bg-[#E8A832] px-2 py-0.5 text-xs font-semibold text-[#3D2A00]">
                  DEMO
                </span>
              )}
            </div>
            <h3 className="mt-3 text-xl font-bold text-[#102F25]">{currentTrial.trial_name}</h3>
            <p className="text-sm font-medium text-[#285A4B]">
              {completeCount} of {totalPlots} plots complete ({completionPct}%)
            </p>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[#D0E5DA] bg-white">
            <table className="min-w-full text-sm">
              <thead className="bg-[#EEF7F3] text-left text-[#153E31]">
                <tr>
                  <th className="px-3 py-2">Plot ID</th>
                  <th className="px-3 py-2">Rep</th>
                  <th className="px-3 py-2">Block</th>
                  <th className="px-3 py-2">Treatment</th>
                  {currentTrial.traits.map((trait) => (
                    <th key={trait} className="px-3 py-2">{trait}</th>
                  ))}
                  <th className="px-3 py-2">Complete</th>
                </tr>
              </thead>
              <tbody>
                {observations.map((row) => (
                  <tr
                    key={row.plot_id}
                    className={row.is_complete ? "bg-[#EAF3DE]" : "bg-[#FFF8E8]"}
                  >
                    <td className="px-3 py-2 font-semibold">{row.plot_id}</td>
                    <td className="px-3 py-2">{row.rep}</td>
                    <td className="px-3 py-2">{row.block ?? "-"}</td>
                    <td className="px-3 py-2 font-medium">{row.treatment}</td>
                    {currentTrial.traits.map((trait) => (
                      <td key={`${row.plot_id}-${trait}`} className="px-3 py-2">
                        {row.traits[trait] ?? ""}
                      </td>
                    ))}
                    <td className="px-3 py-2">
                      {row.is_complete ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-800">
                          <CheckCircle2 className="h-3.5 w-3.5" /> Yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-900">
                          <AlertTriangle className="h-3.5 w-3.5" /> No
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => exportForAnalysis(currentTrial.trial_id)}
              className="min-h-11 rounded-lg bg-[#0F6E56] px-4 text-base font-semibold text-white"
            >
              📥 Export for Analysis (CSV)
            </button>
            <button
              type="button"
              onClick={() => exportRawCSV(currentTrial.trial_id)}
              className="min-h-11 rounded-lg border border-[#0F6E56] bg-white px-4 text-base font-semibold text-[#0F6E56]"
            >
              📥 Export Raw CSV
            </button>
          </div>

          {currentTrial.is_demo && (
            <button
              type="button"
              onClick={clearDemoData}
              className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-[#E8A832] bg-[#FFF8E8] px-4 text-base font-semibold text-[#6A4C00]"
            >
              <Trash2 className="h-4 w-4" />
              ⚠ Clear Demo Data
            </button>
          )}

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setScreen("entry")}
              className="min-h-11 rounded-lg border border-[#0F6E56] bg-white px-4 text-base font-semibold text-[#0F6E56]"
            >
              Return to Data Entry
            </button>
            <button
              type="button"
              onClick={() => {
                if (!currentTrial) return;
                void syncToCloud(currentTrial.trial_id).then(() => {
                  setGlobalMessage("Cloud sync coming in VivaSense v2.");
                });
              }}
              className="min-h-11 rounded-lg border border-[#BFDCCD] bg-white px-4 text-base font-semibold text-[#1F4A3B]"
            >
              Sync to Cloud (Coming Soon)
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

export default FieldDataCollection;
