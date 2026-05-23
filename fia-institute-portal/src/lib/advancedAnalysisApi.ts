/**
 * API client for VivaSense Advanced Analysis endpoints.
 * Backend: unified Docker engine (GENETICS_API_BASE).
 */

import { GENETICS_API_BASE } from "@/config/vivasense";
import type {
  StabilityRequest,
  StabilityResponse,
  StabilityMethod,
  GgeBiplotType,
  BlupRequest,
  BlupResponse,
  PcaRequest,
  PcaResponse,
  ClusterRequest,
  ClusterResponse,
  NonparametricRequest,
  NonparametricResponse,
  NonparametricTestType,
  ManovaRequest,
  ManovaResponse,
  ManovaTestStat,
  PathAnalysisRequest,
  PathAnalysisResponse,
  PathAnalysisMethod,
  SelectionIndexRequest,
  SelectionIndexResponse,
} from "@/types/advancedAnalysis";

const BASE = GENETICS_API_BASE;

async function postJson<TReq, TRes>(path: string, body: TReq, label: string): Promise<TRes> {
  const url = `${BASE}${path}`;
  console.log(`[MODULE] ${label}`);
  console.log(`[REQUEST] ${label}`, url, body);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      console.error(`[${label} ERROR]`, res.status, text);
      let detail = `${label} failed (${res.status})`;
      try {
        const parsed = JSON.parse(text);
        detail = parsed.detail || parsed.message || detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    const parsed = (await res.json()) as TRes;
    if (label === "stability") {
      const response = { data: parsed };
      console.log("FULL STABILITY RESPONSE", response);
      console.log("AMMI DATA", response.data);
      console.log("AVAILABLE KEYS", Object.keys((response.data as Record<string, unknown>) || {}));
    }
    return parsed;
  } catch (e) {
    if ((e as Error).name === "AbortError") {
      throw new Error(`${label} timed out. The backend may be cold-starting — please retry.`);
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
}

// ── Dedicated per-endpoint payload builders ────────────────────────────────
// Do not pass UI form-state directly to fetch; always build through these.

export function buildStabilityPayload(args: {
  datasetToken: string;
  trait: string;
  methods?: StabilityMethod[];
  biplotType?: GgeBiplotType;
  ammiComponents?: number;
}): StabilityRequest {
  const payload: StabilityRequest = {
    dataset_token: args.datasetToken,
    trait_column: args.trait,
  };
  if (args.methods?.length) payload.methods = args.methods;
  if (args.biplotType) payload.biplot_type = args.biplotType;
  if (args.ammiComponents != null) payload.ammi_components = args.ammiComponents;
  return payload;
}

export function buildBlupPayload(args: {
  datasetToken: string;
  trait: string;
  randomEffects?: string[];
  fixedEffects?: string[];
}): BlupRequest {
  return {
    dataset_token: args.datasetToken,
    trait_column: args.trait,
    random_effects: args.randomEffects?.length ? args.randomEffects : ["genotype"],
    fixed_effects: args.fixedEffects ?? [],
  };
}

export function buildPcaPayload(args: {
  datasetToken: string;
  traits: string[];
  scale?: boolean;
  nComponents?: number | null;
}): PcaRequest {
  return {
    dataset_token: args.datasetToken,
    trait_columns: args.traits,
    scale: args.scale ?? true,
    n_components: args.nComponents ?? null,
  };
}

export function buildClusterPayload(args: {
  datasetToken: string;
  traits: string[];
  method?: "single" | "complete" | "average" | "ward";
  k?: number | null;
  scale?: boolean;
}): ClusterRequest {
  return {
    dataset_token: args.datasetToken,
    trait_columns: args.traits,
    method: args.method ?? "ward",
    k: args.k ?? null,
    scale: args.scale ?? true,
  };
}

export const runStability = (req: StabilityRequest) =>
  postJson<StabilityRequest, StabilityResponse>("/analysis/stability", req, "stability");

export const runBlup = (req: BlupRequest) =>
  postJson<BlupRequest, BlupResponse>("/analysis/blup", req, "blup");

export const runPca = (req: PcaRequest) =>
  postJson<PcaRequest, PcaResponse>("/analysis/pca", req, "pca");

export const runCluster = (req: ClusterRequest) =>
  postJson<ClusterRequest, ClusterResponse>("/analysis/cluster", req, "cluster");

// ── Phase 2 builders ───────────────────────────────────────────────────────

export function buildNonparametricPayload(args: {
  datasetToken: string;
  traitColumn: string;
  groupColumn: string;
  testType: NonparametricTestType;
  blockColumn?: string;
  alpha?: number;
}): NonparametricRequest {
  const payload: NonparametricRequest = {
    dataset_token: args.datasetToken,
    trait_column: args.traitColumn,
    group_column: args.groupColumn,
    test_type: args.testType,
    alpha: args.alpha ?? 0.05,
  };
  if (args.blockColumn) payload.block_column = args.blockColumn;
  return payload;
}

export function buildManovaPayload(args: {
  datasetToken: string;
  traitColumns: string[];
  factorColumn: string;
  covariates?: string[];
  testStatistic?: ManovaTestStat;
  alpha?: number;
}): ManovaRequest {
  return {
    dataset_token: args.datasetToken,
    trait_columns: args.traitColumns,
    factor_column: args.factorColumn,
    covariates: args.covariates ?? [],
    test_statistic: args.testStatistic ?? "Wilks",
    alpha: args.alpha ?? 0.05,
  };
}

export function buildPathAnalysisPayload(args: {
  datasetToken: string;
  outcomeTrait: string;
  predictorTraits: string[];
  method?: PathAnalysisMethod;
  standardize?: boolean;
}): PathAnalysisRequest {
  return {
    dataset_token: args.datasetToken,
    outcome_trait: args.outcomeTrait,
    predictor_traits: args.predictorTraits,
    method: args.method ?? "correlation",
    standardize: args.standardize ?? true,
  };
}

export function buildSelectionIndexPayload(args: {
  datasetToken: string;
  traitColumns: string[];
  economicWeights: Record<string, number>;
  geneticParameters?: Record<string, number>;
  geneticCorrelations?: number[][];
  selectionIntensity?: number;
}): SelectionIndexRequest {
  const payload: SelectionIndexRequest = {
    dataset_token: args.datasetToken,
    trait_columns: args.traitColumns,
    economic_weights: args.economicWeights,
    selection_intensity: args.selectionIntensity ?? 1.755,
  };
  if (args.geneticParameters) payload.genetic_parameters = args.geneticParameters;
  if (args.geneticCorrelations) payload.genetic_correlations = args.geneticCorrelations;
  return payload;
}

export const runNonparametric = (req: NonparametricRequest) =>
  postJson<NonparametricRequest, NonparametricResponse>("/analysis/nonparametric", req, "nonparametric");

export const runManova = (req: ManovaRequest) =>
  postJson<ManovaRequest, ManovaResponse>("/analysis/manova", req, "manova");

export const runPathAnalysis = (req: PathAnalysisRequest) =>
  postJson<PathAnalysisRequest, PathAnalysisResponse>("/analysis/path-analysis", req, "path-analysis");

export const runSelectionIndex = (req: SelectionIndexRequest) =>
  postJson<SelectionIndexRequest, SelectionIndexResponse>("/analysis/selection-index", req, "selection-index");
