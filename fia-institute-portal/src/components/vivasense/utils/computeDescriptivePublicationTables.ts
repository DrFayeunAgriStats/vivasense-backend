/**
 * Compute publication-ready tables from multi-trait descriptive analysis results.
 * Generates: overall descriptive stats, per-trait genotype performance, multi-trait ranking.
 */

export interface OverallDescRow {
  trait: string;
  mean: number;
  minimum: number;
  maximum: number;
  standard_deviation: number;
  cv_percent: number;
}

export interface GenotypePerformanceRow {
  genotype: string;
  mean: number;
  standard_deviation: number;
  cv_percent: number;
  rank: number;
  performance_class: string;
}

export interface MultiTraitRankingRow {
  genotype: string;
  trait_ranks: Record<string, number>;
  rank_sum: number;
  overall_assessment: string;
}

export interface PublicationTablesData {
  overall_descriptive_statistics: OverallDescRow[];
  trait_performance_tables: Record<string, GenotypePerformanceRow[]>;
  multi_trait_ranking: MultiTraitRankingRow[];
  trait_names: string[];
}

// Traits where lower is better
const LOWER_IS_BETTER_PATTERNS = [
  /days?\s*(to|of)\s*(flower|maturity|heading|anthesis|silk)/i,
  /duration/i,
  /lodging/i,
  /disease/i,
  /pest/i,
  /insect/i,
  /mortality/i,
  /infection/i,
];

function isLowerBetter(traitName: string): boolean {
  return LOWER_IS_BETTER_PATTERNS.some((p) => p.test(traitName));
}

function getPerformanceClass(rank: number, total: number): string {
  if (total <= 1) return "—";
  const pct = (rank - 1) / (total - 1); // 0 = best, 1 = worst
  if (pct <= 0.05) return "Best";
  if (pct <= 0.25) return "Strong";
  if (pct <= 0.55) return "Intermediate";
  if (pct <= 0.8) return "Below average";
  return "Lowest";
}

function getOverallAssessment(rankSum: number, minRankSum: number, maxRankSum: number): string {
  if (maxRankSum === minRankSum) return "Intermediate";
  const pct = (rankSum - minRankSum) / (maxRankSum - minRankSum);
  if (pct <= 0.1) return "Superior across traits";
  if (pct <= 0.3) return "Consistently strong";
  if (pct <= 0.6) return "Intermediate";
  if (pct <= 0.85) return "Below average";
  return "Weak across traits";
}

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdDev(arr: number[]): number {
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((sum, v) => sum + (v - m) ** 2, 0) / (arr.length - 1));
}

/**
 * Extract numeric values for a trait from per_trait data.
 * Handles multiple backend shapes: descriptive_stats with group data, or raw data arrays.
 */
function extractGenotypeValues(
  traitData: Record<string, unknown>,
  allData?: Record<string, unknown>[]
): { genotype: string; values: number[] }[] | null {
  // Shape 1: descriptive_stats has group-level data (e.g., { overall: {...}, Genotype: { G1: {mean, std, ...} } })
  const descStats = traitData.descriptive_stats as Record<string, unknown> | undefined;
  if (descStats) {
    const groupKey = Object.keys(descStats).find((k) => k !== "overall");
    if (groupKey) {
      const groups = descStats[groupKey] as Record<string, Record<string, unknown>>;
      if (groups && typeof groups === "object") {
        return Object.entries(groups).map(([name, stats]) => ({
          genotype: name,
          values: [], // We'll use pre-computed stats
          _stats: stats, // Internal: pre-computed
        })) as any;
      }
    }
  }

  // Shape 2: tables.means has group means
  const tables = traitData.tables as Record<string, unknown> | undefined;
  if (tables?.means) {
    const meansData = tables.means as Record<string, unknown>;
    const treatKey = Object.keys(meansData)[0];
    const meansMap = (meansData[treatKey] ?? meansData) as Record<string, number>;
    return Object.entries(meansMap).map(([name, m]) => ({
      genotype: name,
      values: [m], // Single mean value
    }));
  }

  return null;
}

function getTraitOverallStats(traitData: Record<string, unknown>): {
  mean: number; min: number; max: number; std: number; cv: number;
} | null {
  const descStats = traitData.descriptive_stats as Record<string, unknown> | undefined;
  const overall = descStats?.overall as Record<string, unknown> | undefined;
  if (overall) {
    return {
      mean: Number(overall.mean ?? 0),
      min: Number(overall.min ?? 0),
      max: Number(overall.max ?? 0),
      std: Number(overall.std ?? 0),
      cv: Number(overall.cv ?? 0),
    };
  }
  return null;
}

export function computeDescriptivePublicationTables(
  results: Record<string, unknown>
): PublicationTablesData | null {
  const perTrait = (results.per_trait || results.trait_results) as Record<string, Record<string, unknown>> | undefined;
  if (!perTrait || typeof perTrait !== "object") return null;

  const traitNames = Object.keys(perTrait);
  if (traitNames.length === 0) return null;

  // 1. Overall descriptive statistics
  const overallStats: OverallDescRow[] = [];
  traitNames.forEach((trait) => {
    const stats = getTraitOverallStats(perTrait[trait]);
    if (stats) {
      overallStats.push({
        trait,
        mean: stats.mean,
        minimum: stats.min,
        maximum: stats.max,
        standard_deviation: stats.std,
        cv_percent: stats.cv,
      });
    }
  });

  if (overallStats.length === 0) return null;

  // 2. Per-trait genotype performance tables
  const traitPerformance: Record<string, GenotypePerformanceRow[]> = {};
  const genotypeTraitRanks: Record<string, Record<string, number>> = {};

  traitNames.forEach((trait) => {
    const genoData = extractGenotypeValues(perTrait[trait]);
    if (!genoData || genoData.length === 0) return;

    const lowerBetter = isLowerBetter(trait);

    // Build performance rows
    const rows: { genotype: string; mean: number; sd: number; cv: number }[] = genoData.map((g) => {
      const stats = (g as any)._stats as Record<string, unknown> | undefined;
      if (stats) {
        const m = Number(stats.mean ?? 0);
        const sd = Number(stats.std ?? 0);
        return { genotype: g.genotype, mean: m, sd, cv: m !== 0 ? (sd / m) * 100 : 0 };
      }
      // Fallback: single value
      const m = g.values.length > 0 ? mean(g.values) : 0;
      const sd = g.values.length > 1 ? stdDev(g.values) : 0;
      return { genotype: g.genotype, mean: m, sd, cv: m !== 0 ? (sd / m) * 100 : 0 };
    });

    // Sort by performance (best first)
    rows.sort((a, b) => lowerBetter ? a.mean - b.mean : b.mean - a.mean);

    const total = rows.length;
    const performanceRows: GenotypePerformanceRow[] = rows.map((r, i) => {
      const rank = i + 1;
      if (!genotypeTraitRanks[r.genotype]) genotypeTraitRanks[r.genotype] = {};
      genotypeTraitRanks[r.genotype][trait] = rank;
      return {
        genotype: r.genotype,
        mean: r.mean,
        standard_deviation: r.sd,
        cv_percent: r.cv,
        rank,
        performance_class: getPerformanceClass(rank, total),
      };
    });

    traitPerformance[trait] = performanceRows;
  });

  // 3. Multi-trait ranking
  const allGenotypes = Object.keys(genotypeTraitRanks);
  if (allGenotypes.length === 0) {
    return {
      overall_descriptive_statistics: overallStats,
      trait_performance_tables: traitPerformance,
      multi_trait_ranking: [],
      trait_names: traitNames,
    };
  }

  const rankedTraits = Object.keys(traitPerformance);
  const rankings: MultiTraitRankingRow[] = allGenotypes.map((geno) => {
    const ranks = genotypeTraitRanks[geno];
    const rankSum = rankedTraits.reduce((sum, t) => sum + (ranks[t] ?? allGenotypes.length), 0);
    return {
      genotype: geno,
      trait_ranks: ranks,
      rank_sum: rankSum,
      overall_assessment: "", // filled below
    };
  });

  rankings.sort((a, b) => a.rank_sum - b.rank_sum);
  const minRS = rankings[0]?.rank_sum ?? 0;
  const maxRS = rankings[rankings.length - 1]?.rank_sum ?? 0;
  rankings.forEach((r) => {
    r.overall_assessment = getOverallAssessment(r.rank_sum, minRS, maxRS);
  });

  return {
    overall_descriptive_statistics: overallStats,
    trait_performance_tables: traitPerformance,
    multi_trait_ranking: rankings,
    trait_names: traitNames,
  };
}

/**
 * Convert publication tables data to Word-exportable HTML tables.
 */
export function publicationTablesToHtml(data: PublicationTablesData): Record<string, string> {
  const tables: Record<string, string> = {};
  let tableNum = 1;

  // Table 1: Overall descriptive stats
  {
    let html = `<table>\n<caption>Table ${tableNum}. Overall descriptive statistics of the evaluated traits</caption>\n`;
    html += `<thead><tr><th>Trait</th><th>Mean</th><th>Minimum</th><th>Maximum</th><th>Standard Deviation</th><th>CV (%)</th></tr></thead>\n<tbody>\n`;
    data.overall_descriptive_statistics.forEach((r) => {
      html += `<tr><td>${r.trait}</td><td>${r.mean.toFixed(2)}</td><td>${r.minimum.toFixed(2)}</td><td>${r.maximum.toFixed(2)}</td><td>${r.standard_deviation.toFixed(2)}</td><td>${r.cv_percent.toFixed(1)}</td></tr>\n`;
    });
    html += `</tbody></table>`;
    tables[`table_${tableNum}_overall_descriptive`] = html;
    tableNum++;
  }

  // Per-trait genotype tables
  Object.entries(data.trait_performance_tables).forEach(([trait, rows]) => {
    if (rows.length === 0) return;
    let html = `<table>\n<caption>Table ${tableNum}. Descriptive performance of genotypes for ${trait}</caption>\n`;
    html += `<thead><tr><th>Genotype</th><th>Mean</th><th>Standard Deviation</th><th>CV (%)</th><th>Rank</th><th>Performance Class</th></tr></thead>\n<tbody>\n`;
    rows.forEach((r) => {
      html += `<tr><td>${r.genotype}</td><td>${r.mean.toFixed(2)}</td><td>${r.standard_deviation.toFixed(2)}</td><td>${r.cv_percent.toFixed(1)}</td><td>${r.rank}</td><td>${r.performance_class}</td></tr>\n`;
    });
    html += `</tbody></table>`;
    tables[`table_${tableNum}_${trait.replace(/\s+/g, "_").toLowerCase()}`] = html;
    tableNum++;
  });

  // Multi-trait ranking table
  if (data.multi_trait_ranking.length > 0) {
    const rankedTraits = data.trait_names.filter((t) => data.trait_performance_tables[t]?.length > 0);
    let html = `<table>\n<caption>Table ${tableNum}. Multi-trait ranking of evaluated genotypes</caption>\n`;
    html += `<thead><tr><th>Genotype</th>`;
    rankedTraits.forEach((t) => { html += `<th>${t} Rank</th>`; });
    html += `<th>Rank Sum</th><th>Overall Assessment</th></tr></thead>\n<tbody>\n`;
    data.multi_trait_ranking.forEach((r) => {
      html += `<tr><td>${r.genotype}</td>`;
      rankedTraits.forEach((t) => { html += `<td>${r.trait_ranks[t] ?? "—"}</td>`; });
      html += `<td>${r.rank_sum}</td><td>${r.overall_assessment}</td></tr>\n`;
    });
    html += `</tbody></table>`;
    tables[`table_${tableNum}_multi_trait_ranking`] = html;
  }

  return tables;
}
