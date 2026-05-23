export interface GgePoint {
  genotype: string;
  mean: number;
  pc1: number;
  pc2: number;
}

export interface EnvironmentPoint {
  environment: string;
  mean: number;
  pc1: number;
  pc2: number;
}

export interface Vec2 {
  x: number;
  y: number;
}

export interface AECModel {
  origin: Vec2;
  avgEnvironment: Vec2;
  abscissaUnit: Vec2;
  ordinateUnit: Vec2;
}

export interface GenotypeProjection {
  genotype: string;
  mean: number;
  point: Vec2;
  projectionPoint: Vec2;
  abscissaScore: number;
  ordinateScore: number;
  stabilityDistance: number;
}

export interface IdealRanking {
  genotype: string;
  distance_from_ideal: number;
  rank: number;
}

export interface IdealGenotypeModel {
  idealPoint: Vec2;
  idealAbscissaScore: number;
  idealGenotype: string;
  rankings: IdealRanking[];
  ringRadii: number[];
}

const EPS = 1e-9;

const dot = (a: Vec2, b: Vec2) => a.x * b.x + a.y * b.y;
const mag = (v: Vec2) => Math.sqrt(v.x * v.x + v.y * v.y);

function normalize(v: Vec2): Vec2 {
  const m = mag(v);
  if (m < EPS) return { x: 1, y: 0 };
  return { x: v.x / m, y: v.y / m };
}

function corr(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n < 2) return 0;
  const mx = x.slice(0, n).reduce((s, v) => s + v, 0) / n;
  const my = y.slice(0, n).reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const xv = x[i] - mx;
    const yv = y[i] - my;
    num += xv * yv;
    dx += xv * xv;
    dy += yv * yv;
  }
  if (dx < EPS || dy < EPS) return 0;
  return num / Math.sqrt(dx * dy);
}

export function computeAEC(
  environments: EnvironmentPoint[],
  genotypesForDirection: Pick<GgePoint, "pc1" | "pc2" | "mean">[] = []
): AECModel {
  const n = Math.max(environments.length, 1);
  const avgEnvironment = {
    x: environments.reduce((s, e) => s + e.pc1, 0) / n,
    y: environments.reduce((s, e) => s + e.pc2, 0) / n,
  };

  let abscissaUnit = normalize(avgEnvironment);

  if (genotypesForDirection.length >= 2) {
    const proj = genotypesForDirection.map((g) => g.pc1 * abscissaUnit.x + g.pc2 * abscissaUnit.y);
    const means = genotypesForDirection.map((g) => g.mean);
    if (corr(proj, means) < 0) {
      abscissaUnit = { x: -abscissaUnit.x, y: -abscissaUnit.y };
    }
  }

  const ordinateUnit = { x: -abscissaUnit.y, y: abscissaUnit.x };

  return {
    origin: { x: 0, y: 0 },
    avgEnvironment,
    abscissaUnit,
    ordinateUnit,
  };
}

export function projectGenotypesToAEC(
  genotypes: GgePoint[],
  aec: AECModel
): GenotypeProjection[] {
  return genotypes.map((g) => {
    const point = { x: g.pc1, y: g.pc2 };
    const abscissaScore = dot(point, aec.abscissaUnit);
    const ordinateScore = dot(point, aec.ordinateUnit);
    const projectionPoint = {
      x: aec.abscissaUnit.x * abscissaScore,
      y: aec.abscissaUnit.y * abscissaScore,
    };
    return {
      genotype: g.genotype,
      mean: g.mean,
      point,
      projectionPoint,
      abscissaScore,
      ordinateScore,
      stabilityDistance: Math.abs(ordinateScore),
    };
  });
}

export function computeIdealGenotype(
  projected: GenotypeProjection[]
): IdealGenotypeModel {
  if (!projected.length) {
    return {
      idealPoint: { x: 0, y: 0 },
      idealAbscissaScore: 0,
      idealGenotype: "",
      rankings: [],
      ringRadii: [0.3, 0.6, 0.9],
    };
  }

  const maxAbscissa = Math.max(...projected.map((p) => p.abscissaScore));
  const axisPointByGenotype = new Map(projected.map((p) => [p.genotype, p.projectionPoint]));

  // The ideal genotype has the highest AEC mean score and zero instability.
  const idealPoint = {
    x: (projected.find((p) => p.abscissaScore === maxAbscissa)?.projectionPoint.x ?? 0),
    y: (projected.find((p) => p.abscissaScore === maxAbscissa)?.projectionPoint.y ?? 0),
  };

  const withDistances = projected.map((p) => {
    const dx = p.point.x - idealPoint.x;
    const dy = p.point.y - idealPoint.y;
    return {
      genotype: p.genotype,
      distance_from_ideal: Math.sqrt(dx * dx + dy * dy),
      axisPoint: axisPointByGenotype.get(p.genotype) ?? p.projectionPoint,
      abscissaScore: p.abscissaScore,
      stabilityDistance: p.stabilityDistance,
    };
  });

  withDistances.sort((a, b) => {
    if (Math.abs(a.distance_from_ideal - b.distance_from_ideal) > EPS) {
      return a.distance_from_ideal - b.distance_from_ideal;
    }
    // Tie-breaker: higher mean score then better stability.
    if (Math.abs(a.abscissaScore - b.abscissaScore) > EPS) {
      return b.abscissaScore - a.abscissaScore;
    }
    return a.stabilityDistance - b.stabilityDistance;
  });

  const rankings: IdealRanking[] = withDistances.map((d, i) => ({
    genotype: d.genotype,
    distance_from_ideal: d.distance_from_ideal,
    rank: i + 1,
  }));

  const idealGenotype = rankings[0]?.genotype ?? "";
  const maxDistance = Math.max(...rankings.map((r) => r.distance_from_ideal), 0.1);
  const ringRadii = [0.25, 0.5, 0.75, 1].map((k) => Number((maxDistance * k).toFixed(6)));

  return {
    idealPoint,
    idealAbscissaScore: maxAbscissa,
    idealGenotype,
    rankings,
    ringRadii,
  };
}

export function drawStabilityAxis(aec: AECModel, extent: number): { from: Vec2; to: Vec2 } {
  const span = Math.max(extent, 1) * 1.15;
  return {
    from: { x: -aec.ordinateUnit.x * span, y: -aec.ordinateUnit.y * span },
    to: { x: aec.ordinateUnit.x * span, y: aec.ordinateUnit.y * span },
  };
}

export function drawAbscissaAxis(aec: AECModel, extent: number): { from: Vec2; to: Vec2 } {
  const span = Math.max(extent, 1) * 1.15;
  return {
    from: { x: -aec.abscissaUnit.x * span, y: -aec.abscissaUnit.y * span },
    to: { x: aec.abscissaUnit.x * span, y: aec.abscissaUnit.y * span },
  };
}
