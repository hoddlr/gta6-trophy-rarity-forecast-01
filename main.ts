// gta6-trophy-rarity-forecast-01 — logistic-saturation trophy-rate forecaster (Node 22, stdlib only).
//
// Model (research spec section 7): for each trophy, fit a saturating curve over the
// rate's history, rate(t) = A / (1 + exp(-(a + b*ln(1+h)))), h = hours since release,
// with the asymptote A picked on a grid by least squares and shrunk toward the tier
// mean when the per-trophy fit is unstable. Spreads combine a random-walk term with
// fit-residual and refresh-jump terms, and are asymmetric in the direction the median
// moves. A sanity check on implied earners (E = rate * N) pulls the median back toward
// persistence when the fit would imply earners falling or exceeding players tracked.
//
// Everything is computed from the case file at runtime: no network, no clock, no
// randomness, no hard-coded rates. When the history is too thin to fit (fewer than 3
// distinct snapshots, or a horizon under 6 hours) the entry falls back to the house
// baseline (last rate, random-walk spread) with the lower tail widened 1.5x against
// an unseen Exophase refresh.
//
// Input: every <id>.in in MAKEFUN_INPUT_DIR (default /input); output: <id>.out in
// MAKEFUN_OUTPUT_DIR (default /output), one row per trophy, five quantiles each.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { byteLength } from 'node:buffer';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, in logit units per square-root day (house baseline). */
const SIGMA_RW = 0.5;
/** Floor / cap of the total logit spread, and the asymmetric widening factor. */
const SIGMA_MIN = 1.0;
const SIGMA_MAX = 2.0;
const ASYM = 1.25;
/** Fallback lower-tail widening against an unseen refresh. */
const FALLBACK_LOWER = 1.5;
/** Shortest horizon (hours) on which the saturation fit is trusted. */
const MIN_FIT_HORIZON_H = 6;
/** Output size cap, bytes. */
const MAX_OUT_BYTES = 65_536;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
  game?: { release_date?: string };
}
/** One distinct snapshot after merging consecutive identical reads. */
interface Snap {
  /** Hours since the game's release (00:00 UTC on release_date). */
  h: number;
  /** Players tracked at this snapshot. */
  n: number;
  /** Rates in trophy order, nulls carried forward from the previous read. */
  rates: Array<number | null>;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
/** Logit of a percent, clamped the way the scorer clamps it. */
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
/** Inverse logit, as a percent. */
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const mean = (xs: number[]): number => xs.reduce((s, x) => s + x, 0) / xs.length;
/** Population standard deviation. */
const sd = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
};
const medianOf = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};
/** Ordinary least squares y = a + b*x; returns {a, b, rss}. */
function ols(xs: number[], ys: number[]): { a: number; b: number; rss: number } {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += r * r;
  }
  return { a, b, rss };
}

/**
 * Collapse the case history into distinct snapshots: consecutive reads with the
 * same players-tracked count and the same rates (after carrying nulls forward)
 * are one snapshot. Hours are measured from the game's release date at 00:00 UTC;
 * when the case carries no release date, the first read is t = 0.
 */
function buildSnapshots(c: Case): Snap[] {
  const releaseMs = c.game?.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : NaN;
  const t0 = Number.isFinite(releaseMs) ? releaseMs : Date.parse(c.history[0].at);
  const snaps: Snap[] = [];
  const carried: Array<number | null> = c.trophies.map(() => null);
  for (const read of c.history) {
    const h = (Date.parse(read.at) - t0) / 3_600_000;
    const rates = read.rates.map((r, i) => {
      if (r !== null && r !== undefined) {
        carried[i] = r;
        return r;
      }
      return carried[i];
    });
    const prev = snaps[snaps.length - 1];
    if (prev && prev.n === read.players_tracked && prev.rates.every((r, i) => r === rates[i])) continue;
    snaps.push({ h, n: read.players_tracked, rates });
  }
  return snaps;
}

interface Fit {
  a: number;
  b: number;
  /** Saturation asymptote, as a fraction (0..1). */
  A: number;
  /** Population SD of the fit residuals, in logit units. */
  residSd: number;
  /** Number of snapshots the fit used. */
  n: number;
}

/**
 * Fit the logistic saturation curve for one trophy over its snapshots.
 * x = ln(1+h), y = rate/100. For each asymptote A on a 0.05 grid, w = logit(y/A)
 * is regressed on x by OLS; the A with the smallest residual sum of squares wins
 * (ties within 1% go to the smallest A, the more conservative one). Returns null
 * when the trophy has fewer than 3 usable snapshots.
 */
function fitSaturation(snaps: Snap[], idx: number, lastRate: number): Fit | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const s of snaps) {
    const r = s.rates[idx];
    if (r === null || r === undefined) continue;
    xs.push(Math.log(Math.max(1 + s.h, 1e-6)));
    ys.push(r / 100);
  }
  if (xs.length < 3) return null;
  const grid: number[] = [];
  for (let a = 0.05; a <= 1.0 + 1e-9; a += 0.05) grid.push(Math.round(a * 100) / 100);
  if (lastRate > 95 && !grid.includes(Math.round(lastRate) / 100)) {
    grid.push(lastRate / 100);
    grid.sort((a, b) => a - b);
  }
  let best: { A: number; a: number; b: number; rss: number } | null = null;
  for (const A of grid) {
    const w = ys.map((y) => {
      const p = Math.min(0.999, Math.max(1e-4, y / A));
      return Math.log(p / (1 - p));
    });
    const { a, b, rss } = ols(xs, w);
    if (best === null || rss < best.rss) best = { A, a, b, rss };
  }
  if (best === null) return null;
  // Tie-break: smallest A whose RSS is within 1% of the minimum.
  for (const A of grid) {
    const w = ys.map((y) => {
      const p = Math.min(0.999, Math.max(1e-4, y / A));
      return Math.log(p / (1 - p));
    });
    const { a, b, rss } = ols(xs, w);
    if (rss <= best.rss * 1.01) {
      return { a, b, A, residSd: Math.sqrt(rss / xs.length), n: xs.length };
    }
  }
  return { a: best.a, b: best.b, A: best.A, residSd: Math.sqrt(best.rss / xs.length), n: xs.length };
}

/** Median (percent) of a fitted saturation curve at the target hour. */
function fitMedian(fit: Fit, hTarget: number): number {
  const x = Math.log(Math.max(1 + hTarget, 1e-6));
  return clampPct((100 * fit.A) / (1 + Math.exp(-(fit.a + fit.b * x))));
}

/**
 * The five quantiles for a median with separate lower / upper logit spreads.
 * Rounded to 4 decimals, sorted ascending, clamped to [0.01, 99.99].
 */
function quantiles(medianPct: number, sigmaDown: number, sigmaUp: number): number[] {
  const l = logit(medianPct);
  const qs = Z.map((z) => round4(clampPct(invLogit(l + z * (z < 0 ? sigmaDown : sigmaUp)))));
  qs.sort((a, b) => a - b);
  return qs;
}

/** House baseline quantiles: last rate, random-walk spread, lower tail widened. */
function baselineQs(lastRate: number, dDays: number): number[] {
  const s = SIGMA_RW * Math.sqrt(dDays);
  return quantiles(lastRate, s * FALLBACK_LOWER, s);
}

/**
 * E/N sanity check: fit ln N = c + f*ln(1+h) over the snapshots and project
 * players tracked at the target. From the candidate median, implied earners at
 * the target must not fall below the last observed earners, must not exceed the
 * projected players, and the projection itself must not fall below the last
 * observed players. On failure, pull the median halfway (in logit) toward the
 * last rate and re-check once; if it still fails, use the last rate.
 */
function sanityCheckedMedian(
  mPct: number,
  lastRate: number,
  snaps: Snap[],
  hTarget: number,
): number {
  const xs = snaps.map((s) => Math.log(Math.max(1 + s.h, 1e-6)));
  const ys = snaps.map((s) => Math.log(Math.max(s.n, 1)));
  const { a: c, b: f } = ols(xs, ys);
  const nTarget = Math.exp(c + f * Math.log(Math.max(1 + hTarget, 1e-6)));
  const last = snaps[snaps.length - 1];
  const eLast = Math.round((lastRate * last.n) / 100);
  const ok = (m: number): boolean => {
    const eTarget = (m / 100) * nTarget;
    return eTarget >= eLast && eTarget <= nTarget && nTarget >= last.n;
  };
  if (ok(mPct)) return mPct;
  const pulled = invLogit((logit(mPct) + logit(lastRate)) / 2);
  if (ok(pulled)) return pulled;
  return lastRate;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const dDays = c.horizon_minutes / 1440;
  const releaseMs = c.game?.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : NaN;
  const t0 = Number.isFinite(releaseMs) ? releaseMs : Date.parse(c.history[0].at);
  const hTarget = (Date.parse(c.target) - t0) / 3_600_000;
  const snaps = buildSnapshots(c);
  const lastSnap = snaps[snaps.length - 1];
  const useFit = snaps.length >= 3 && c.horizon_minutes / 60 >= MIN_FIT_HORIZON_H;

  // First pass: fit every trophy that can be fit, so tier means exist for shrinkage.
  const fits: Array<Fit | null> = c.trophies.map((_, i) =>
    useFit ? fitSaturation(snaps, i, lastSnap?.rates[i] ?? 0) : null,
  );
  const tierStats = new Map<string, { b: number[]; A: number[] }>();
  const allB: number[] = [];
  const allA: number[] = [];
  c.trophies.forEach((t, i) => {
    const f = fits[i];
    if (!f) return;
    const key = t.tier || '';
    if (!tierStats.has(key)) tierStats.set(key, { b: [], A: [] });
    tierStats.get(key)!.b.push(f.b);
    tierStats.get(key)!.A.push(f.A);
    allB.push(f.b);
    allA.push(f.A);
  });

  // Fit ln N for the sanity check once per case.
  const rows: Array<{ id: number; q: number[] }> = [];
  c.trophies.forEach((t, i) => {
    const lastRate = lastSnap?.rates[i];
    // Trophy with no history at all: tier median of last rates, spread 2.0.
    if (lastRate === null || lastRate === undefined) {
      const peers = c.trophies
        .map((u, j) => ({ tier: u.tier, r: lastSnap?.rates[j] }))
        .filter((p) => p.tier === t.tier && p.r !== null && p.r !== undefined)
        .map((p) => p.r as number);
      const pool = peers.length > 0 ? peers : c.trophies
        .map((_, j) => lastSnap?.rates[j])
        .filter((r): r is number => r !== null && r !== undefined);
      const med = pool.length > 0 ? medianOf(pool) : 1;
      rows.push({ id: t.id, q: quantiles(med, 2.0, 2.0) });
      return;
    }

    // Special case: (near-)universal trophy — median ~99.9, wide lower tail.
    if (lastRate >= 99.9) {
      rows.push({ id: t.id, q: quantiles(99.9, 2.0, 0.3) });
      return;
    }

    let medianPct: number;
    let sigmaDown: number;
    let sigmaUp: number;
    const fit = fits[i];
    if (fit !== null) {
      // Shrinkage: unstable fits (few snapshots or large residuals) are pulled
      // halfway toward the tier mean of slope and asymptote.
      let { a, b, A } = fit;
      if (fit.n < 6 || fit.residSd > 0.5) {
        const stats = tierStats.get(t.tier || '');
        const meanB = stats && stats.b.length > 0 ? mean(stats.b) : allB.length > 0 ? mean(allB) : b;
        const meanA = stats && stats.A.length > 0 ? mean(stats.A) : allA.length > 0 ? mean(allA) : A;
        b = 0.5 * b + 0.5 * meanB;
        A = 0.5 * A + 0.5 * meanA;
        // Re-fit the intercept for the shrunk slope and asymptote.
        const xs: number[] = [];
        const w: number[] = [];
        for (const s of snaps) {
          const r = s.rates[i];
          if (r === null || r === undefined) continue;
          xs.push(Math.log(Math.max(1 + s.h, 1e-6)));
          const p = Math.min(0.999, Math.max(1e-4, r / 100 / A));
          w.push(Math.log(p / (1 - p)));
        }
        a = mean(w.map((wi, k) => wi - b * xs[k]));
      }
      medianPct = fitMedian({ a, b, A, residSd: fit.residSd, n: fit.n }, hTarget);
      medianPct = sanityCheckedMedian(medianPct, lastRate, snaps, hTarget);
      // Spread: random walk + fit residuals + refresh jumps, floored and capped.
      const sigmaRw = SIGMA_RW * Math.sqrt(dDays);
      const sigmaFit = Math.min(1.0, Math.max(0.3, fit.residSd));
      const jumps: number[] = [];
      let prevLogit: number | null = null;
      for (const s of snaps) {
        const r = s.rates[i];
        if (r === null || r === undefined) continue;
        const l = logit(r);
        if (prevLogit !== null) jumps.push(l - prevLogit);
        prevLogit = l;
      }
      const sigmaRef = jumps.length < 2 ? 0.8 : Math.max(0.3, sd(jumps));
      const sigmaTotal = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, Math.sqrt(sigmaRw * sigmaRw + sigmaFit * sigmaFit + sigmaRef * sigmaRef)));
      sigmaDown = sigmaTotal;
      sigmaUp = sigmaTotal;
      if (medianPct > lastRate) sigmaUp = sigmaTotal * ASYM;
      else sigmaDown = sigmaTotal * ASYM;
    } else {
      // Fallback: house baseline with the lower tail widened against an unseen refresh.
      medianPct = lastRate;
      const s = SIGMA_RW * Math.sqrt(dDays);
      sigmaDown = s * FALLBACK_LOWER;
      sigmaUp = s;
    }

    // Special case: (near-)unearned trophy — median at least 0.02, wide upper tail.
    if (lastRate <= 0.01) {
      medianPct = Math.max(medianPct, 0.02);
      sigmaDown = 0.5;
      sigmaUp = 2.0;
    }
    rows.push({ id: t.id, q: quantiles(medianPct, sigmaDown, sigmaUp) });
  });
  return rows;
}

/** Plain house baseline for every trophy (emergency output if validation fails). */
function baselineRows(c: Case): Array<{ id: number; q: number[] }> {
  const dDays = c.horizon_minutes / 1440;
  const snaps = buildSnapshots(c);
  const last = snaps[snaps.length - 1];
  return c.trophies.map((t, i) => {
    const r = last?.rates[i];
    const rate = r === null || r === undefined ? 1 : r;
    const s = SIGMA_RW * Math.sqrt(dDays);
    return { id: t.id, q: quantiles(rate, s, s) };
  });
}

function validateOut(c: Case, text: string): boolean {
  if (byteLength(text, 'utf8') > MAX_OUT_BYTES) return false;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return false;
  }
  if (typeof parsed !== 'object' || parsed === null) return false;
  const trophies = (parsed as { trophies?: unknown }).trophies;
  if (!Array.isArray(trophies) || trophies.length !== c.trophies.length) return false;
  const seen = new Set<number>();
  for (const row of trophies) {
    if (typeof row !== 'object' || row === null) return false;
    const { id, q } = row as { id?: unknown; q?: unknown };
    if (typeof id !== 'number' || seen.has(id)) return false;
    seen.add(id);
    if (!Array.isArray(q) || q.length !== LEVELS.length) return false;
    let prev = -Infinity;
    for (const v of q) {
      if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 100 || v < prev) return false;
      prev = v;
    }
  }
  const want = new Set(c.trophies.map((t) => t.id));
  if (seen.size !== want.size) return false;
  for (const id of seen) if (!want.has(id)) return false;
  return true;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  let text = `${JSON.stringify({ trophies: forecastCase(c) })}\n`;
  if (!validateOut(c, text)) text = `${JSON.stringify({ trophies: baselineRows(c) })}\n`;
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), text);
}
