// linalg.js
// Small, dependency-free vector math for embeddings.

/** Dot product of two equal-length vectors. */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Euclidean (L2) norm. */
export function norm(a) {
  return Math.sqrt(dot(a, a));
}

/** Return a unit-length copy of `a`. Zero vectors are returned unchanged. */
export function normalize(a) {
  const n = norm(a);
  if (n === 0) return a.slice();
  return a.map((v) => v / n);
}

/**
 * Cosine similarity in [-1, 1]. Robust to non-unit inputs.
 * Returns 0 if either vector is all zeros.
 */
export function cosineSimilarity(a, b) {
  const denom = norm(a) * norm(b);
  if (denom === 0) return 0;
  return dot(a, b) / denom;
}

/** Clamp to the first-quadrant range [0, 1]. */
export function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

/**
 * mulberry32 PRNG. Deterministic given a 32-bit seed.
 * Returns a function producing floats in [0, 1). Used by the local mock.
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A vector of `dim` samples from a standard normal, using the supplied rng
 * (a function returning floats in [0,1)). Box–Muller transform. Used by the mock.
 */
export function randomGaussianVector(dim, rng) {
  const out = new Array(dim);
  for (let i = 0; i < dim; i += 2) {
    let u1 = rng();
    const u2 = rng();
    if (u1 < 1e-12) u1 = 1e-12; // avoid log(0)
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    out[i] = r * Math.cos(theta);
    if (i + 1 < dim) out[i + 1] = r * Math.sin(theta);
  }
  return out;
}