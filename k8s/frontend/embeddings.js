// embeddings.js
// Turns a sentence into a 384-dim vector, either via the remote endpoint
// or via a deterministic local mock (see CONFIG.USE_MOCK).

import { CONFIG } from "./config.js";
import { mulberry32, randomGaussianVector, normalize } from "./linalg.js";

/**
 * Deterministic stand-in for a real embedding model.
 * Hashes the sentence to a seed, draws a Gaussian vector, and unit-normalizes it.
 * The same sentence always maps to the same vector, so cosine geometry is stable.
 * NOTE: this captures none of the *meaning* of a sentence — it only lets you
 * exercise the UI and math before a real endpoint is wired up.
 */
function mockEmbedding(sentence) {
  let h = 2166136261 >>> 0; // FNV-1a
  for (let i = 0; i < sentence.length; i++) {
    h ^= sentence.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const rng = mulberry32(h >>> 0);
  return normalize(randomGaussianVector(CONFIG.EMBEDDING_DIM, rng));
}

/**
 * Get the embedding for a sentence.
 * @returns {Promise<number[]>} a 384-length array of floats.
 * @throws if the remote call fails or the response is malformed.
 */
export async function getEmbedding(sentence) {
  if (CONFIG.USE_MOCK) {
    await new Promise((r) => setTimeout(r, 120)); // pretend it's a network hop
    return mockEmbedding(sentence);
  }

  const res = await fetch(CONFIG.EMBEDDING_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: sentence }),
  });

  if (!res.ok) {
    throw new Error(`Embedding endpoint returned ${res.status}`);
  }

  const data = await res.json();
  let emb = data.embedding;
  // Some servers return a batch, i.e. shape (1, 384) -> [[...]]. Flatten to a vector.
  if (Array.isArray(emb) && emb.length === 1 && Array.isArray(emb[0])) {
    emb = emb[0];
  }
  if (!Array.isArray(emb) || emb.length === 0 || typeof emb[0] !== "number") {
    throw new Error("Response 'embedding' was not a flat numeric array");
  }
  if (emb.length !== CONFIG.EMBEDDING_DIM) {
    console.warn(`Expected ${CONFIG.EMBEDDING_DIM} dims, got ${emb.length}.`);
  }
  return emb;
}