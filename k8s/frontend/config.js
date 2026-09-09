// config.js
// Central knobs for the app. Edit these to point at your own embedding service.

export const CONFIG = {
  // Your deployed embedding endpoint (AWS Lambda Function URL or API Gateway).
  // The app POSTs { "text": "..." } and expects { "embedding": [ ...384 floats... ] }.
  // See lambda/handler.py for a reference implementation and the exact contract.
  EMBEDDING_ENDPOINT: "/embed",

  // While true, embeddings are generated locally so the whole UI runs with no backend.
  // The mock is deterministic (same sentence -> same vector) and unit-length, so the
  // geometry behaves sensibly. Flip to false once EMBEDDING_ENDPOINT is live.
  USE_MOCK: false,

  // MiniLM (all-MiniLM-L6-v2) outputs 384 dimensions.
  EMBEDDING_DIM: 384,

  // Fixed seed for the orthogonal reference vector. Keeping it fixed makes the
  // orthogonal axis reproducible across reloads for a given axis sentence.
  ORTHO_SEED: 1337,
};