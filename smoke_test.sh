#!/usr/bin/env bash
#
# Container smoke test for the embedding API.
#
# Boots the built image locally and exercises it over HTTP: the real CMD,
# the packaged model, the readiness gate, the response shape, and a clean
# SIGTERM shutdown. Deliberately shallow -- correctness of the handler and
# the model is covered by pytest; this proves the *container* starts and
# serves.
#
# Runs CPU-only. The image is GPU-capable, but ORT_PROVIDERS pins the ONNX
# session to CPU so this works on any runner with no driver present.
#
# Usage:  ./smoke_test.sh <image-ref>
#     or: IMAGE_URI=<image-ref> ./smoke_test.sh
#
# Requires: docker, curl, jq.

set -euo pipefail

IMAGE="${1:-${IMAGE_URI:?Pass the image ref as arg 1 or set IMAGE_URI}}"
CONTAINER_NAME="embed-smoke-$$"
PORT="${PORT:-8080}"
BASE="http://localhost:${PORT}"
EXPECT_DIM="${EXPECT_DIM:-384}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"

cleanup() {
  echo "--- container logs ---"
  docker logs "$CONTAINER_NAME" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

echo "Starting container from ${IMAGE} ..."
docker run -d --name "$CONTAINER_NAME" -p "${PORT}:8080" \
  -e REQUIRE_GPU=0 \
  -e ORT_PROVIDERS=CPUExecutionProvider \
  -e OTEL_SERVICE_NAME=embedvisual-ci \
  "$IMAGE" >/dev/null
# No OTLP endpoint is set on purpose: that exercises the "telemetry not
# configured" path in otel_setup, which must warn and no-op rather than crash.

# ---- 1. Readiness -----------------------------------------------------------
# /readyz is 503 until lifespan finishes model.load(), so this doubles as the
# startupProbe rehearsal.
echo "Waiting for /readyz ..."
deadline=$(( SECONDS + READY_TIMEOUT ))
while true; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${BASE}/readyz" || true)
  [[ "$code" == "200" ]] && break
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo false)
  [[ "$running" == "true" ]] || fail "container exited before becoming ready"
  (( SECONDS < deadline )) || fail "/readyz did not return 200 within ${READY_TIMEOUT}s (last: ${code:-none})"
  sleep 2
done

providers=$(curl -sf "${BASE}/readyz" | jq -r '.providers | join(",")')
echo "PASS: ready -> providers [${providers}]"
grep -q 'CPUExecutionProvider' <<<"$providers" \
  || fail "expected CPUExecutionProvider in the session, got [${providers}]"

# ---- 2. Liveness ------------------------------------------------------------
health=$(curl -sf -m 5 "${BASE}/healthz" | jq -r '.status')
[[ "$health" == "ok" ]] || fail "/healthz returned status=${health}"
echo "PASS: /healthz -> ok"

# ---- 3. Happy path ----------------------------------------------------------
resp=$(curl -sf -m 60 -XPOST "${BASE}/embed" \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello world"}') || fail "POST /embed (single) did not return 2xx"

rows=$(jq '.embedding | length' <<<"$resp")
dims=$(jq '.embedding[0] | length' <<<"$resp")
[[ "$rows" == "1" ]] || fail "expected 1 embedding, got ${rows}"
[[ "$dims" == "$EXPECT_DIM" ]] || fail "expected ${EXPECT_DIM} dims, got ${dims}"

# The model L2-normalizes before returning, so the vector must be unit length.
norm_ok=$(jq '[.embedding[0][] | . * .] | add | (. - 1 | fabs) < 1e-4' <<<"$resp")
[[ "$norm_ok" == "true" ]] || fail "embedding is not unit length"
echo "PASS: single -> 200, 1 x ${dims}, unit norm"

# ---- 4. Batch ---------------------------------------------------------------
batch=$(curl -sf -m 60 -XPOST "${BASE}/embed" \
  -H 'Content-Type: application/json' \
  -d '{"text": ["the man plays the guitar", "a dog sleeps on the porch"]}') \
  || fail "POST /embed (batch) did not return 2xx"

brows=$(jq '.embedding | length' <<<"$batch")
[[ "$brows" == "2" ]] || fail "expected 2 embeddings, got ${brows}"
echo "PASS: batch -> 200, 2 x ${EXPECT_DIM}"

# ---- 5. Bad input -----------------------------------------------------------
# Must be a 400 from the RequestValidationError handler, not a 422 and not a 500.
err_code=$(curl -s -o /tmp/err.json -w '%{http_code}' -m 10 -XPOST "${BASE}/embed" \
  -H 'Content-Type: application/json' -d '{"text": 123}')
[[ "$err_code" == "400" ]] || fail "bad body should be 400, got ${err_code}: $(cat /tmp/err.json)"

mal_code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -XPOST "${BASE}/embed" \
  -H 'Content-Type: application/json' -d '{not valid json')
[[ "$mal_code" == "400" ]] || fail "malformed JSON should be 400, got ${mal_code}"
echo "PASS: bad input -> 400"

# ---- 6. Graceful shutdown ---------------------------------------------------
# terminationGracePeriodSeconds is 30 in the Deployment; SIGTERM must run the
# lifespan shutdown (otel.flush) and exit 0 well inside that.
echo "Stopping container (SIGTERM) ..."
start=$SECONDS
docker stop -t 30 "$CONTAINER_NAME" >/dev/null
elapsed=$(( SECONDS - start ))
exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")
[[ "$exit_code" == "0" ]] || fail "container exited ${exit_code} on SIGTERM (expected 0)"
(( elapsed < 30 )) || fail "shutdown took ${elapsed}s, at or over the grace period"
echo "PASS: clean shutdown in ${elapsed}s"

echo "Smoke test passed."