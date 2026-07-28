#!/usr/bin/env bash
#
# Container smoke test for the embedding Lambda.
#
# Boots the built image locally through the Lambda Runtime Interface
# Emulator (RIE, included in the AWS base images) and invokes the handler
# over HTTP -- exercising the real CMD/entrypoint, the packaged model, and
# the end-to-end response shape. This is deliberately shallow: correctness
# of the handler and model is covered by the unit tests; this proves the
# *container* actually starts and serves.
#
# Usage:  ./smoke_test.sh <image-ref>
#     or: IMAGE_URI=<image-ref> ./smoke_test.sh
#
# Requires: docker, curl, jq. Assumes the image is built FROM an AWS
# Lambda base image (so RIE is present) with CMD ["app.handler"]. If you
# use a custom base, you must add the aws-lambda-rie binary yourself.

set -euo pipefail

IMAGE="${1:-${IMAGE_URI:?Pass the image ref as arg 1 or set IMAGE_URI}}"
CONTAINER_NAME="lambda-smoke-$$"
PORT=9000
URL="http://localhost:${PORT}/2015-03-31/functions/function/invocations"

cleanup() {
  echo "--- container logs ---"
  docker logs "$CONTAINER_NAME" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting container from ${IMAGE} ..."
docker run -d --name "$CONTAINER_NAME" -p "${PORT}:8080" "$IMAGE" >/dev/null

# Note: RIE returns HTTP 200 for any *successful invocation*, even when the
# function itself returns statusCode 400 -- so we assert on the function's
# statusCode inside the JSON body, never on curl's HTTP code.

# Happy path. The first call is a cold start: it triggers module import and
# loads the model, which can take a while, so we retry with a generous
# per-request timeout until it answers 200 (or we give up).
happy_event='{"body": "{\"text\": \"hello world\"}"}'
deadline=$(( SECONDS + 180 ))
resp=""
status=""
while (( SECONDS < deadline )); do
  resp=$(curl -s -m 120 -XPOST "$URL" -d "$happy_event" 2>/dev/null || true)
  status=$(printf '%s' "$resp" | jq -r '.statusCode // empty' 2>/dev/null || true)
  [[ "$status" == "200" ]] && break
  sleep 2
done

if [[ "$status" != "200" ]]; then
  echo "FAIL: happy path did not return 200 within timeout."
  echo "Last response: ${resp:-<none>}"
  exit 1
fi

# The function body is itself a JSON string; parse it and check the vector.
len=$(printf '%s' "$resp" | jq -r '.body | fromjson | .embedding | length')
if [[ -z "$len" || "$len" -lt 1 ]]; then
  echo "FAIL: 200 returned but no embedding in body."
  echo "Response: $resp"
  exit 1
fi
echo "PASS: happy path -> 200, embedding length ${len}"

# Error path (warm now): missing body must be handled, not crash -> 400.
err_resp=$(curl -s -m 30 -XPOST "$URL" -d '{}')
err_status=$(printf '%s' "$err_resp" | jq -r '.statusCode // empty')
if [[ "$err_status" != "400" ]]; then
  echo "FAIL: missing body should return 400, got: $err_resp"
  exit 1
fi
echo "PASS: missing body -> 400"

echo "Smoke test passed."