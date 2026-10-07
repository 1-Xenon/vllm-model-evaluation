# Operator guide

## Start the application

- Copy `config.example.toml` to `config.toml` and configure storage and the runner.
- Run `uvicorn vllm_evaluation.app:create_app --factory --reload` during development.
- Open `/docs` for the operator API or `/` for the local landing page.

## Recommended workflow

- Import a bundle with `POST /api/snapshots/import`.
- Create one model configuration for each checkpoint with `POST /api/model-configs`.
- Create a run against the immutable snapshot with `POST /api/runs`.
- Execute one model at a time with `POST /api/runs/{run_id}/execute`.
- Inspect `/api/runs/{run_id}/progress` and `/api/runs/{run_id}/failures`.
- Unload or stop the first model externally, then repeat the run workflow for the second model.
- Create a comparison with the selected successful attempt IDs, freeze it, and create reviewer assignments.

The application does not automatically swap model servers. This is deliberate: the initial release records the endpoint and checkpoint identity while the operator controls local GPU allocation.

## Comparison and review

- A comparison is created as a draft so missing, failed, and truncated tasks are visible before freezing.
- Only paired successful attempts are added to the frozen review set.
- Reviewer left/right placement is deterministic for the comparison seed, reviewer ID, and task ID.
- Model names, endpoint URLs, timing, and run metadata are omitted from the reviewer payload.
- Judgment submissions create a new version rather than overwriting history.

## Export

- JSONL is the full-fidelity export format.
- CSV is a flat analysis-oriented export.
- Set `include_model_mapping` only for operator exports; leave it false for reviewer-facing exports.
- Summary counts explicitly identify reviewer votes rather than unique tasks.
+

## Canonical live two-model test

Use this section for the complete application test. It keeps the model identities as placeholders.

Replace all values beginning with REPLACE_WITH_ before running commands:

~~~bash
cd /home/nic/projects/vllm-model-evaluation
source .venv/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export APP_PORT=8002
export APP_BASE_URL="http://127.0.0.1:$APP_PORT"
export MODEL_ENDPOINT_URL="http://127.0.0.1:8001/v1/chat/completions"
export MODEL_ENDPOINT_PORT=8001

export MODEL_A_ID="REPLACE_WITH_MODEL_A_SERVED_NAME"
export MODEL_B_ID="REPLACE_WITH_MODEL_B_SERVED_NAME"
export MODEL_A_CHECKPOINT="REPLACE_WITH_MODEL_A_CHECKPOINT_IDENTITY"
export MODEL_B_CHECKPOINT="REPLACE_WITH_MODEL_B_CHECKPOINT_IDENTITY"

export MODEL_A_CONTAINER="evaluation-model-a"
export MODEL_B_CONTAINER="evaluation-model-b"
export VLLM_IMAGE="vllm/vllm-openai:latest"
export MODEL_CACHE="$PROJECT_ROOT/data/huggingface"
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export BUNDLE_ROOT="$PROJECT_ROOT/data/task-bundles/live-smoke-bundle"
export DB_PATH="$PROJECT_ROOT/data/evaluation.db"
~~~

MODEL_A_ID and MODEL_B_ID must exactly match the IDs returned by each server's /v1/models endpoint. If the models are not served by vLLM's official Docker image, replace the Docker launch commands below. If the endpoint is not on port 8001, replace MODEL_ENDPOINT_URL, MODEL_ENDPOINT_PORT, and every matching application environment value.

Shell variables are not shared between terminals. Repeat the export block in each terminal that needs it.

### Prepare the live image bundle

The repository example contains SVG. Most vLLM image processors expect PNG, JPEG, or WebP, so convert the fixture before the live test:

~~~bash
command -v ffmpeg
mkdir -p "$BUNDLE_ROOT/media"

ffmpeg -loglevel error -y \
  -i "$PROJECT_ROOT/examples/evaluation-bundle/media/example.svg" \
  "$BUNDLE_ROOT/media/example.png"

sed 's#media/example.svg#media/example.png#g' \
  "$PROJECT_ROOT/examples/evaluation-bundle/manifest.jsonl" \
  > "$BUNDLE_ROOT/manifest.jsonl"

test -f "$BUNDLE_ROOT/manifest.jsonl"
test -f "$BUNDLE_ROOT/media/example.png"
~~~

If ffmpeg is unavailable, provide a bundle containing PNG, JPEG, or WebP images and set BUNDLE_ROOT to its absolute path.

### Start model A

Run this in Terminal 1:

~~~bash
mkdir -p "$MODEL_CACHE"
docker rm -f "$MODEL_A_CONTAINER" 2>/dev/null || true

docker run -d \
  --name "$MODEL_A_CONTAINER" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p "$MODEL_ENDPOINT_PORT:8000" \
  -v "$MODEL_CACHE:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  "$MODEL_A_ID" \
  --dtype half \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image":1,"video":0}'
~~~

Wait until it is ready:

~~~bash
for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$MODEL_ENDPOINT_PORT/v1/models"; then
    break
  fi
  sleep 5
done
~~~

Confirm the returned model ID is MODEL_A_ID, then test it:

~~~bash
curl -sS "$MODEL_ENDPOINT_URL" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"$MODEL_A_ID\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: model-a-ok\"}],
    \"max_tokens\":8,
    \"temperature\":0
  }" | "$PYTHON_BIN" -m json.tool
~~~

Do not continue until the model request succeeds.

### Start the application for model A

Run this in Terminal 2:

~~~bash
export VLLM_EVAL_RUNNER_TYPE=openai_compatible
export VLLM_EVAL_RUNNER_ENDPOINT_URL="$MODEL_ENDPOINT_URL"
export VLLM_EVAL_RUNNER_MODEL_NAME="$MODEL_A_ID"
export VLLM_EVAL_TASK_BUNDLE_ROOT="$BUNDLE_ROOT"

uvicorn vllm_evaluation.app:create_app \
  --factory \
  --host 127.0.0.1 \
  --port "$APP_PORT"
~~~

Keep it running. In Terminal 3, repeat the project and variable setup, then verify:

~~~bash
curl -sS "$APP_BASE_URL/health" | "$PYTHON_BIN" -m json.tool
~~~

Expected runner type is openai_compatible. If this request fails, run curl -i "$APP_BASE_URL/health" and inspect the application terminal before continuing.

### Import the snapshot and run model A

Run in Terminal 3:

~~~bash
SNAPSHOT_RESPONSE=$(
  curl -sS -X POST "$APP_BASE_URL/api/snapshots/import" \
    -H 'Content-Type: application/json' \
    -d "{\"bundle_root\":\"$BUNDLE_ROOT\"}"
)
echo "$SNAPSHOT_RESPONSE" | "$PYTHON_BIN" -m json.tool

SNAPSHOT_ID=$(
  printf '%s' "$SNAPSHOT_RESPONSE" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)
echo "$SNAPSHOT_ID"

MODEL_A_CONFIG_ID=$(
  curl -sS -X POST "$APP_BASE_URL/api/model-configs" \
    -H 'Content-Type: application/json' \
    -d "{
      \"display_name\":\"Model A\",
      \"endpoint_url\":\"$MODEL_ENDPOINT_URL\",
      \"served_model_name\":\"$MODEL_A_ID\",
      \"checkpoint_identity\":\"$MODEL_A_CHECKPOINT\",
      \"generation_settings\":{\"temperature\":0,\"max_tokens\":128}
    }" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

RUN_A_ID=$(
  curl -sS -X POST "$APP_BASE_URL/api/runs" \
    -H 'Content-Type: application/json' \
    -d "{
      \"snapshot_id\":\"$SNAPSHOT_ID\",
      \"model_config_id\":\"$MODEL_A_CONFIG_ID\",
      \"run_label\":\"Model A live run\"
    }" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

curl -sS -X POST "$APP_BASE_URL/api/runs/$RUN_A_ID/execute" \
  -H 'Content-Type: application/json' \
  -d "{\"asset_root\":\"$BUNDLE_ROOT\",\"concurrency\":1}" |
  "$PYTHON_BIN" -m json.tool

curl -sS "$APP_BASE_URL/api/runs/$RUN_A_ID/progress" | "$PYTHON_BIN" -m json.tool
curl -sS "$APP_BASE_URL/api/runs/$RUN_A_ID/failures" | "$PYTHON_BIN" -m json.tool
~~~

Do not continue until model A has completed and its failures have been reviewed.

### Replace model A with model B

In Terminal 1:

~~~bash
docker stop "$MODEL_A_CONTAINER"
docker rm "$MODEL_A_CONTAINER"

docker rm -f "$MODEL_B_CONTAINER" 2>/dev/null || true
docker run -d \
  --name "$MODEL_B_CONTAINER" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p "$MODEL_ENDPOINT_PORT:8000" \
  -v "$MODEL_CACHE:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  "$MODEL_B_ID" \
  --dtype half \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image":1,"video":0}'
~~~

Wait for model B with the same /v1/models loop. Confirm its returned ID is MODEL_B_ID.

Stop the application in Terminal 2 with Ctrl+C. Restart it with:

~~~bash
export VLLM_EVAL_RUNNER_TYPE=openai_compatible
export VLLM_EVAL_RUNNER_ENDPOINT_URL="$MODEL_ENDPOINT_URL"
export VLLM_EVAL_RUNNER_MODEL_NAME="$MODEL_B_ID"
export VLLM_EVAL_TASK_BUNDLE_ROOT="$BUNDLE_ROOT"

uvicorn vllm_evaluation.app:create_app \
  --factory \
  --host 127.0.0.1 \
  --port "$APP_PORT"
~~~

The restart is required because runner settings are loaded when the application starts. Keep the original database, BUNDLE_ROOT, and SNAPSHOT_ID.

### Run model B on the same snapshot

Run in Terminal 3:

~~~bash
MODEL_B_CONFIG_ID=$(
  curl -sS -X POST "$APP_BASE_URL/api/model-configs" \
    -H 'Content-Type: application/json' \
    -d "{
      \"display_name\":\"Model B\",
      \"endpoint_url\":\"$MODEL_ENDPOINT_URL\",
      \"served_model_name\":\"$MODEL_B_ID\",
      \"checkpoint_identity\":\"$MODEL_B_CHECKPOINT\",
      \"generation_settings\":{\"temperature\":0,\"max_tokens\":128}
    }" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

RUN_B_ID=$(
  curl -sS -X POST "$APP_BASE_URL/api/runs" \
    -H 'Content-Type: application/json' \
    -d "{
      \"snapshot_id\":\"$SNAPSHOT_ID\",
      \"model_config_id\":\"$MODEL_B_CONFIG_ID\",
      \"run_label\":\"Model B live run\"
    }" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

curl -sS -X POST "$APP_BASE_URL/api/runs/$RUN_B_ID/execute" \
  -H 'Content-Type: application/json' \
  -d "{\"asset_root\":\"$BUNDLE_ROOT\",\"concurrency\":1}" |
  "$PYTHON_BIN" -m json.tool

curl -sS "$APP_BASE_URL/api/runs/$RUN_B_ID/progress" | "$PYTHON_BIN" -m json.tool
curl -sS "$APP_BASE_URL/api/runs/$RUN_B_ID/failures" | "$PYTHON_BIN" -m json.tool
~~~

The two runs now reference the same immutable snapshot but different recorded model configurations.

### Create and freeze the blind comparison

Collect successful attempt IDs:

~~~bash
ATTEMPTS_A=$(
  "$PYTHON_BIN" -c '
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    "select id from generation_attempts "
    "where run_id = ? and status = ? order by task_id, attempt_number",
    (sys.argv[2], "succeeded"),
).fetchall()
print(json.dumps([row[0] for row in rows]))
' "$DB_PATH" "$RUN_A_ID"
)

ATTEMPTS_B=$(
  "$PYTHON_BIN" -c '
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    "select id from generation_attempts "
    "where run_id = ? and status = ? order by task_id, attempt_number",
    (sys.argv[2], "succeeded"),
).fetchall()
print(json.dumps([row[0] for row in rows]))
' "$DB_PATH" "$RUN_B_ID"
)

echo "Model A attempts: $ATTEMPTS_A"
echo "Model B attempts: $ATTEMPTS_B"
~~~

The arrays must contain the same number of attempts and correspond to the same task IDs.

~~~bash
COMPARISON_RESPONSE=$(
  curl -sS -X POST "$APP_BASE_URL/api/comparisons" \
    -H 'Content-Type: application/json' \
    -d "{
      \"snapshot_id\":\"$SNAPSHOT_ID\",
      \"name\":\"Model A versus Model B live comparison\",
      \"assignment_seed\":\"live-smoke-seed-001\",
      \"model_a_attempt_ids\":$ATTEMPTS_A,
      \"model_b_attempt_ids\":$ATTEMPTS_B,
      \"created_by\":\"operator\"
    }"
)
echo "$COMPARISON_RESPONSE" | "$PYTHON_BIN" -m json.tool

COMPARISON_ID=$(
  printf '%s' "$COMPARISON_RESPONSE" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["comparison_id"])'
)

curl -sS -X POST "$APP_BASE_URL/api/comparisons/$COMPARISON_ID/freeze" |
  "$PYTHON_BIN" -m json.tool
~~~

Do not freeze if the response reports missing pairs, snapshot mismatches, or non-successful attempts.

### Review and export

Create a reviewer assignment:

~~~bash
ASSIGNMENTS_RESPONSE=$(
  curl -sS "$APP_BASE_URL/api/comparisons/$COMPARISON_ID/assignments/reviewer-1"
)
echo "$ASSIGNMENTS_RESPONSE" | "$PYTHON_BIN" -m json.tool

ASSIGNMENT_ID=$(
  printf '%s' "$ASSIGNMENTS_RESPONSE" |
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)[0]["assignment_id"])'
)

curl -sS "$APP_BASE_URL/api/review/assignments/$ASSIGNMENT_ID" |
  "$PYTHON_BIN" -m json.tool
~~~

The reviewer response should contain the prompt, inputs, and anonymous left/right outputs. It must not contain model names, endpoint URLs, checkpoint identities, or timing data.

Submit a judgment:

~~~bash
curl -sS -X POST \
  "$APP_BASE_URL/api/review/assignments/$ASSIGNMENT_ID/judgment" \
  -H 'Content-Type: application/json' \
  -d '{
    "overall_preference":"left",
    "left_acceptability":"acceptable",
    "right_acceptability":"needs_revision",
    "disposition":"complete",
    "error_tags":["verbosity"],
    "comments":"Record reviewer observations here."
  }' |
  "$PYTHON_BIN" -m json.tool
~~~

View the summary and export:

~~~bash
curl -sS "$APP_BASE_URL/api/exports/$COMPARISON_ID/summary" |
  "$PYTHON_BIN" -m json.tool

curl -sS -X POST "$APP_BASE_URL/api/exports/$COMPARISON_ID" \
  -H 'Content-Type: application/json' \
  -d "{
    \"format\":\"jsonl\",
    \"path\":\"$PROJECT_ROOT/data/exports/live-comparison.jsonl\",
    \"include_model_mapping\":true
  }" | "$PYTHON_BIN" -m json.tool

curl -sS -X POST "$APP_BASE_URL/api/exports/$COMPARISON_ID" \
  -H 'Content-Type: application/json' \
  -d "{
    \"format\":\"csv\",
    \"path\":\"$PROJECT_ROOT/data/exports/live-comparison.csv\"
  }" | "$PYTHON_BIN" -m json.tool

ls -lh "$PROJECT_ROOT/data/exports/"
~~~

### Troubleshooting

- Address already in use: keep the application on port 8002, or replace APP_PORT and APP_BASE_URL.
- Empty JSON parser input: run curl -i "$APP_BASE_URL/health" before piping a response into Python.
- Model endpoint unavailable: run docker ps and docker logs for the active model container.
- Model ID mismatch: compare the placeholder with curl http://127.0.0.1:8001/v1/models.
- Image failure: use PNG, JPEG, or WebP rather than SVG.
- Missing reviewer media: ensure VLLM_EVAL_TASK_BUNDLE_ROOT and BUNDLE_ROOT point to the same absolute directory.
- Comparison issues: inspect both progress and failure endpoints; failed or truncated attempts cannot form a normal blind pair.
- Wrong model after switching: restart the application after changing VLLM_EVAL_RUNNER_MODEL_NAME.

The fake runner remains available for offline application tests, but it is not the live model validation procedure.
