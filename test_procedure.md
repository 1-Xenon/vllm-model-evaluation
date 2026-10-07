## 1\. Activate the project environment

From the project root:

```
cd vllm-model-evaluation
source .venv/bin/activate
```

Run the automated checks first:

```
pytest -q
ruff check src tests
ruff format --check src tests
alembic upgrade head
```

Expected result: all tests pass.

## 2\. Prepare the example bundle

The application’s media route expects bundles under the configured task-bundle root:

```
mkdir -p data/task-bundles/smoke-bundle/media

cp examples/evaluation-bundle/manifest.jsonl \
  data/task-bundles/smoke-bundle/manifest.jsonl

cp examples/evaluation-bundle/media/example.svg \
  data/task-bundles/smoke-bundle/media/example.svg
```

Set the bundle location:

```
export BUNDLE_ROOT="$PWD/data/task-bundles/smoke-bundle"
```

## 3\. Start the application

Use 8000 first, only use other ports if its already occupied

In Terminal 1:

```
cd /home/nic/projects/vllm-model-evaluation
source .venv/bin/activate

export VLLM_EVAL_RUNNER_TYPE=fake
export VLLM_EVAL_TASK_BUNDLE_ROOT="$PWD/data/task-bundles/smoke-bundle"

uvicorn vllm_evaluation.app:create_app \
  --factory \
  --host 127.0.0.1 \
  --port 8002
```

In Terminal 2, verify the application:

```
curl http://127.0.0.1:8002/health
```

Expected response:

```
{"status":"ok","runner_type":"fake"}
```

Useful pages:

- Application: http://127.0.0.1:8002
- API documentation: http://127.0.0.1:8002/docs
- Reviewer page: http://127.0.0.1:8002/review

## 4\. Import the task bundle

```
SNAPSHOT_ID=$(
  curl -sS -X POST http://127.0.0.1:8002/api/snapshots/import \
    -H 'Content-Type: application/json' \
    -d "{\"bundle_root\":\"$BUNDLE_ROOT\"}" |
  python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

echo "$SNAPSHOT_ID"
```

Inspect imported snapshots:

```
curl -sS http://127.0.0.1:8002/api/snapshots | python -m json.tool
```

## 5\. Create two model configurations

These use the fake runner, so the endpoint values are placeholders.

```
MODEL_A_CONFIG_ID=$(
  curl -sS -X POST http://127.0.0.1:8002/api/model-configs \
    -H 'Content-Type: application/json' \
    -d '{
      "display_name": "Model A",
      "endpoint_url": "http://model-a.invalid/v1/chat/completions",
      "served_model_name": "placeholder-model-a",
      "checkpoint_identity": "placeholder-checkpoint-a"
    }' |
  python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

MODEL_B_CONFIG_ID=$(
  curl -sS -X POST http://127.0.0.1:8002/api/model-configs \
    -H 'Content-Type: application/json' \
    -d '{
      "display_name": "Model B",
      "endpoint_url": "http://model-b.invalid/v1/chat/completions",
      "served_model_name": "placeholder-model-b",
      "checkpoint_identity": "placeholder-checkpoint-b"
    }' |
  python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

echo "$MODEL_A_CONFIG_ID"
echo "$MODEL_B_CONFIG_ID"
```

## 6\. Create and execute Model A’s run

```
RUN_A_ID=$(
  curl -sS -X POST http://127.0.0.1:8002/api/runs \
    -H 'Content-Type: application/json' \
    -d "{
      \"snapshot_id\":\"$SNAPSHOT_ID\",
      \"model_config_id\":\"$MODEL_A_CONFIG_ID\",
      \"run_label\":\"Model A smoke test\"
    }" |
  python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

curl -sS -X POST "http://127.0.0.1:8002/api/runs/$RUN_A_ID/execute" \
  -H 'Content-Type: application/json' \
  -d "{\"asset_root\":\"$BUNDLE_ROOT\",\"concurrency\":1}" |
  python -m json.tool
```

Check progress:

```
curl -sS \
  "http://127.0.0.1:8002/api/runs/$RUN_A_ID/progress" |
  python -m json.tool
```

Check failures:

```
curl -sS \
  "http://127.0.0.1:8002/api/runs/$RUN_A_ID/failures" |
  python -m json.tool
```

## 7\. Create and execute Model B’s run

```
RUN_B_ID=$(
  curl -sS -X POST http://127.0.0.1:8002/api/runs \
    -H 'Content-Type: application/json' \
    -d "{
      \"snapshot_id\":\"$SNAPSHOT_ID\",
      \"model_config_id\":\"$MODEL_B_CONFIG_ID\",
      \"run_label\":\"Model B smoke test\"
    }" |
  python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)

curl -sS -X POST "http://127.0.0.1:8002/api/runs/$RUN_B_ID/execute" \
  -H 'Content-Type: application/json' \
  -d "{\"asset_root\":\"$BUNDLE_ROOT\",\"concurrency\":1}" |
  python -m json.tool
```

## 8\. Obtain attempt IDs

The current API exposes run progress and failures, while attempt IDs can be inspected directly from SQLite:

```
ATTEMPTS_A=$(
  python -c '
import json, sqlite3, sys
db = sqlite3.connect("data/evaluation.db")
rows = db.execute(
    "select id from generation_attempts "
    "where run_id = ? and status = ? order by task_id, attempt_number",
    (sys.argv[1], "succeeded"),
).fetchall()
print(json.dumps([row[0] for row in rows]))
' "$RUN_A_ID"
)

ATTEMPTS_B=$(
  python -c '
import json, sqlite3, sys
db = sqlite3.connect("data/evaluation.db")
rows = db.execute(
    "select id from generation_attempts "
    "where run_id = ? and status = ? order by task_id, attempt_number",
    (sys.argv[1], "succeeded"),
).fetchall()
print(json.dumps([row[0] for row in rows]))
' "$RUN_B_ID"
)

echo "$ATTEMPTS_A"
echo "$ATTEMPTS_B"
```

## 9\. Create and freeze a comparison

```
COMPARISON_ID=$(
  curl -sS -X POST http://127.0.0.1:8002/api/comparisons \
    -H 'Content-Type: application/json' \
    -d "{
      \"snapshot_id\":\"$SNAPSHOT_ID\",
      \"name\":\"Model A versus Model B smoke test\",
      \"assignment_seed\":\"smoke-seed-001\",
      \"model_a_attempt_ids\":$ATTEMPTS_A,
      \"model_b_attempt_ids\":$ATTEMPTS_B,
      \"created_by\":\"operator\"
    }" |
  python -c 'import json,sys; print(json.load(sys.stdin)["comparison_id"])'
)

echo "$COMPARISON_ID"
```

Freeze the comparison:

```
curl -sS -X POST \
  "http://127.0.0.1:8002/api/comparisons/$COMPARISON_ID/freeze" |
  python -m json.tool
```

## 10\. Create a reviewer assignment

```
curl -sS \
  "http://127.0.0.1:8002/api/comparisons/$COMPARISON_ID/assignments/reviewer-1" |
  python -m json.tool
```

Capture the assignment ID:

```
ASSIGNMENT_ID=$(
  curl -sS \
    "http://127.0.0.1:8002/api/comparisons/$COMPARISON_ID/assignments/reviewer-1" |
  python -c 'import json,sys; print(json.load(sys.stdin)[0]["assignment_id"])'
)

echo "$ASSIGNMENT_ID"
```

Load the anonymous reviewer view:

```
curl -sS \
  "http://127.0.0.1:8002/api/review/assignments/$ASSIGNMENT_ID" |
  python -m json.tool
```

The reviewer payload should not contain model names, endpoint URLs, or timing metadata.

## 11\. Submit a judgment

```
curl -sS -X POST \
  "http://127.0.0.1:8002/api/review/assignments/$ASSIGNMENT_ID/judgment" \
  -H 'Content-Type: application/json' \
  -d '{
    "overall_preference": "left",
    "left_acceptability": "acceptable",
    "right_acceptability": "needs_revision",
    "disposition": "complete",
    "error_tags": ["verbosity"],
    "comments": "Model A was clearer for this example."
  }' |
  python -m json.tool
```

## 12\. View summaries and export results

Summary:

```
curl -sS \
  "http://127.0.0.1:8002/api/exports/$COMPARISON_ID/summary" |
  python -m json.tool
```

JSONL export:

```
curl -sS -X POST \
  "http://127.0.0.1:8002/api/exports/$COMPARISON_ID" \
  -H 'Content-Type: application/json' \
  -d "{
    \"format\":\"jsonl\",
    \"path\":\"$PWD/data/exports/smoke-test.jsonl\",
    \"include_model_mapping\":false
  }" |
  python -m json.tool
```

CSV export:

```
curl -sS -X POST \
  "http://127.0.0.1:8002/api/exports/$COMPARISON_ID" \
  -H 'Content-Type: application/json' \
  -d "{
    \"format\":\"csv\",
    \"path\":\"$PWD/data/exports/smoke-test.csv\"
  }" |
  python -m json.tool
```

Check the files:

```
ls -lh data/exports/
```

## 13\. Test the live 7B vLLM endpoint

Check whether the model server is running:

```
curl http://127.0.0.1:8001/v1/models
```

Test text generation:

```
curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: vllm-ok"
      }
    ],
    "max_tokens": 8,
    "temperature": 0
  }' |
  python -m json.tool
```

For image testing, vLLM expects a raster image such as PNG or JPEG. The example SVG is useful for application import testing, but vLLM does not decode it directly.

```
ffmpeg -loglevel error -y \
  -i examples/evaluation-bundle/media/example.svg \
  /tmp/vllm-eval-example.png

IMAGE_DATA=$(base64 -w0 /tmp/vllm-eval-example.png)

curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"Qwen/Qwen2.5-VL-7B-Instruct-AWQ\",
    \"messages\":[
      {
        \"role\":\"user\",
        \"content\":[
          {
            \"type\":\"text\",
            \"text\":\"Describe this image in one short sentence.\"
          },
          {
            \"type\":\"image_url\",
            \"image_url\":{
              \"url\":\"data:image/png;base64,${IMAGE_DATA}\"
            }
          }
        ]
      }
    ],
    \"max_tokens\":32,
    \"temperature\":0
  }" |
  python -m json.tool
```

## 14\. Run the application against vLLM

Stop the fake-runner application with `Ctrl+C`, then restart it with:

```
export VLLM_EVAL_RUNNER_TYPE=openai_compatible
export VLLM_EVAL_RUNNER_ENDPOINT_URL=http://127.0.0.1:8001/v1/chat/completions
export VLLM_EVAL_RUNNER_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
export VLLM_EVAL_TASK_BUNDLE_ROOT="$PWD/data/task-bundles/smoke-bundle"

uvicorn vllm_evaluation.app:create_app \
  --factory \
  --host 127.0.0.1 \
  --port 8002
```

You can then repeat the import, model configuration, run, and progress steps above.

The current runner settings are loaded when the application starts. When switching between the 3B and 7B models, restart the application after changing `VLLM_EVAL_RUNNER_MODEL_NAME`.

## 15\. Switch between the two models

Stop the active model:

```
docker stop qwen25-vl-7b
docker rm qwen25-vl-7b
```

Start the 3B model:

```
VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
VLLM_CONTAINER_NAME=qwen25-vl-3b \
./model-serving/serve-model.sh
```

Switch back by stopping the 3B container and starting the 7B container:

```
docker stop qwen25-vl-3b
docker rm qwen25-vl-3b

VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
VLLM_CONTAINER_NAME=qwen25-vl-7b \
./model-serving/serve-model.sh
```

Only run one model container at a time on the RTX A4000.
