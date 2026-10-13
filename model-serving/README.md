# Sequential vLLM model serving

The initial model pair is configured for the available RTX A4000 16 GB GPU:

- `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
- `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`

Only one model should occupy the GPU at a time. The application endpoint remains configurable and connects to the active model through the OpenAI-compatible API.

## Setup

- Ensure Docker has the NVIDIA runtime and verify `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- Copy `models.env.example` to `models.env` and adjust values if needed.
- Start one model with `./model-serving/serve-model.sh`.
- Wait for `curl http://127.0.0.1:8001/v1/models` to list the configured model.
- Run the evaluation model A tasks, stop the container, change `VLLM_MODEL`, then start model B.
- Keep `data/huggingface` so later starts reuse downloaded weights.

For this host, the verified settings are port `8001`, maximum context `4096`, one image per prompt, and `0.90` GPU memory utilization. Reduce the context length if a different model or workload causes an out-of-memory error.

## Application configuration

Set these values when using the live runner:

```bash
export VLLM_EVAL_RUNNER_TYPE=openai_compatible
export VLLM_EVAL_RUNNER_ENDPOINT_URL=http://127.0.0.1:8001/v1/chat/completions
export VLLM_EVAL_RUNNER_MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

Change only `VLLM_EVAL_RUNNER_MODEL_NAME` between sequential runs, and record the checkpoint identity in each model configuration. Do not run both containers on the same GPU unless the serving configuration has been deliberately changed and tested.

## Automatic model switching

The operator dashboard's **Automated evaluation** section can control this launcher and wait for
each model to become ready. It runs the selected bundle on one model before swapping to the next,
then creates and freezes all pairwise comparisons. The application must run on the Docker host
with access to Docker and this script.

The launcher also accepts explicit overrides after loading `models.env`:

```bash
./model-serving/serve-model.sh --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --port 8001 --detach
```

Without those arguments, the existing foreground workflow is unchanged. Both modes replace only
the container named by `VLLM_CONTAINER_NAME` (default `qwen-vl-evaluation`). Automated model and port
overrides do not rewrite `models.env`. Inspect startup problems with
`docker logs qwen-vl-evaluation` (or your configured container name).
