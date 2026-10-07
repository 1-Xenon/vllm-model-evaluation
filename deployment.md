# Deployment and offline operation

## Container

- Build on a connected staging host with `docker compose build`.
- Transfer the built image, this repository’s deployment files, and a data backup to the offline host.
- Load the image with `docker load` and start it with `docker compose up -d`.
- Keep `/data` on a persistent volume or host bind mount; replacing the container must not remove it.

The application container does not contain vLLM or model weights. Model servers are separate local services, as required by the design specification. Point `VLLM_EVAL_RUNNER_ENDPOINT_URL` and `VLLM_EVAL_RUNNER_MODEL_NAME` at the currently loaded local endpoint.

## Backup and restore

- Stop the application or use SQLite’s backup tooling before copying `evaluation.db`.
- Copy `/data/task-bundles`, `/data/results`, `/data/exports`, and `evaluation.db` together.
- Restore the complete directory before starting a replacement container.
- Apply migrations with `alembic upgrade head` when upgrading an existing installation.

## Compatibility envelope

- Linux host with Docker Engine or a compatible OCI runtime.
- `linux/amd64` is the initial tested architecture.
- Python 3.12 is used inside the application image.
- Model serving requires a separately configured local endpoint and appropriate GPU runtime.
- Core application workflows do not require internet access after the image and model images/weights have been transferred.

## Offline troubleshooting

- If startup fails, check that `/data` is writable and that the database URL points inside the persistent volume.
- If generation fails, check the model server health endpoint and the configured served model name.
- If images do not render, confirm the task bundle was transferred with its `media/` files and that the configured asset root is inside the container.
- Do not add cloud credentials or remote analytics to the offline deployment.
