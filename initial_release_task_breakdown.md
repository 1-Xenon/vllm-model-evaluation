# Initial Release Task Breakdown

**Status:** Implementation planning  
**Date:** 7 October 2026  
**Related documents:**

- `vllm_evaluation_design_spec.md`
- `initial_release_decisions.md`

This document breaks the initial release into small, independently trackable tasks. The design specification remains authoritative; the decisions document records the current implementation assumptions and configurable areas.

## Ultimate end goal

The project’s ultimate goal is to provide a practical, traceable, offline-capable evaluation application for comparing two locally hosted fine-tuned vision-language models on representative stateless tasks.

The completed system should allow an operator to freeze one task set, run model A and model B separately under recorded configurations, preserve each input, output, failure, and timing measurement, and then let subject matter experts review anonymous side-by-side results. It should capture reviewer preferences, acceptability judgments, error patterns, and comments, while keeping model identities hidden until review is complete.

The resulting evidence should help the team understand each model’s useful capabilities, recurring failure modes, reviewer disagreement, and latency tradeoffs. It should support decisions about fine-tuning, acceptance criteria, and the design of a future golden evaluation set. It is an exploratory comparison tool, not an automatic accuracy benchmark, production load-testing system, or guarantee of model superiority.

## Phase 0: Project foundation

- Create the application directory structure.
- Select and document the initial framework and persistence technology.
- Add dependency management.
- Add environment and configuration loading.
- Define configuration precedence: defaults, config file, environment variables, and secrets.
- Add structured logging.
- Add standard error types and error responses.
- Add unit-test and integration-test setup.
- Add formatting and linting configuration.

## Phase 1: Core data model

- Define identifiers and status values.
- Define the task entity.
- Define the immutable task-snapshot entity.
- Define the media-asset entity.
- Define model-configuration records.
- Define model-run records.
- Define generation-attempt records.
- Define comparison-session records.
- Define reviewer-assignment records.
- Define reviewer-judgment records.
- Add database migrations.
- Add repositories or persistence services.
- Add foreign-key and uniqueness constraints.
- Add persistence and restart tests.

## Phase 2: Task manifest and assets

- Define the JSON/JSONL manifest schema.
- Define supported input types for text and images.
- Implement relative-path resolution.
- Implement media-existence checks.
- Implement media-type validation.
- Implement duplicate task-ID detection.
- Implement task-setting validation.
- Implement asset hashing.
- Implement immutable snapshot creation.
- Store original asset metadata and hashes.
- Record application-controlled transformations.
- Produce structured validation errors.
- Produce a small example evaluation bundle.
- Add import and snapshot tests.

## Phase 3: Runner abstraction

- Define the internal model-runner interface.
- Define request and normalized-response types.
- Define attempt lifecycle states.
- Implement the deterministic fake runner.
- Add fake streaming responses.
- Add fake reasoning responses.
- Add fake missing-usage responses.
- Add fake timeout and malformed-stream responses.
- Add fake truncation responses.
- Add placeholder OpenAI-compatible runner configuration.
- Add request-timeout handling.
- Add bounded-retry handling.
- Create distinct attempt records for retries.
- Preserve raw response/event data safely.
- Add runner unit and integration tests.

## Phase 4: Execution and measurements

- Create a run from a frozen task snapshot.
- Validate run configuration before starting.
- Implement sequential task execution.
- Implement configurable concurrency, defaulting to one.
- Implement warm-up requests.
- Implement monotonic timing capture.
- Implement TTFT-proxy measurement.
- Implement first-answer timing when available.
- Implement total-response latency.
- Record server-reported token usage.
- Implement TPS calculation rules.
- Leave unavailable metrics as null or missing.
- Persist progress after each attempt.
- Resume incomplete runs.
- Prevent completed attempts from being silently overwritten.
- Add run-progress and failure inspection.

## Phase 5: Operator workflow

- Import a task bundle through the application.
- List and inspect task snapshots.
- Configure a model run.
- Configure generation and timeout settings.
- Verify endpoint readiness through the placeholder interface.
- Start a run.
- View run progress.
- Inspect failures and incomplete tasks.
- Retry selected tasks.
- Select attempts for comparison.
- Freeze the comparison selection.
- Confirm that model A and model B reference the same snapshot.

## Phase 6: Blind review

- Create a comparison session.
- Pair selected attempts by stable task ID.
- Detect missing, failed, or truncated results.
- Generate randomized left/right placement.
- Persist reviewer-specific placement.
- Display the shared prompt.
- Display source images with adequate detail.
- Render model output safely as untrusted content.
- Hide model names and identifying metadata.
- Add collapsible reasoning panels.
- Add preference options.
- Add per-output acceptability options.
- Add error tags.
- Add comments.
- Add skip/not-qualified disposition.
- Save judgment history.
- Support returning to unfinished reviews.
- Add tests for blinding and stable assignments.

## Phase 7: Export and summaries

- Export full records as JSONL.
- Export tabular data as CSV.
- Export operator-only model mappings.
- Preserve blinding in reviewer-facing exports.
- Summarize preferences and ties.
- Summarize per-model acceptability.
- Summarize error tags.
- Summarize failures and truncation.
- Summarize latency by task category.
- Include sample counts and missing-data indicators.
- Distinguish task counts from reviewer-vote counts.
- Add export and re-import identity tests.

## Phase 8: Containerisation

- Create the application container.
- Bundle all UI resources locally.
- Add persistent-volume configuration.
- Add example deployment configuration.
- Add startup instructions.
- Add image-import instructions.
- Add backup and restore instructions.
- Document dependency and version inventory.
- Document supported host architecture and runtime.
- Add offline troubleshooting guidance.
- Verify startup without internet access.

## Phase 9: Validation

- Add text-summarisation fixtures.
- Add image-captioning fixtures.
- Add object-recognition fixtures.
- Add OSS-model integration fixtures when available.
- Test model A followed by model B sequentially.
- Test application restart during a run.
- Test resume after restart.
- Test review after both model deployments are stopped.
- Test export after review.
- Test malformed streams and missing usage.
- Test offline startup from a transferred bundle.
- Write the validation report.
- Record known limitations.

## First development milestone

The first milestone ends after task 52 and should deliver:

- Application skeleton.
- Configuration loading.
- Database model.
- Manifest validation.
- Immutable task snapshots.
- Deterministic fake runner.
- Placeholder endpoint interface.
- Automated tests.

This milestone creates a working foundation without depending on the final vLLM contract. The live vLLM adapter, review interface, container packaging, and offline acceptance test can then be added incrementally.
