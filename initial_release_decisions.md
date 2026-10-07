# Initial Release Decisions and Configuration

**Status:** Working implementation decisions  
**Date:** 7 October 2026  
**Source of truth:** `vllm_evaluation_design_spec.md`

This document records decisions made while preparing the initial release. The design specification remains authoritative. This document makes its current interpretations and configurable areas explicit so implementation can proceed without hard-coding environment-specific assumptions.

## 1. Confirmed initial-release scope

The first release supports stateless evaluation tasks containing:

- Text-only inputs.
- Text with one image.
- Text with multiple ordered images.

Audio and video are outside the initial release. They must be rejected during task validation with a clear unsupported-modality error.

The application will support the following workflow:

1. Import and validate a task collection.
2. Create an immutable task snapshot.
3. Run the snapshot against model A.
4. Stop or replace model A externally.
5. Run the same snapshot against model B.
6. Select saved attempts for comparison.
7. Conduct blind side-by-side review.
8. Export results and feedback.

The application will not implement multi-turn conversation state, automatic model swapping, GPU scheduling, production load testing, automated LLM judging, or model training in the initial release.

## 2. Task manifest and snapshot decisions

The initial import format will be a documented JSON or JSONL manifest accompanied by local media assets.

Recommended bundle layout:

```text
evaluation-bundle/
  manifest.jsonl
  media/
    example-001.jpg
```

Task records must contain:

- A stable task ID.
- A task category.
- A prompt.
- Ordered text and/or image inputs.

Optional task fields may include system instructions, source metadata, and operator-only notes. Operator-only notes must not be shown to reviewers unless deliberately included in the review protocol.

Asset references should be relative to the evaluation bundle or another documented bundle root. Absolute host-specific paths should not be required.

Import validation must identify at least:

- Duplicate task IDs.
- Missing assets.
- Invalid task settings.
- Unsupported media types.
- Invalid or unusable asset references.

Import creates an immutable task snapshot. Runs reference the snapshot ID rather than reading a mutable source manifest directly. A later edit to the source manifest creates a new snapshot and must not change the inputs used by existing runs or reviews.

The application will retain original assets and a content hash or equivalent identity. Application-controlled transformations, such as resizing, cropping, compression, or frame sampling where applicable, must be recorded. Unknown server-side transformations must be labelled as not observable rather than inferred.

## 3. Endpoint and runner decisions

The exact vLLM version and complete endpoint contract remain provisional during the skeleton phase.

The application will expose a runner abstraction with a configurable OpenAI-compatible implementation and a deterministic fake runner for development and tests. The runner contract must cover:

- Endpoint URL.
- Served model name.
- Authentication configuration.
- Generation settings.
- Streaming preference.
- Request timeout.
- Bounded retry policy.
- Output normalization.
- Timing capture.
- Failure and truncation statuses.
- Optional token usage.
- Optional reasoning output.

The fake runner should be able to simulate successful responses, streaming chunks, missing usage, reasoning output, timeouts, malformed streams, empty responses, and truncation. This allows the application workflow to be developed before the fine-tuned models or final vLLM deployment are available.

The runner must preserve raw response or event data sufficiently to diagnose parsing and measurement issues. Credentials must never be stored in request records or exports.

## 4. Configurable settings

All deployment-specific and runtime-specific values should be provided through configuration rather than source-code changes.

### Endpoint and model configuration

- Endpoint URL.
- Served model name.
- Endpoint type or adapter name.
- Authentication mechanism and secret reference.
- Operator-confirmed checkpoint identity.
- Endpoint-reported model identity, when available.
- Model-specific preprocessing or generation overrides.

### Runner configuration

- Streaming enabled or disabled.
- Request timeout.
- Maximum retry count.
- Retryable error categories.
- Concurrency, initially defaulting to one.
- Warm-up count and warm-up behavior.
- Prompt and completion token limits.
- Temperature and other generation settings supported by the endpoint.
- Request headers or endpoint-specific options where required.

### Storage configuration

- Database or persistence location.
- Task and media storage root.
- Results and raw-event storage location.
- Export directory.
- Backup location or backup command configuration.
- Data-retention settings, subject to internal policy.

### Review configuration

- Reviewer identifier strategy.
- Assignment seed or randomization source.
- Whether reasoning panels are enabled.
- Available error tags.
- Whether comments are required for selected dispositions.
- Review-session and assignment expiration behavior, if needed.

### Deployment configuration

- Application listen address and port.
- Container runtime settings.
- Persistent volume mappings.
- Local asset directories.
- Local endpoint network addresses.
- Logging level and log destination.
- Offline deployment bundle locations.

Secrets must be supplied at deployment time through the documented configuration mechanism. They must not be committed to the repository or embedded in container images.

## 5. Required invariants

The following are not optional configuration choices because they protect evaluation validity:

- Tasks are independent; conversational history must not carry between tasks.
- Model A and model B use the same frozen task snapshot for a comparison.
- Retries receive distinct attempt identities.
- Existing completed attempts are not silently overwritten.
- Review assignments and left/right mappings remain stable when a reviewer returns.
- Model identity and run metadata remain hidden from reviewers before judgment submission.
- Model output is rendered as untrusted content.
- Infrastructure failures remain distinct from quality failures.
- Missing metrics remain missing; they must not be represented as zero.
- Browser rendering time must not determine stored model timings.
- Raw evidence and relationships must survive application restarts and container replacement.
- Reviewer-facing exports preserve blinding.

## 6. Measurement defaults

The initial implementation will use the specification's client-observed definitions:

- **TTFT proxy:** request dispatch to the first nonempty generated content, including reasoning where emitted.
- **Time to first answer:** request dispatch to the first final-answer content where distinguishable.
- **Total latency:** request dispatch to a documented completion boundary.
- **Output token count:** server-reported generated tokens where available.
- **TPS:** `(N - 1) / (total latency - TTFT)` only when token counts and timing boundaries are compatible.

Streaming chunks must not be treated as individual tokens. If usage is absent, a tokenizer-based estimate may be added only when clearly labelled and matched to the model. Otherwise TPS remains unavailable.

The initial baseline uses concurrency one. Hardware, vLLM version, quantisation, parallelism, cache settings, preprocessing, and concurrent load should be recorded when known.

## 7. Provisional values and open confirmations

The following may use documented defaults or placeholders during skeleton development:

- Exact vLLM version.
- Exact OpenAI-compatible streaming event fields.
- Exact reasoning field names and delimiters.
- Final task volume and total media size.
- Reviewer concurrency.
- Offline host operating system and CPU architecture.
- Container runtime and supported browser matrix.
- Internal authentication and access-control model.
- Long-term retention and backup policy.
- Production tokenizer availability.
- Server-side preprocessing visibility.

These values must remain configurable or isolated behind adapters. They must be confirmed before live fine-tuned-model evaluation and before the final offline acceptance test.

## 8. Recommended implementation defaults

Until the open confirmations are supplied, the implementation should use:

- JSONL task manifests.
- Relative paths inside a transferable evaluation bundle.
- SHA-256 or an equivalent documented content hash.
- A local persistent database.
- A single application container or small maintainable service layout.
- Concurrency one.
- A fake runner for automated tests.
- A configurable OpenAI-compatible runner skeleton.
- JSONL and CSV export.
- Local UI resources with no runtime internet dependency.
- No GPU requirement for the application container.

These defaults are implementation choices, not replacements for the specification's explicitly required behavior.
