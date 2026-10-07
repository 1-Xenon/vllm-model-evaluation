# Stateless Multimodal Model Evaluation — Design Specification

**Status:** Engineering handoff draft  
**Date:** 7 October 2026  
**Scope:** Initial evaluation of two locally hosted fine-tuned vision-language models (VLMs)

## 1. Purpose and problem statement

The team needs to understand how two fine-tuned VLMs behave on representative operational tasks before investing in a formal evaluation dataset or integrating either model into production pipelines. Inputs consist of a prompt and associated text and/or multimedia. Outputs may include a final answer and an emitted reasoning trace.

There is currently no golden evaluation set. The immediate goal is therefore to gather initial impressions, identify strengths and failure modes, and collect blind side-by-side judgments from subject matter experts (SMEs). Basic speed measurements should accompany the saved outputs to reveal practical tradeoffs between answer quality and responsiveness.

The application should turn informal “vibes testing” into a traceable exercise: the team should be able to establish exactly which input, model configuration, and generated output a reviewer assessed, and why they preferred one result. It should make useful evaluation easy without requiring formal benchmark development first.

The application is successful if it helps answer:

- Which model produces more useful and acceptable outputs for each intended task category?
- What recurring mistakes, omissions, or unsupported claims does each model make?
- Do reviewers agree, and where does domain judgment remain uncertain?
- What latency and generation-speed tradeoffs accompany these outputs under recorded serving conditions?
- Which examples and reviewer observations should inform subsequent fine-tuning, acceptance criteria, and a future golden evaluation set?

Preference judgments are evidence about the selected cases and reviewers. They do not by themselves establish objective accuracy, production readiness, or general superiority. Both models may produce unacceptable outputs on the same case.

## 2. Intended use and users

The models are intended for **stateless pipelines**, such as image captioning, article summarisation, and object recognition. Each task is independent. The application must not carry conversational history between tasks. An endpoint may encode an independent request using a messages array; this does not imply a multi-turn chat workflow.

An evaluation operator prepares tasks, configures model runs, monitors completion, and creates comparison sessions. SMEs inspect the original inputs and saved outputs, judge usefulness and acceptability, and record comments. An analyst or engineer examines the resulting quality observations and speed measurements.

The normal workflow is deliberately asynchronous. Model A can be run against the entire task set, unloaded, and replaced by model B, which is then run against the same task set. Human review occurs after generation and does not require either model to remain loaded. This accommodates hardware that cannot host both models simultaneously.

## 3. Scope and boundaries

### Required for the initial release

- Import and validate a reusable collection of independent tasks.
- Call configurable local OpenAI-compatible endpoints, initially vLLM.
- Execute and persist separate runs for two models, including sequential deployment.
- Capture answers, emitted reasoning where available, basic speed metrics, and failures.
- Present saved results in a blind side-by-side review interface.
- Collect and export reviewer feedback and run results.
- Operate as a containerised application in an offline environment.
- Validate the workflow using suitable open-source models before loading the fine-tuned models.

### Outside the initial scope

Multi-turn chat, model training, automatic model swapping or GPU scheduling, production load testing, automated LLM judging, and comprehensive benchmark integration are not required. The system does not need to create golden answers automatically. Explicit extensions can be added later without complicating the initial user workflow.

Model serving is an external dependency. The application need not package vLLM or model weights inside its own container, but its deployment instructions must explain how it connects to separately hosted local models.

## 4. End-to-end workflow

1. **Prepare and freeze tasks.** Define prompts, media, task categories, and shared evaluation settings. Validate that all assets exist and are supported. Assign a version or immutable snapshot to the task set.
2. **Configure model A.** Record the actual checkpoint identity, endpoint model name, generation settings, and serving environment. Confirm readiness and perform warm-up requests excluded from measured results.
3. **Run model A.** Execute the task set, capture streamed output and timing, and persist each attempt. Interrupted runs can resume without silently replacing existing results.
4. **Swap models externally.** The operator unloads A and loads B as needed. The application retains all completed work.
5. **Configure and run model B.** Confirm its identity and use the same task snapshot. Any model-specific settings or preprocessing differences must be recorded.
6. **Prepare comparison.** Pair selected results by stable task ID. Make missing, failed, and truncated cases visible to the operator. Freeze the selected result attempts for review.
7. **Conduct blind review.** SMEs inspect the shared input and anonymous outputs, submit judgments, and continue later if necessary.
8. **Analyse and export.** Examine preferences, acceptability, errors, reviewer disagreements, and timing by task category. Use observations to improve the next evaluation cycle.

## 5. Task and input requirements

Each task must contain a stable identifier, category, prompt, and any ordered text/media inputs. Optional fields may include system instructions, source metadata, and operator notes. Notes or expected outcomes that could anchor reviewer judgments should remain operator-only unless intentionally included in the review protocol.

The import format is an engineering choice; a documented JSON/JSONL manifest with local asset references is a reasonable option. Validation must identify duplicate IDs, missing files, invalid settings, and unsupported media before a batch begins. Asset references must remain usable after moving the evaluation bundle to another host.

Text-only and image-plus-text tasks should be supported initially, covering the named use cases. The design should allow multiple images and preserve their order. Video and audio support should be explicitly agreed for the first release rather than assumed from the term “multimedia.” Unsupported modalities must produce a clear validation error.

Retain original assets and a content hash or equivalent identity. Record model-facing transformations such as resizing, cropping, compression, frame sampling, or audio conversion. Reviewers should see the source input, and be able to inspect the actual transformed input where the application controls a transformation that materially affects interpretation. Server-side transformations should be recorded through available configuration rather than claimed to be directly observable.

## 6. Model runner and output capture

Endpoint URL, served model name, authentication if required, and request settings must be configurable without changing application code. This specification assumes an **OpenAI-compatible API**; if the actual service instead exposes a different OpenAPI-described interface, an adapter will be needed.

Use streaming where supported to measure initial output latency. Persist the final answer, any emitted reasoning, completion status or finish reason, token usage where available, and sufficient raw response/event data to diagnose parsing and measurement issues. Credentials must not be included in saved request records or exports.

Reasoning capture is conditional on what the model and endpoint expose. It is not a requirement to recover hidden internal reasoning. Distinguish “no reasoning emitted” from “reasoning not supported or not parsed.” Do not assume identical field names or reasoning delimiters across models and vLLM versions. Preserve raw output when separation is uncertain.

Timeouts, endpoint errors, empty responses, malformed streams, unsupported requests, and token-limit truncation must be explicit outcomes. Retry policy should be bounded and documented. Retried attempts must retain their own identity; the operator chooses which attempt enters comparison. Review assignments must not change when a new attempt is generated.

Runs should survive application restarts. Operators must be able to inspect progress and resume incomplete work. A reused endpoint alias must not be treated as proof of checkpoint identity: record operator-confirmed checkpoint information and endpoint-reported identity separately, and detect mismatches where feasible.

## 7. Speed measurements and comparability

The primary measurements are client-observed, taken in the runner using a monotonic clock. Browser rendering time must not determine the stored model measurements.

| Measurement | Intended definition |
|---|---|
| Time to first output (TTFT proxy) | Request dispatch to first nonempty generated content, including emitted reasoning. Exclude role-only, metadata-only, and heartbeat events. |
| Time to first answer | Request dispatch to first final-answer content, where the endpoint permits this distinction. |
| Total response latency | Request dispatch to a documented generation-completion boundary. Record when final usage or transport closure is used as the boundary. |
| Output token count | Server-reported generated tokens where available; document whether reasoning is included. |
| Decode tokens per second (TPS) | Approximate per-request rate: `(N − 1) / (total latency − TTFT)`, where N is the matching generated token count. |

Streaming events are not necessarily individual tokens, and parser buffering can delay the first visible content. Label TTFT as a client-observed proxy unless true token-level timing is available. Do not count chunks as tokens. If usage is absent, a model-matched tokenizer may provide an explicitly labelled estimate; otherwise leave TPS unavailable. TPS is undefined for insufficient token counts, incompatible timing/token boundaries, or a nonpositive denominator. Never substitute zero for missing measurements.

The operator should be able to set concurrency, request timeout, and generation settings. Begin with concurrency one for an interpretable baseline. Warm up both deployments consistently and record hardware, vLLM version, quantisation, parallelism, relevant cache settings, concurrent load, and preprocessing where known. Keep prompt content and shared settings aligned, while retaining any necessary model-specific differences.

Sequential runs are valid for this workflow but can be confounded by changes in machine load, caching, or serving configuration. Report results with those conditions. Token rates are not perfectly comparable across different tokenizers; answer latency remains useful alongside TPS. A single generation per case is sufficient for exploratory review, but repeat representative cases before making firm speed claims. Production capacity benchmarking is a separate exercise.

## 8. Blind human review

Display the complete shared prompt and usable media viewer above two saved answer panels. Preserve image detail needed for recognition tasks, support zoom where useful, and render outputs consistently. Treat model-produced content as untrusted display content rather than executable HTML.

Randomize left/right model placement per task and reviewer, persist the mapping, and keep it stable when the reviewer returns. Do not expose model names, checkpoint identifiers, timing, or identifying run metadata before judgment submission. Model wording may still reveal identity; blinding cannot guarantee that reviewers never infer the source.

Default to final-answer assessment, with emitted reasoning in clearly labelled collapsible panels. If reasoning is evaluated, collect that assessment separately from answer quality. A persuasive trace should not substitute for checking the output against the input.

The initial feedback form should be short:

- Overall preference: left, right, tie, or both unacceptable.
- Acceptability of each output: acceptable, unacceptable, or unsure.
- Optional error tags and comments; encourage an explanation for strong preferences or unacceptable outputs.

Useful starting error tags include unsupported claim, incorrect recognition, missed detail, incomplete output, instruction failure, and formatting failure. Allow task-specific rubrics to evolve. Provide skip/not qualified as a separate disposition so it is not counted as a tie. Keep infrastructure failures distinct from quality failures.

Track a reviewer identifier and judgment history. Multiple reviewers must be able to assess the same case independently without seeing one another’s judgments first. Some overlapping assignments are desirable to expose disagreement. Formal adjudication and advanced assignment management are optional extensions.

## 9. Records, summaries, and export

Keep logically distinct records for task snapshots, model runs, individual attempts, comparison sessions with selected result IDs, and reviewer judgments. Storage technology and schema are left to the engineer. Identity and relationships must survive export and re-import or backup restoration.

Provide a readable export of inputs/references, outputs, configurations, timing, status, and feedback. JSON/JSONL suits full-fidelity export; CSV can support analysis of tabular metrics and judgments. An operator export may contain the model mapping; reviewer-facing material must preserve blinding.

Summaries should include preferences and ties, per-model acceptability, common error tags, failure/truncation rates, and latency distributions by task category. Show sample counts and missing data. State whether an aggregate counts tasks or reviewer votes; multiple votes on one task must not silently appear as independent tasks. Avoid presenting a small exploratory sample as a definitive model leaderboard.

## 10. Containerisation and offline portability

The application must be containerised and transferable to a host with no internet access. Offline operation covers startup, task import, media viewing, generation against local endpoints, review, persistence, and export.

- Supply a reproducible container build and a simple documented startup configuration. Docker Compose or an equivalent approach is acceptable.
- Supply a transferable deployment bundle with prebuilt image(s), configuration examples, instructions, and required local assets. Startup must not fetch packages, fonts, JavaScript, tokenizers, models, or other runtime dependencies from the internet.
- Use locally bundled UI resources. No cloud authentication, SaaS storage, hosted judge, remote analytics, or telemetry may be required for core functionality.
- Store tasks, media, results, and reviews on documented persistent volumes. Container replacement must not erase evaluation data.
- Allow endpoint URLs, local paths, ports, and secrets to be configured at deployment. Do not hard-code host-specific addresses or embed credentials in images.
- Document supported operating environment, CPU architecture, container runtime, disk needs, and network connectivity to model servers. Portability applies within this documented compatibility envelope.
- Provide image import/startup instructions, backup/restore guidance, dependency/version inventory, and a troubleshooting guide for offline operation.

Model weights and the vLLM deployment can be transferred separately. The application itself should not require a GPU when inference runs on another local host. Any optional local inference dependency must be clearly separated from the core application.

## 11. Development and validation using OSS models

The engineer may develop and test the full application using suitable **open-source models before the fine-tuned models are available**. Use locally served text and vision-capable models appropriate to the accepted modalities and hardware. Record their identities and applicable licences. They are integration fixtures, not proxies for the fine-tuned models’ quality.

Demonstrate the real sequential workflow: run one model across a small task set, stop or replace that deployment, run the second model, then review the saved outputs with both deployments stopped. Synthetic endpoint fixtures can supplement live testing for edge cases such as malformed streams, usage omissions, timeouts, and reasoning fields.

The initial validation set should include text summarisation, image captioning, and object-recognition examples; multi-image and reasoning cases where supported; and deliberately failed or truncated requests. Keep development tasks clearly separate from the team’s eventual evaluation set.

An offline acceptance test must start the application from the transferred prebuilt bundle on a clean compatible host with internet access disabled, retaining access only to required local services. Exercise generation, review, restart/resume, media rendering, and export. Demonstrate that no dependency download or external account is needed.

## 12. Acceptance criteria and engineering handoff

The initial release is acceptable when:

1. The same frozen task set can be run against A and later B without simultaneous deployment or cross-task conversation state.
2. Saved records link every output, timing value, failure, and review to its exact task snapshot, run, and attempt.
3. Available answer/reasoning output is preserved; unavailable metrics and unsupported capabilities are clearly labelled.
4. Interruptions, retries, and restarts preserve completed work and existing review assignments.
5. SMEs can inspect source inputs, submit blind comparisons independently, and revisit their work with stable side assignment.
6. Exports support task-level analysis and distinguish missing results, infrastructure failures, quality failures, ties, and skipped judgments.
7. The sequential workflow has been demonstrated with OSS models before fine-tuned-model deployment.
8. The application passes the offline deployment test and retains data across container replacement.

Deliver source code, container build/startup materials, example configuration and task manifest, OSS smoke-test fixtures, an operator/reviewer guide, export documentation, and a concise validation report with known limitations.

The engineer should propose the UI framework, persistence mechanism, endpoint adapter, and deployment layout. These choices should favour maintainability and a small operational footprint; this document does not mandate Streamlit, SQLite, or a particular service decomposition. Logical separation between execution, stored evidence, and review is required, but separate deployed services are not.

Confirm during implementation: exact initial media modalities and output formats; endpoint/vLLM versions and reasoning support; expected task volume and reviewer concurrency; offline host architecture; and internal access-control needs. These details should refine the implementation without changing the primary purpose: a practical, traceable first comparison of stateless model behaviour.
