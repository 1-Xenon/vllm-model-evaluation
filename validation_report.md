# Validation report

## Current status

- Phases 0–4 are implemented and covered by automated tests.
- Phases 5–7 provide the operator API, blind-review service, local review page, exports, and summaries.
- Phase 8 provides the application image and persistent-volume deployment materials.
- Phase 9 fixtures and automated acceptance checks cover the offline application path.
- Live vLLM execution remains environment-dependent and is documented separately from the application acceptance path.

## Known limitations

- The vLLM endpoint contract remains the configurable OpenAI-compatible placeholder agreed for the initial release.
- The application does not perform model lifecycle management or GPU scheduling.
- Reviewer authentication and access control are not included in the initial release.
- The bundled reviewer page supports assignment loading, text and image input rendering, blind left/right/tie judgments, acceptability assessments, error tags, comments, and judgment submission. The API remains the stable integration surface for automation.
- CSV export is intentionally lossy compared with JSONL.
- Timing and TPS comparisons remain sensitive to server load, tokenizer differences, and generation settings.

## Acceptance scenarios

- Import the example bundle and verify an immutable snapshot hash.
- Run the fake runner for text and image tasks.
- Create a second run, freeze a complete comparison, and submit two judgment versions.
- Stop both model servers and reopen the stored review assignment.
- Export JSONL and CSV and verify stable record identities.
- Start the container with internet access disabled after its image has been built and confirm `/health` responds.
