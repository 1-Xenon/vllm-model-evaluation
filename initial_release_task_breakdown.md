# Initial Release Task Breakdown

**Status:** Implementation planning  
**Date:** 7 October 2026  
**Related documents:**

- `vllm_evaluation_design_spec.md`
- `initial_release_decisions.md`

This document breaks the initial release into small, independently trackable tasks. The design specification remains authoritative; the decisions document records the current implementation assumptions and configurable areas.

## Phase 0: Project foundation

1. Create the application directory structure.
2. Select and document the initial framework and persistence technology.
3. Add dependency management.
4. Add environment and configuration loading.
5. Define configuration precedence: defaults, config file, environment variables, and secrets.
6. Add structured logging.
7. Add standard error types and error responses.
8. Add unit-test and integration-test setup.
9. Add formatting and linting configuration.

## Phase 1: Core data model

10. Define identifiers and status values.
11. Define the task entity.
12. Define the immutable task-snapshot entity.
13. Define the media-asset entity.
14. Define model-configuration records.
15. Define model-run records.
16. Define generation-attempt records.
17. Define comparison-session records.
18. Define reviewer-assignment records.
19. Define reviewer-judgment records.
20. Add database migrations.
21. Add repositories or persistence services.
22. Add foreign-key and uniqueness constraints.
23. Add persistence and restart tests.

## Phase 2: Task manifest and assets

24. Define the JSON/JSONL manifest schema.
25. Define supported input types for text and images.
26. Implement relative-path resolution.
27. Implement media-existence checks.
28. Implement media-type validation.
29. Implement duplicate task-ID detection.
30. Implement task-setting validation.
31. Implement asset hashing.
32. Implement immutable snapshot creation.
33. Store original asset metadata and hashes.
34. Record application-controlled transformations.
35. Produce structured validation errors.
36. Produce a small example evaluation bundle.
37. Add import and snapshot tests.

## Phase 3: Runner abstraction

38. Define the internal model-runner interface.
39. Define request and normalized-response types.
40. Define attempt lifecycle states.
41. Implement the deterministic fake runner.
42. Add fake streaming responses.
43. Add fake reasoning responses.
44. Add fake missing-usage responses.
45. Add fake timeout and malformed-stream responses.
46. Add fake truncation responses.
47. Add placeholder OpenAI-compatible runner configuration.
48. Add request-timeout handling.
49. Add bounded-retry handling.
50. Create distinct attempt records for retries.
51. Preserve raw response/event data safely.
52. Add runner unit and integration tests.

## Phase 4: Execution and measurements

53. Create a run from a frozen task snapshot.
54. Validate run configuration before starting.
55. Implement sequential task execution.
56. Implement configurable concurrency, defaulting to one.
57. Implement warm-up requests.
58. Implement monotonic timing capture.
59. Implement TTFT-proxy measurement.
60. Implement first-answer timing when available.
61. Implement total-response latency.
62. Record server-reported token usage.
63. Implement TPS calculation rules.
64. Leave unavailable metrics as null or missing.
65. Persist progress after each attempt.
66. Resume incomplete runs.
67. Prevent completed attempts from being silently overwritten.
68. Add run-progress and failure inspection.

## Phase 5: Operator workflow

69. Import a task bundle through the application.
70. List and inspect task snapshots.
71. Configure a model run.
72. Configure generation and timeout settings.
73. Verify endpoint readiness through the placeholder interface.
74. Start a run.
75. View run progress.
76. Inspect failures and incomplete tasks.
77. Retry selected tasks.
78. Select attempts for comparison.
79. Freeze the comparison selection.
80. Confirm that model A and model B reference the same snapshot.

## Phase 6: Blind review

81. Create a comparison session.
82. Pair selected attempts by stable task ID.
83. Detect missing, failed, or truncated results.
84. Generate randomized left/right placement.
85. Persist reviewer-specific placement.
86. Display the shared prompt.
87. Display source images with adequate detail.
88. Render model output safely as untrusted content.
89. Hide model names and identifying metadata.
90. Add collapsible reasoning panels.
91. Add preference options.
92. Add per-output acceptability options.
93. Add error tags.
94. Add comments.
95. Add skip/not-qualified disposition.
96. Save judgment history.
97. Support returning to unfinished reviews.
98. Add tests for blinding and stable assignments.

## Phase 7: Export and summaries

99. Export full records as JSONL.
100. Export tabular data as CSV.
101. Export operator-only model mappings.
102. Preserve blinding in reviewer-facing exports.
103. Summarize preferences and ties.
104. Summarize per-model acceptability.
105. Summarize error tags.
106. Summarize failures and truncation.
107. Summarize latency by task category.
108. Include sample counts and missing-data indicators.
109. Distinguish task counts from reviewer-vote counts.
110. Add export and re-import identity tests.

## Phase 8: Containerisation

111. Create the application container.
112. Bundle all UI resources locally.
113. Add persistent-volume configuration.
114. Add example deployment configuration.
115. Add startup instructions.
116. Add image-import instructions.
117. Add backup and restore instructions.
118. Document dependency and version inventory.
119. Document supported host architecture and runtime.
120. Add offline troubleshooting guidance.
121. Verify startup without internet access.

## Phase 9: Validation

122. Add text-summarisation fixtures.
123. Add image-captioning fixtures.
124. Add object-recognition fixtures.
125. Add OSS-model integration fixtures when available.
126. Test model A followed by model B sequentially.
127. Test application restart during a run.
128. Test resume after restart.
129. Test review after both model deployments are stopped.
130. Test export after review.
131. Test malformed streams and missing usage.
132. Test offline startup from a transferred bundle.
133. Write the validation report.
134. Record known limitations.

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
