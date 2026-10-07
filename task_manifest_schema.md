# Task Manifest Schema

The initial release accepts JSON or JSONL task manifests. JSONL is the recommended format because each task is a separate record and large collections can be processed incrementally.

## Bundle layout

Asset references are relative to the evaluation bundle root:

```text
evaluation-bundle/
  manifest.jsonl
  media/
    street-001.jpg
```

The bundle is imported and validated before an immutable task snapshot is created. Audio and video references are rejected in the initial release.

## Task record

Each JSONL line is one task:

```json
{
  "task_id": "caption-001",
  "category": "image_captioning",
  "prompt": "Describe the image.",
  "inputs": [
    {"type": "image", "path": "media/street-001.jpg"}
  ],
  "system_instruction": "Answer concisely.",
  "source_metadata": {"source": "example"},
  "operator_notes": "Do not show this note to reviewers."
}
```

Required fields:

- `task_id`: stable identifier unique within the manifest.
- `category`: task category used for summaries.
- `prompt`: task prompt.
- `inputs`: non-empty ordered list of text and/or image inputs.

Optional fields:

- `system_instruction`: model-facing system instruction.
- `source_metadata`: structured source information.
- `operator_notes`: operator-only notes.

## Input records

Text input:

```json
{"type": "text", "text": "Use the following context: ..."}
```

Image input:

```json
{
  "type": "image",
  "path": "media/street-001.jpg",
  "transform": {"resize": {"width": 1024, "height": 768}}
}
```

Image paths must be relative, use `/` separators, and remain inside the bundle. The initial supported image types are JPEG, PNG, WebP, GIF, and SVG. The importer records the original asset path, media type, byte size, SHA-256 content hash, and any application-controlled transformation metadata.

## Validation behavior

Import rejects the complete bundle when any of the following occurs:

- Duplicate task IDs.
- Missing or unreadable assets.
- Absolute, traversal, or non-portable asset paths.
- Unsupported input types.
- Unsupported image media types.
- Missing required task or input fields.
- Invalid optional field types.
- Invalid JSON or JSONL records.

Validation errors include a location, stable error code, and human-readable message. No snapshot is persisted when validation fails.

## Snapshot identity

The importer calculates:

- A manifest hash from canonical task JSON.
- A snapshot hash from the manifest hash and sorted asset content hashes.

Re-importing an unchanged bundle returns the existing snapshot rather than creating a duplicate. Changing task content, asset content, or asset references creates a different snapshot identity.

