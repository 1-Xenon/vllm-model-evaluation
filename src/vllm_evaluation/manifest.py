"""Task manifest parsing, validation, and portable asset inspection."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ManifestValidationError, ValidationIssue

SUPPORTED_IMAGE_TYPES = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/webp",
    }
)


@dataclass(frozen=True, slots=True)
class ManifestInput:
    """One ordered text or image input from a task manifest."""

    input_type: str
    text: str | None = None
    path: str | None = None
    transform: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ManifestTask:
    """Validated task data before it is persisted into a snapshot."""

    task_id: str
    category: str
    prompt: str
    inputs: tuple[ManifestInput, ...]
    system_instruction: str | None = None
    source_metadata: dict[str, Any] | None = None
    operator_notes: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        """Return the canonical manifest representation used for hashing."""

        value: dict[str, Any] = {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "inputs": [asdict(task_input) for task_input in self.inputs],
        }
        if self.system_instruction is not None:
            value["system_instruction"] = self.system_instruction
        if self.source_metadata is not None:
            value["source_metadata"] = self.source_metadata
        if self.operator_notes is not None:
            value["operator_notes"] = self.operator_notes
        return value


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """Validated local asset metadata used to materialize a snapshot."""

    relative_path: str
    absolute_path: Path
    media_type: str
    content_hash: str
    size_bytes: int
    original_filename: str


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for reproducible content hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def calculate_manifest_hash(tasks: list[ManifestTask]) -> str:
    return sha256_bytes(canonical_json([task.as_mapping() for task in tasks]))


def calculate_snapshot_hash(manifest_hash: str, assets: list[ResolvedAsset]) -> str:
    asset_values = [
        {"path": asset.relative_path, "content_hash": asset.content_hash}
        for asset in sorted(assets, key=lambda item: item.relative_path)
    ]
    return sha256_bytes(canonical_json({"manifest_hash": manifest_hash, "assets": asset_values}))


def _issue(issues: list[ValidationIssue], location: str, code: str, message: str) -> None:
    issues.append(ValidationIssue(location=location, code=code, message=message))


def _parse_input(value: Any, location: str, issues: list[ValidationIssue]) -> ManifestInput | None:
    if not isinstance(value, dict):
        _issue(issues, location, "input_not_object", "input must be an object")
        return None

    input_type = value.get("type")
    if input_type not in {"text", "image"}:
        _issue(
            issues,
            f"{location}.type",
            "unsupported_input_type",
            "input type must be 'text' or 'image'",
        )
        return None

    transform = value.get("transform")
    if transform is not None and not isinstance(transform, dict):
        _issue(issues, f"{location}.transform", "invalid_transform", "transform must be an object")
        transform = None

    if input_type == "text":
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            _issue(issues, f"{location}.text", "missing_text", "text input must be non-empty")
            return None
        return ManifestInput(input_type="text", text=text, transform=transform)

    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        _issue(issues, f"{location}.path", "missing_asset_path", "image input requires a path")
        return None
    return ManifestInput(input_type="image", path=path, transform=transform)


def _parse_task(value: Any, location: str, issues: list[ValidationIssue]) -> ManifestTask | None:
    if not isinstance(value, dict):
        _issue(issues, location, "task_not_object", "task must be an object")
        return None

    task_id = value.get("task_id")
    category = value.get("category")
    prompt = value.get("prompt")
    raw_inputs = value.get("inputs")
    if not isinstance(task_id, str) or not task_id.strip():
        _issue(issues, f"{location}.task_id", "missing_task_id", "task_id must be non-empty")
    if not isinstance(category, str) or not category.strip():
        _issue(issues, f"{location}.category", "missing_category", "category must be non-empty")
    if not isinstance(prompt, str):
        _issue(issues, f"{location}.prompt", "missing_prompt", "prompt must be a string")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        _issue(issues, f"{location}.inputs", "missing_inputs", "inputs must be a non-empty list")

    if not isinstance(task_id, str) or not isinstance(category, str) or not isinstance(prompt, str):
        return None
    if not isinstance(raw_inputs, list) or not raw_inputs:
        return None

    parsed_inputs = tuple(
        parsed
        for index, raw_input in enumerate(raw_inputs)
        if (parsed := _parse_input(raw_input, f"{location}.inputs[{index}]", issues)) is not None
    )
    if len(parsed_inputs) != len(raw_inputs):
        return None

    system_instruction = value.get("system_instruction")
    if system_instruction is not None and not isinstance(system_instruction, str):
        _issue(
            issues,
            f"{location}.system_instruction",
            "invalid_system_instruction",
            "system_instruction must be a string",
        )
        system_instruction = None

    source_metadata = value.get("source_metadata")
    if source_metadata is not None and not isinstance(source_metadata, dict):
        _issue(
            issues,
            f"{location}.source_metadata",
            "invalid_source_metadata",
            "source_metadata must be an object",
        )
        source_metadata = None

    operator_notes = value.get("operator_notes")
    if operator_notes is not None and not isinstance(operator_notes, str):
        _issue(
            issues,
            f"{location}.operator_notes",
            "invalid_operator_notes",
            "operator_notes must be a string",
        )
        operator_notes = None

    return ManifestTask(
        task_id=task_id,
        category=category,
        prompt=prompt,
        inputs=parsed_inputs,
        system_instruction=system_instruction,
        source_metadata=source_metadata,
        operator_notes=operator_notes,
    )


def _load_raw_records(path: Path) -> tuple[list[Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _issue(issues, "$", "manifest_unreadable", str(exc))
        return [], issues

    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _issue(
                    issues,
                    f"line {line_number}",
                    "invalid_json",
                    f"invalid JSON: {exc.msg}",
                )
        return records, issues

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        _issue(issues, "$", "invalid_json", f"invalid JSON: {exc.msg}")
        return [], issues

    if isinstance(document, list):
        return document, issues
    if isinstance(document, dict) and isinstance(document.get("tasks"), list):
        return document["tasks"], issues
    _issue(
        issues, "$", "invalid_manifest_root", 'JSON manifest must be a task list or {"tasks": []}'
    )
    return [], issues


def load_manifest(path: str | Path) -> list[ManifestTask]:
    """Load and validate the structural contents of a JSON or JSONL manifest."""

    manifest_path = Path(path)
    records, issues = _load_raw_records(manifest_path)
    tasks = [
        parsed
        for index, record in enumerate(records)
        if (parsed := _parse_task(record, f"tasks[{index}]", issues)) is not None
    ]

    task_ids = [task.task_id for task in tasks]
    seen: set[str] = set()
    for index, task_id in enumerate(task_ids):
        if task_id in seen:
            _issue(
                issues,
                f"tasks[{index}].task_id",
                "duplicate_task_id",
                f"duplicate task_id: {task_id}",
            )
        seen.add(task_id)

    if issues:
        raise ManifestValidationError(issues)
    return tasks


def _portable_relative_path(
    raw_path: str, location: str, issues: list[ValidationIssue]
) -> str | None:
    if "\\" in raw_path:
        _issue(issues, location, "non_portable_asset_path", "asset paths must use '/' separators")
        return None
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        _issue(issues, location, "unsafe_asset_path", "asset path must remain inside the bundle")
        return None
    return path.as_posix()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset_file:
        for chunk in iter(lambda: asset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_assets(tasks: list[ManifestTask], bundle_root: str | Path) -> list[ResolvedAsset]:
    """Resolve, validate, and hash image references in a task collection."""

    root = Path(bundle_root).resolve()
    issues: list[ValidationIssue] = []
    asset_paths: dict[str, str] = {}
    for task_index, task in enumerate(tasks):
        for input_index, task_input in enumerate(task.inputs):
            if task_input.input_type != "image" or task_input.path is None:
                continue
            location = f"tasks[{task_index}].inputs[{input_index}].path"
            relative_path = _portable_relative_path(task_input.path, location, issues)
            if relative_path is not None:
                asset_paths[relative_path] = location

    resolved_assets: list[ResolvedAsset] = []
    for relative_path, location in sorted(asset_paths.items()):
        absolute_path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
        try:
            absolute_path.relative_to(root)
        except ValueError:
            _issue(issues, location, "asset_path_escape", "asset path resolves outside the bundle")
            continue
        if not absolute_path.is_file():
            _issue(issues, location, "missing_asset", f"asset does not exist: {relative_path}")
            continue
        media_type, _ = mimetypes.guess_type(absolute_path.name)
        if media_type not in SUPPORTED_IMAGE_TYPES:
            _issue(
                issues,
                location,
                "unsupported_media_type",
                f"unsupported image media type for {relative_path}: {media_type or 'unknown'}",
            )
            continue
        resolved_assets.append(
            ResolvedAsset(
                relative_path=relative_path,
                absolute_path=absolute_path,
                media_type=media_type,
                content_hash=_hash_file(absolute_path),
                size_bytes=absolute_path.stat().st_size,
                original_filename=absolute_path.name,
            )
        )

    if issues:
        raise ManifestValidationError(issues)
    return resolved_assets
