"""Materialize validated task bundles as immutable database snapshots."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from .errors import ManifestValidationError, ValidationIssue
from .manifest import (
    ManifestTask,
    calculate_manifest_hash,
    calculate_snapshot_hash,
    load_manifest,
    resolve_assets,
)
from .models import MediaAsset, Task, TaskInput, TaskSnapshot


def import_task_bundle(
    session: Session,
    *,
    bundle_root: str | Path,
    manifest_path: str | Path = "manifest.jsonl",
) -> TaskSnapshot:
    """Validate and import a task bundle without partially persisting invalid data."""

    root = Path(bundle_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise ManifestValidationError(
            [
                ValidationIssue(
                    location="manifest",
                    code="manifest_path_escape",
                    message="manifest must be located inside the bundle",
                )
            ]
        ) from exc

    tasks = load_manifest(manifest)
    assets = resolve_assets(tasks, root)
    manifest_hash = calculate_manifest_hash(tasks)
    snapshot_hash = calculate_snapshot_hash(manifest_hash, assets)

    existing = session.scalar(
        select(TaskSnapshot).where(TaskSnapshot.snapshot_hash == snapshot_hash)
    )
    if existing is not None:
        return existing

    snapshot = TaskSnapshot(
        source_name=manifest.name,
        manifest_hash=manifest_hash,
        snapshot_hash=snapshot_hash,
        task_count=len(tasks),
    )
    session.add(snapshot)
    session.flush()

    asset_records = {
        asset.relative_path: MediaAsset(
            snapshot_id=snapshot.id,
            relative_path=asset.relative_path,
            original_filename=asset.original_filename,
            media_type=asset.media_type,
            content_hash=asset.content_hash,
            size_bytes=asset.size_bytes,
        )
        for asset in assets
    }
    session.add_all(asset_records.values())

    for task_order, manifest_task in enumerate(tasks):
        task = _materialize_task(snapshot, manifest_task, task_order, asset_records)
        session.add(task)

    session.flush()
    return snapshot


def _materialize_task(
    snapshot: TaskSnapshot,
    manifest_task: ManifestTask,
    task_order: int,
    asset_records: dict[str, MediaAsset],
) -> Task:
    task = Task(
        snapshot=snapshot,
        stable_task_id=manifest_task.task_id,
        category=manifest_task.category,
        prompt=manifest_task.prompt,
        system_instruction=manifest_task.system_instruction,
        operator_notes=manifest_task.operator_notes,
        source_metadata_json=manifest_task.source_metadata,
        task_order=task_order,
    )
    for ordinal, task_input in enumerate(manifest_task.inputs):
        normalized_path = (
            PurePosixPath(task_input.path).as_posix() if task_input.path is not None else None
        )
        task.inputs.append(
            TaskInput(
                ordinal=ordinal,
                input_type=task_input.input_type,
                text_content=task_input.text,
                transform_metadata_json=task_input.transform,
                media_asset=asset_records.get(normalized_path)
                if task_input.input_type == "image" and task_input.path is not None
                else None,
            )
        )
    return task
