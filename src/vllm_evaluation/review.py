"""Frozen comparison construction and reviewer-specific blind assignments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .errors import ValidationError
from .models import (
    AssignmentStatus,
    ComparisonItem,
    ComparisonSession,
    ComparisonStatus,
    GenerationAttempt,
    ReviewerAssignment,
    ReviewerJudgment,
    Task,
)


@dataclass(frozen=True, slots=True)
class SelectionIssue:
    task_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    session: ComparisonSession
    issues: tuple[SelectionIssue, ...]


def _attempts_by_task(session: Session, attempt_ids: list[str]) -> dict[str, GenerationAttempt]:
    if not attempt_ids:
        return {}
    attempts = list(
        session.scalars(
            select(GenerationAttempt)
            .options(joinedload(GenerationAttempt.task), joinedload(GenerationAttempt.run))
            .where(GenerationAttempt.id.in_(attempt_ids))
        )
    )
    return {attempt.task.stable_task_id: attempt for attempt in attempts}


def create_comparison(
    session: Session,
    *,
    snapshot_id: str,
    name: str,
    assignment_seed: str,
    model_a_attempt_ids: list[str],
    model_b_attempt_ids: list[str],
    created_by: str | None = None,
) -> SelectionResult:
    """Create a draft comparison and report unusable pairs before freezing."""

    a_by_task = _attempts_by_task(session, model_a_attempt_ids)
    b_by_task = _attempts_by_task(session, model_b_attempt_ids)
    tasks = list(
        session.scalars(
            select(Task).where(Task.snapshot_id == snapshot_id).order_by(Task.task_order)
        )
    )
    comparison = ComparisonSession(
        snapshot_id=snapshot_id,
        name=name,
        assignment_seed=assignment_seed,
        created_by=created_by,
    )
    session.add(comparison)
    session.flush()
    issues: list[SelectionIssue] = []
    for task in tasks:
        left = a_by_task.get(task.stable_task_id)
        right = b_by_task.get(task.stable_task_id)
        if left is None or right is None:
            issues.append(
                SelectionIssue(
                    task.stable_task_id, "missing_pair", "both model attempts are required"
                )
            )
            continue
        if left.run.snapshot_id != snapshot_id or right.run.snapshot_id != snapshot_id:
            issues.append(
                SelectionIssue(
                    task.stable_task_id,
                    "snapshot_mismatch",
                    "attempts must use the comparison snapshot",
                )
            )
            continue
        if left.status != "succeeded" or right.status != "succeeded":
            issues.append(
                SelectionIssue(
                    task.stable_task_id,
                    "non_success",
                    "failed and truncated attempts are excluded from review",
                )
            )
            continue
        comparison.items.append(
            ComparisonItem(task_id=task.id, left_attempt_id=left.id, right_attempt_id=right.id)
        )
    session.flush()
    return SelectionResult(session=comparison, issues=tuple(issues))


def freeze_comparison(session: Session, comparison_id: str) -> ComparisonSession:
    comparison = session.get(ComparisonSession, comparison_id)
    if comparison is None:
        raise ValidationError(f"comparison not found: {comparison_id}")
    if comparison.status != ComparisonStatus.DRAFT.value:
        raise ValidationError("only draft comparisons can be frozen")
    if not comparison.items:
        raise ValidationError("comparison must contain at least one complete pair")
    comparison.status = ComparisonStatus.FROZEN.value
    comparison.frozen_at = datetime.now(UTC)
    session.flush()
    return comparison


def _swap_for_reviewer(seed: str, reviewer_id: str, task_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{reviewer_id}:{task_id}".encode()).digest()
    return bool(digest[0] & 1)


def ensure_reviewer_assignments(
    session: Session, comparison_id: str, reviewer_id: str
) -> list[ReviewerAssignment]:
    comparison = session.get(ComparisonSession, comparison_id)
    if comparison is None:
        raise ValidationError(f"comparison not found: {comparison_id}")
    if comparison.status not in {ComparisonStatus.FROZEN.value, ComparisonStatus.OPEN.value}:
        raise ValidationError("comparison must be frozen before review")
    comparison.status = ComparisonStatus.OPEN.value
    existing = {
        assignment.comparison_item_id: assignment
        for assignment in session.scalars(
            select(ReviewerAssignment)
            .where(ReviewerAssignment.reviewer_id == reviewer_id)
            .join(ComparisonItem)
            .where(ComparisonItem.comparison_session_id == comparison_id)
        )
    }
    assignments: list[ReviewerAssignment] = []
    for item in session.scalars(
        select(ComparisonItem)
        .where(ComparisonItem.comparison_session_id == comparison_id)
        .order_by(ComparisonItem.created_at)
    ):
        if item.id in existing:
            assignments.append(existing[item.id])
            continue
        swapped = _swap_for_reviewer(comparison.assignment_seed, reviewer_id, item.task_id)
        assignment = ReviewerAssignment(
            comparison_item_id=item.id,
            reviewer_id=reviewer_id,
            left_attempt_id=item.right_attempt_id if swapped else item.left_attempt_id,
            right_attempt_id=item.left_attempt_id if swapped else item.right_attempt_id,
        )
        session.add(assignment)
        assignments.append(assignment)
    session.flush()
    return assignments


def reviewer_view(session: Session, assignment_id: str) -> dict[str, Any]:
    assignment = session.scalar(
        select(ReviewerAssignment)
        .options(
            joinedload(ReviewerAssignment.comparison_item)
            .joinedload(ComparisonItem.task)
            .joinedload(Task.inputs),
            joinedload(ReviewerAssignment.left_attempt),
            joinedload(ReviewerAssignment.right_attempt),
        )
        .where(ReviewerAssignment.id == assignment_id)
    )
    if assignment is None:
        raise ValidationError(f"review assignment not found: {assignment_id}")
    item = assignment.comparison_item
    inputs = []
    for task_input in sorted(item.task.inputs, key=lambda value: value.ordinal):
        if task_input.input_type == "text":
            inputs.append({"type": "text", "text": task_input.text_content})
        elif task_input.media_asset is not None:
            inputs.append(
                {
                    "type": "image",
                    "relative_path": task_input.media_asset.relative_path,
                    "media_type": task_input.media_asset.media_type,
                }
            )
    return {
        "assignment_id": assignment.id,
        "reviewer_id": assignment.reviewer_id,
        "status": assignment.status,
        "snapshot_id": item.comparison_session.snapshot_id,
        "task": {
            "task_id": item.task.stable_task_id,
            "category": item.task.category,
            "prompt": item.task.prompt,
            "inputs": inputs,
        },
        "left": {
            "answer": assignment.left_attempt.final_answer,
            "reasoning": assignment.left_attempt.reasoning,
            "status": assignment.left_attempt.status,
        },
        "right": {
            "answer": assignment.right_attempt.final_answer,
            "reasoning": assignment.right_attempt.reasoning,
            "status": assignment.right_attempt.status,
        },
    }


def submit_judgment(
    session: Session, assignment_id: str, payload: dict[str, Any]
) -> ReviewerJudgment:
    assignment = session.get(ReviewerAssignment, assignment_id)
    if assignment is None:
        raise ValidationError(f"review assignment not found: {assignment_id}")
    version = (
        session.scalar(
            select(ReviewerJudgment)
            .where(ReviewerJudgment.assignment_id == assignment_id)
            .order_by(ReviewerJudgment.version.desc())
        )
        or None
    )
    next_version = (version.version + 1) if version else 1
    judgment = ReviewerJudgment(
        assignment_id=assignment_id,
        version=next_version,
        overall_preference=payload.get("overall_preference"),
        left_acceptability=payload.get("left_acceptability"),
        right_acceptability=payload.get("right_acceptability"),
        disposition=payload.get("disposition"),
        error_tags_json=payload.get("error_tags"),
        comments=payload.get("comments"),
        reasoning_assessment_json=payload.get("reasoning_assessment"),
    )
    session.add(judgment)
    assignment.status = (
        AssignmentStatus.SKIPPED.value
        if payload.get("disposition") == "skip"
        else AssignmentStatus.SUBMITTED.value
    )
    assignment.completed_at = datetime.now(UTC)
    session.flush()
    return judgment
