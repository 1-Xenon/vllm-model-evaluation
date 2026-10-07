"""Full-fidelity exports and reviewer-safe summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .models import (
    ComparisonItem,
    GenerationAttempt,
    ModelRun,
    ReviewerAssignment,
)


def _attempt_record(attempt: GenerationAttempt) -> dict[str, Any]:
    return {
        "attempt_id": attempt.id,
        "run_id": attempt.run_id,
        "task_id": attempt.task.stable_task_id,
        "attempt_number": attempt.attempt_number,
        "status": attempt.status,
        "answer": attempt.final_answer,
        "reasoning": attempt.reasoning,
        "reasoning_status": attempt.reasoning_status,
        "finish_reason": attempt.finish_reason,
        "error_code": attempt.error_code,
        "error_message": attempt.error_message,
        "token_usage": attempt.token_usage_json,
        "ttft_seconds": attempt.ttft_seconds,
        "first_answer_seconds": attempt.first_answer_seconds,
        "total_latency_seconds": attempt.total_latency_seconds,
        "output_token_count": attempt.output_token_count,
        "tps": attempt.tps,
    }


def export_jsonl(
    session: Session,
    path: str | Path,
    *,
    comparison_id: str | None = None,
    include_model_mapping: bool = True,
) -> Path:
    path = Path(path)
    attempts = session.scalars(
        select(GenerationAttempt)
        .options(
            joinedload(GenerationAttempt.task),
            joinedload(GenerationAttempt.run).joinedload(ModelRun.model_config),
        )
        .order_by(GenerationAttempt.created_at)
    )
    records: list[dict[str, Any]] = []
    for attempt in attempts:
        record = _attempt_record(attempt)
        record["model"] = (
            {
                "display_name": attempt.run.model_config.display_name,
                "served_model_name": attempt.run.model_config.served_model_name,
            }
            if include_model_mapping
            else None
        )
        records.append({"record_type": "attempt", **record})
    if comparison_id:
        assignments = (
            session.scalars(
                select(ReviewerAssignment)
                .join(ComparisonItem)
                .where(ComparisonItem.comparison_session_id == comparison_id)
                .options(joinedload(ReviewerAssignment.judgments))
            )
            .unique()
            .all()
        )
        for assignment in assignments:
            for judgment in assignment.judgments:
                records.append(
                    {
                        "record_type": "judgment",
                        "assignment_id": assignment.id,
                        "reviewer_id": assignment.reviewer_id,
                        "version": judgment.version,
                        "overall_preference": judgment.overall_preference,
                        "left_acceptability": judgment.left_acceptability,
                        "right_acceptability": judgment.right_acceptability,
                        "disposition": judgment.disposition,
                        "error_tags": judgment.error_tags_json,
                        "comments": judgment.comments,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return path


def export_csv(session: Session, path: str | Path, *, comparison_id: str | None = None) -> Path:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    attempts = session.scalars(
        select(GenerationAttempt)
        .options(
            joinedload(GenerationAttempt.task),
            joinedload(GenerationAttempt.run).joinedload(ModelRun.model_config),
        )
        .order_by(GenerationAttempt.created_at)
    )
    for attempt in attempts:
        rows.append(
            {
                "record_type": "attempt",
                "task_id": attempt.task.stable_task_id,
                "run_id": attempt.run_id,
                "model": attempt.run.model_config.display_name,
                "attempt_number": attempt.attempt_number,
                "status": attempt.status,
                "answer": attempt.final_answer,
                "error_code": attempt.error_code,
                "ttft_seconds": attempt.ttft_seconds,
                "total_latency_seconds": attempt.total_latency_seconds,
                "tps": attempt.tps,
            }
        )
    if comparison_id:
        assignments = (
            session.scalars(
                select(ReviewerAssignment)
                .join(ComparisonItem)
                .where(ComparisonItem.comparison_session_id == comparison_id)
                .options(joinedload(ReviewerAssignment.judgments))
            )
            .unique()
            .all()
        )
        for assignment in assignments:
            for judgment in assignment.judgments:
                rows.append(
                    {
                        "record_type": "judgment",
                        "task_id": None,
                        "run_id": None,
                        "model": None,
                        "attempt_number": judgment.version,
                        "status": judgment.disposition,
                        "answer": judgment.overall_preference,
                        "error_code": json.dumps(judgment.error_tags_json),
                        "ttft_seconds": None,
                        "total_latency_seconds": None,
                        "tps": None,
                    }
                )
    fields = [
        "record_type",
        "task_id",
        "run_id",
        "model",
        "attempt_number",
        "status",
        "answer",
        "error_code",
        "ttft_seconds",
        "total_latency_seconds",
        "tps",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def comparison_summary(session: Session, comparison_id: str) -> dict[str, Any]:
    assignments = list(
        session.scalars(
            select(ReviewerAssignment)
            .join(ComparisonItem)
            .where(ComparisonItem.comparison_session_id == comparison_id)
            .options(joinedload(ReviewerAssignment.judgments))
        ).unique()
    )
    preferences = Counter()
    acceptability = Counter()
    errors = Counter()
    for assignment in assignments:
        if not assignment.judgments:
            continue
        judgment = assignment.judgments[-1]
        preferences[judgment.overall_preference or "missing"] += 1
        acceptability[f"left:{judgment.left_acceptability or 'missing'}"] += 1
        acceptability[f"right:{judgment.right_acceptability or 'missing'}"] += 1
        errors.update(judgment.error_tags_json or [])
    return {
        "comparison_id": comparison_id,
        "assignment_count": len(assignments),
        "judgment_count": sum(bool(a.judgments) for a in assignments),
        "preferences": dict(preferences),
        "acceptability": dict(acceptability),
        "error_tags": dict(errors),
        "counts_are_reviewer_votes": True,
    }
