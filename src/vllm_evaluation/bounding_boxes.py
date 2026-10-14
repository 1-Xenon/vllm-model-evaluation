"""Read display-only annotations from varied model answers without changing the evidence."""

from __future__ import annotations

import ast
import json
import math
import re
from difflib import SequenceMatcher
from typing import Any

_MAX_TEXT = 200_000
_MAX_BOXES = 200
_ALIASES = {
    "bbox",
    "bbox2d",
    "bboxes",
    "boundingbox",
    "boundingboxes",
    "boundingbox2d",
    "box",
    "box2d",
    "boundingrect",
    "boundingrectangle",
    "bboxcoordinates",
}
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _box_key(value: str) -> bool:
    folded = _key(value)
    if folded in _ALIASES:
        return True
    return (
        len(folded) >= 5
        and "box" in folded
        and any(
            SequenceMatcher(None, folded, alias).ratio() >= 0.9
            for alias in ("boundingbox", "boundingbox2d", "bbox2d")
        )
    )


def _structured(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, RecursionError):
            return None


def _label(value: dict, inherited: str = "Subject") -> str:
    fields = {_key(str(key)): item for key, item in value.items()}
    labels = []
    for key in ("type", "label", "class", "category", "objecttype", "name"):
        item = fields.get(key)
        if isinstance(item, str) and item.strip() and item.strip() not in labels:
            labels.append(item.strip()[:100])
        if len(labels) == 2:
            break
    return " · ".join(labels) or inherited


def _unit(value: dict, inherited: str) -> str:
    for key, item in value.items():
        if _key(str(key)) not in {"coordinatesystem", "coordinatescale", "bboxscale", "units"}:
            continue
        normalized = _key(str(item))
        if normalized in {"normalized", "relative", "01", "0to1"}:
            return "normalized"
        if normalized in {"pixel", "pixels", "absolute"}:
            return "pixels"
        if normalized in {"1000", "01000", "0to1000", "normalized1000"}:
            return "1000"
        if normalized in {"percent", "percentage", "0100"}:
            return "percent"
    return inherited


def _coords(value: Any, box_format: str = "xyxy") -> list[float] | None:
    if isinstance(value, str):
        value = value.strip()
        parsed = _structured(value)
        if parsed is not None and not isinstance(parsed, str):
            value = parsed
        else:
            text = value.strip("[]() ")
            if not re.fullmatch(rf"{_NUMBER}(?:\s*[,; ]\s*{_NUMBER}){{3}}", text):
                return None
            value = re.findall(_NUMBER, text)
    if isinstance(value, dict):
        fields = {_key(str(key)): item for key, item in value.items()}
        for keys, fmt in (
            (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
            (("x1", "y1", "x2", "y2"), "xyxy"),
            (("left", "top", "right", "bottom"), "xyxy"),
            (("x", "y", "width", "height"), "xywh"),
            (("cx", "cy", "width", "height"), "cxcywh"),
        ):
            if all(key in fields for key in keys):
                return _coords([fields[key] for key in keys], fmt)
        for key in ("coordinates", "coords", "values"):
            if key in fields:
                return _coords(fields[key], box_format)
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float, str)) for item in value):
        return None
    try:
        x1, y1, x2, y2 = map(float, value)
    except (ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if box_format == "xywh":
        x2, y2 = x1 + x2, y1 + y2
    elif box_format == "cxcywh":
        x1, y1, x2, y2 = x1 - x2 / 2, y1 - y2 / 2, x1 + x2 / 2, y1 + y2 / 2
    elif box_format != "xyxy":
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def extract_bounding_boxes(answer: str | None) -> dict[str, Any]:
    """Accept JSON, fenced JSON, numeric strings, named coordinates, and loose bbox lines.

    Coordinates are returned as xyxy with scale metadata; image dimensions belong to the browser.
    Image indexes are zero based. Multi-image tasks need an image choice for unassigned boxes.
    """
    boxes: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not answer:
        return {"boxes": boxes, "warnings": warnings}
    text = answer[:_MAX_TEXT]
    if len(answer) > _MAX_TEXT:
        warnings.append("Only the first 200,000 characters were inspected for boxes.")
    invalid = 0

    def walk(value, label="Subject", unit="auto", image=None, fmt="xyxy", depth=0):
        nonlocal invalid
        if depth > 30 or len(boxes) >= _MAX_BOXES:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item, label, unit, image, fmt, depth + 1)
            return
        if not isinstance(value, dict):
            return
        label = _label(value, label)
        unit = _unit(value, unit)
        fields = {_key(str(key)): item for key, item in value.items()}
        for key in ("imageindex", "imagenumber", "imagepath", "image", "filename"):
            item = fields.get(key)
            if key in {"imageindex", "imagenumber"}:
                if isinstance(item, int) and not isinstance(item, bool):
                    image = {"index": item - (1 if key == "imagenumber" else 0)}
                    break
            elif isinstance(item, str):
                image = {"path": item}
                break
        fmt = _key(str(fields.get("bboxformat", fields.get("boxformat", fmt))))
        for key, item in value.items():
            if _box_key(str(key)):
                coordinates = _coords(item, fmt)
                if coordinates is not None:
                    if len(boxes) < _MAX_BOXES:
                        boxes.append(
                            {
                                "coordinates": coordinates,
                                "label": _label(item, label) if isinstance(item, dict) else label,
                                "unit": _unit(item, unit) if isinstance(item, dict) else unit,
                                "image": image,
                                "field": str(key),
                            }
                        )
                elif (
                    isinstance(item, (list, tuple))
                    and item
                    and all(isinstance(child, (dict, list, tuple)) for child in item)
                ):
                    for child in item:
                        if isinstance(child, dict) and _coords(child, fmt) is None:
                            walk(child, label, unit, image, fmt, depth + 1)
                        else:
                            walk({"bbox": child}, label, unit, image, fmt, depth + 1)
                elif isinstance(item, dict) and any(_box_key(str(k)) for k in item):
                    walk(item, label, unit, image, fmt, depth + 1)
                elif item is not None and item != "" and item != []:
                    invalid += 1
            else:
                walk(item, label, unit, image, fmt, depth + 1)

    parsed = _structured(text.strip())
    if isinstance(parsed, (dict, list, tuple)):
        walk(parsed)
    else:
        # Parse complete fenced blocks or JSON embedded in surrounding explanatory prose.
        fragments = re.findall(r"```(?:json|python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        for fragment in fragments:
            walk(_structured(fragment.strip()))
        if not boxes:
            decoder = json.JSONDecoder()
            position = 0
            attempts = 0
            while position < len(text) and attempts < 100:
                match = re.search(r"[\[{]", text[position:])
                if not match:
                    break
                position += match.start()
                attempts += 1
                try:
                    document, length = decoder.raw_decode(text[position:])
                except (ValueError, RecursionError):
                    position += 1
                else:
                    walk(document)
                    position += length
        if not boxes:
            # Limited fallback for e.g. type: person, bbox_2d: "[10, 20, 30, 40]".
            pattern = (
                r"[\"']?([a-zA-Z_][\w -]{0,40})[\"']?\s*[:=]\s*[\"']?\s*[\[(]\s*"
                rf"({_NUMBER}(?:\s*[,; ]\s*{_NUMBER}){{3}})\s*[\])]"
            )
            for match in re.finditer(pattern, text):
                if not _box_key(match.group(1)):
                    continue
                start = (
                    max(text.rfind("\n", 0, match.start()), text.rfind("{", 0, match.start())) + 1
                )
                end = text.find("\n", match.end())
                line = text[start : end if end >= 0 else len(text)]
                label = re.search(
                    r"""["']?(?:type|label|class|category)["']?\s*[:=]\s*["']?([^,"'\n}\]]+)""",
                    line,
                )
                walk(
                    {"bbox": match.group(2), "type": label.group(1).strip() if label else "Subject"}
                )
    if invalid:
        warnings.append(f"Skipped {invalid} invalid or unsupported bounding box value(s).")
    if len(boxes) >= _MAX_BOXES:
        warnings.append("Showing at most 200 boxes.")
    return {"boxes": boxes, "warnings": warnings}
