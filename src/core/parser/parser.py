from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.core.parser.parse_jobshop1 import (
    _is_instance_header,
    _validate_file_path,
    list_jobshop1_instances,
    parse_jobshop1,
)
from src.core.parser.parse_jobshop2 import (
    _HEADER_RE,
    list_jobshop2_instances,
    parse_jobshop2,
)
from src.core.parser.parser_errors import ParserError
from src.models.problem_instance import ProblemInstance


def _detect_format(file_path: str | Path) -> str:
    """Detect whether file_path uses job1 or job2 format by inspecting headers."""
    path = _validate_file_path(file_path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if _is_instance_header(stripped):
                    return "job1"
                if stripped[0].isdigit() and _HEADER_RE.match(stripped):
                    return "job2"
    except (OSError, UnicodeDecodeError) as exc:
        raise ParserError(f"Failed to read file '{path}': {exc}") from exc

    raise ParserError(f"Unable to detect dataset format for '{path}'")


def _resolve_format(file_path: str | Path, format_type: str) -> str:
    """Normalize format_type and resolve 'auto' against the dataset file."""
    if not isinstance(format_type, str):
        raise ParserError(f"Unsupported format type: {format_type!r}")
    normalized = format_type.strip().lower()
    if normalized in ("job1", "job2"):
        return normalized
    if normalized == "auto":
        return _detect_format(file_path)
    raise ParserError(f"Unsupported format type: {format_type!r}")


def load_instances(
    file_path: str | Path,
    format_type: str = "job1",
    instance_names: Iterable[str] | None = None,
) -> dict[str, ProblemInstance]:
    """Load all or selected JSP problem instances from a dataset file."""
    resolved = _resolve_format(file_path, format_type)
    if resolved == "job1":
        return parse_jobshop1(file_path, instance_names=instance_names)
    return parse_jobshop2(file_path, instance_names=instance_names)


def load_instance(
    file_path: str | Path,
    instance_name: str,
    format_type: str = "job1",
) -> ProblemInstance:
    """Load a single named JSP problem instance from a dataset file."""
    if not isinstance(instance_name, str) or not instance_name.strip():
        raise ParserError("instance_name must be a non-empty string")
    instances = load_instances(
        file_path,
        format_type=format_type,
        instance_names=(instance_name,),
    )
    return instances[instance_name]


def list_instances(file_path: str | Path, format_type: str = "job1") -> list[str]:
    """List ordered instance names present in a dataset file without generating job data."""
    resolved = _resolve_format(file_path, format_type)
    if resolved == "job1":
        return list_jobshop1_instances(file_path)
    return list_jobshop2_instances(file_path)
