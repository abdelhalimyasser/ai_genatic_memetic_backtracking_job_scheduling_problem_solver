from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.core.parser.parser_errors import ParserError
from src.models.operation import Operation
from src.models.problem_instance import ProblemInstance


def _validate_file_path(file_path: str | Path) -> Path:
    """Validate that file_path points to an existing regular file."""
    path = Path(file_path)
    if not path.exists():
        raise ParserError(f"File not found: {path}")
    if not path.is_file():
        raise ParserError(f"Path is not a file: {path}")
    return path


def _normalize_target_names(instance_names: Iterable[str] | None) -> set[str] | None:
    """Normalize an optional collection of instance names into a set."""
    if instance_names is None:
        return None
    if isinstance(instance_names, (str, bytes)):
        raise ParserError("instance_names must be an iterable of strings, not a single string")
    return set(instance_names)


def _is_instance_header(line: str) -> bool:
    """Return True if a stripped line starts a jobshop1 instance header."""
    return line == "instance" or line.startswith(("instance ", "instance\t"))


def _is_blank_or_separator(line: str) -> bool:
    """Return True if a stripped line is blank or a decorative separator."""
    return not line or not line.strip("+-=")


def _parse_header_name(line: str, line_no: int, seen_names: set[str]) -> str:
    """Validate an instance header line, enforce uniqueness, and return the instance name."""
    parts = line.split()
    if len(parts) != 2:
        raise ParserError(f"Invalid instance header at line {line_no}: {line!r}")
    name = parts[1]
    if name in seen_names:
        raise ParserError(f"Duplicate instance '{name}' at line {line_no}")
    seen_names.add(name)
    return name


def list_jobshop1_instances(file_path: str | Path) -> list[str]:
    """Return ordered instance names declared in a jobshop1-formatted file."""
    path = _validate_file_path(file_path)
    seen_names: set[str] = set()
    names: list[str] = []

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if _is_instance_header(line):
                    names.append(_parse_header_name(line, line_no, seen_names))
    except (OSError, UnicodeDecodeError) as exc:
        raise ParserError(f"Failed to read file '{path}': {exc}") from exc

    if not names:
        raise ParserError("No instances found in jobshop1 file")

    return names


def parse_jobshop1(
    file_path: str | Path,
    instance_names: Iterable[str] | None = None,
) -> dict[str, ProblemInstance]:
    """Parse OR-Library jobshop1 instances into ProblemInstance objects."""
    path = _validate_file_path(file_path)
    target_names = _normalize_target_names(instance_names)

    instances: dict[str, ProblemInstance] = {}
    seen_names: set[str] = set()
    op_cls = Operation

    try:
        with path.open("r", encoding="utf-8") as fh:
            if target_names is not None and not target_names:
                for _ in fh:
                    pass
                return {}

            line_iter = ((line_no, raw_line.strip()) for line_no, raw_line in enumerate(fh, start=1))

            for line_no, line in line_iter:
                if not _is_instance_header(line):
                    continue

                name = _parse_header_name(line, line_no, seen_names)
                if target_names is not None and name not in target_names:
                    continue

                description: str | None = None
                for _, candidate in line_iter:
                    if not _is_blank_or_separator(candidate):
                        description = candidate
                        break
                if description is None:
                    raise ParserError(f"Missing description for instance '{name}'")

                try:
                    dims_line_no, dims_line = next(line_iter)
                except StopIteration:
                    raise ParserError(f"Missing job/machine counts for instance '{name}'") from None

                dims = dims_line.split()
                if len(dims) != 2:
                    raise ParserError(
                        f"Invalid dimensions in instance '{name}' at line {dims_line_no}: {dims_line}"
                    )
                try:
                    num_jobs, num_machines = int(dims[0]), int(dims[1])
                except ValueError as exc:
                    raise ParserError(
                        f"Non-integer dimensions in instance '{name}' at line {dims_line_no}: {dims_line}"
                    ) from exc

                if num_jobs <= 0 or num_machines <= 0:
                    raise ParserError(
                        f"Dimensions must be positive in instance '{name}' at line {dims_line_no}: "
                        f"{num_jobs}x{num_machines}"
                    )

                expected_len = num_machines * 2
                jobs: list[list[Operation]] = []
                for job_id in range(num_jobs):
                    try:
                        job_line_no, job_line = next(line_iter)
                    except StopIteration:
                        raise ParserError(f"Missing job {job_id} data in instance '{name}'") from None

                    tokens = job_line.split()
                    if len(tokens) != expected_len:
                        raise ParserError(
                            f"Expected {expected_len} numbers for job {job_id} in '{name}' "
                            f"at line {job_line_no}, got {len(tokens)}"
                        )
                    try:
                        nums = list(map(int, tokens))
                    except ValueError as exc:
                        raise ParserError(
                            f"Non-integer value in job {job_id} of instance '{name}' at line {job_line_no}"
                        ) from exc

                    ops: list[Operation] = []
                    for op_id, (machine_id, duration) in enumerate(zip(nums[0::2], nums[1::2])):
                        if machine_id < 0 or machine_id >= num_machines:
                            raise ParserError(
                                f"Invalid machine_id {machine_id} (expected 0..{num_machines - 1}) "
                                f"in job {job_id} of '{name}' at line {job_line_no}"
                            )
                        if duration < 0:
                            raise ParserError(
                                f"Negative duration {duration} in job {job_id} of '{name}' at line {job_line_no}"
                            )
                        ops.append(op_cls(job_id, op_id, machine_id, duration))

                    jobs.append(ops)

                try:
                    instances[name] = ProblemInstance(
                        name=name,
                        num_jobs=num_jobs,
                        num_machines=num_machines,
                        jobs=jobs,
                        description=description,
                    )
                except ValueError as exc:
                    raise ParserError(f"Invalid problem instance '{name}': {exc}") from exc

                if target_names is not None and len(instances) == len(target_names):
                    break
    except (OSError, UnicodeDecodeError) as exc:
        raise ParserError(f"Failed to read file '{path}': {exc}") from exc

    if not seen_names:
        raise ParserError("No instances found in jobshop1 file")

    if target_names is not None:
        missing = target_names - instances.keys()
        if missing:
            raise ParserError(
                f"Instance(s) not found in '{path}': {', '.join(sorted(missing))}"
            )

    return instances
