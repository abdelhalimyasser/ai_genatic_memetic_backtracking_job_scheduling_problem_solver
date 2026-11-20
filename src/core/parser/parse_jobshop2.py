from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path

from src.core.parser.parse_jobshop1 import _normalize_target_names, _validate_file_path
from src.core.parser.parser_errors import ParserError
from src.models.operation import Operation
from src.models.problem_instance import ProblemInstance

_MODULUS = 2147483647
_MULTIPLIER = 16807

_HEADER_RE = re.compile(r"^(\d+)\s+jobs\s+(\d+)\s+machines$", re.IGNORECASE)
_SEED_RE = re.compile(r"^([+-]?\d+)\s+([+-]?\d+)\s+(ta\d+|ta4@)$", re.IGNORECASE)


def _unif(state: list[int], low: int, high: int) -> int:
    """Advance Taillard's multiplicative LCG state and return a uniform integer in [low, high]."""
    if not state:
        raise ValueError("state must contain a seed integer")
    seed = state[0]
    if not (0 < seed < _MODULUS):
        raise ValueError(f"seed must be in range 1..{_MODULUS - 1}, got {seed}")
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")

    seed = (seed * _MULTIPLIER) % _MODULUS
    state[0] = seed
    return low + int((seed / _MODULUS) * (high - low + 1))


def _validate_seed(seed: int, label: str, name: str, line_no: int | None = None) -> None:
    """Validate that a Taillard LCG seed is within 1.._MODULUS - 1."""
    if not (0 < seed < _MODULUS):
        loc = f" at line {line_no}" if line_no is not None else ""
        raise ParserError(
            f"Invalid {label} {seed} for '{name}'{loc} (expected 1..{_MODULUS - 1})"
        )


def _generate_taillard_instance(
    name: str, time_seed: int, machine_seed: int, num_jobs: int, num_machines: int
) -> ProblemInstance:
    """Generate a Taillard JSP instance deterministically from its two PRNG seeds."""
    if num_jobs <= 0 or num_machines <= 0:
        raise ParserError(
            f"num_jobs and num_machines must be positive for '{name}', got {num_jobs}x{num_machines}"
        )
    _validate_seed(time_seed, "time_seed", name)
    _validate_seed(machine_seed, "machine_seed", name)

    m = _MODULUS
    a = _MULTIPLIER
    t_seed = time_seed
    m_seed = machine_seed
    op_cls = Operation
    base_machines = list(range(num_machines))
    jobs: list[list[Operation]] = []

    for job_id in range(num_jobs):
        machines = base_machines.copy()
        for i in range(num_machines):
            m_seed = (m_seed * a) % m
            k = i + int((m_seed / m) * (num_machines - i))
            machines[i], machines[k] = machines[k], machines[i]

        ops: list[Operation] = []
        for op_id, machine_id in enumerate(machines):
            t_seed = (t_seed * a) % m
            duration = 1 + int((t_seed / m) * 99)
            ops.append(op_cls(job_id, op_id, machine_id, duration))
        jobs.append(ops)

    try:
        return ProblemInstance(
            name=name,
            num_jobs=num_jobs,
            num_machines=num_machines,
            jobs=jobs,
            description=f"Taillard {num_jobs}x{num_machines} instance ({name})",
        )
    except ValueError as exc:
        raise ParserError(f"Invalid Taillard instance '{name}': {exc}") from exc


def _iter_taillard_specs(path: Path) -> Iterator[tuple[str, int, int, int, int]]:
    """Stream and validate Taillard instance specifications from a jobshop2 file."""
    current_jobs: int | None = None
    current_machines: int | None = None
    seen_names: set[str] = set()

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                first = stripped[0]
                if not (first.isdigit() or first == "+" or first == "-"):
                    continue

                header_match = _HEADER_RE.match(stripped)
                if header_match:
                    num_jobs = int(header_match.group(1))
                    num_machines = int(header_match.group(2))
                    if num_jobs <= 0 or num_machines <= 0:
                        raise ParserError(
                            f"Dimensions must be positive at line {line_no}: {num_jobs}x{num_machines}"
                        )
                    current_jobs = num_jobs
                    current_machines = num_machines
                    continue

                seed_match = _SEED_RE.match(stripped)
                if not seed_match:
                    continue

                time_seed = int(seed_match.group(1))
                machine_seed = int(seed_match.group(2))
                raw_name = seed_match.group(3)
                # Normalize upstream typo "ta4@" in OR-Library jobshop2.txt to "ta42".
                name = "ta42" if raw_name.lower() == "ta4@" else raw_name

                if current_jobs is None or current_machines is None:
                    raise ParserError(
                        f"Seed entry for '{name}' at line {line_no} precedes any dimensions header"
                    )
                if name in seen_names:
                    raise ParserError(f"Duplicate instance '{name}' at line {line_no}")

                _validate_seed(time_seed, "time_seed", name, line_no)
                _validate_seed(machine_seed, "machine_seed", name, line_no)

                seen_names.add(name)
                yield name, time_seed, machine_seed, current_jobs, current_machines
    except (OSError, UnicodeDecodeError) as exc:
        raise ParserError(f"Failed to read file '{path}': {exc}") from exc

    if not seen_names:
        raise ParserError("No Taillard seeds found in jobshop2 file")


def list_jobshop2_instances(file_path: str | Path) -> list[str]:
    """Return ordered Taillard instance names declared in a jobshop2-formatted file."""
    path = _validate_file_path(file_path)
    return [name for name, *_ in _iter_taillard_specs(path)]


def parse_jobshop2(
    file_path: str | Path,
    instance_names: Iterable[str] | None = None,
) -> dict[str, ProblemInstance]:
    """Parse OR-Library jobshop2 seeds and generate Taillard ProblemInstance objects."""
    path = _validate_file_path(file_path)
    target_names = _normalize_target_names(instance_names)

    if target_names is not None and not target_names:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for _ in fh:
                    pass
        except (OSError, UnicodeDecodeError) as exc:
            raise ParserError(f"Failed to read file '{path}': {exc}") from exc
        return {}

    instances: dict[str, ProblemInstance] = {}
    with closing(_iter_taillard_specs(path)) as specs:
        for name, time_seed, machine_seed, num_jobs, num_machines in specs:
            if target_names is not None and name not in target_names:
                continue

            instances[name] = _generate_taillard_instance(
                name=name,
                time_seed=time_seed,
                machine_seed=machine_seed,
                num_jobs=num_jobs,
                num_machines=num_machines,
            )

            if target_names is not None and len(instances) == len(target_names):
                break

    if target_names is not None:
        missing = target_names - instances.keys()
        if missing:
            raise ParserError(
                f"Instance(s) not found in '{path}': {', '.join(sorted(missing))}"
            )

    return instances
