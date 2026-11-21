from __future__ import annotations

from src.core.validator.validation_errors import ValidationError
from src.core.validator.validation_rules import (
    check_job_precedence,
    check_job_schedules_consistency,
    check_machine_no_overlap,
    check_operation_integrity,
)
from src.models.problem_instance import ProblemInstance
from src.models.scheduled_operation import ScheduledOperation


def validate_schedule(
    instance: ProblemInstance,
    machine_schedules: dict[int, list[ScheduledOperation]],
    job_schedules: dict[int, list[ScheduledOperation]] | None = None,
) -> tuple[bool, list[str]]:
    """Validate a complete Job Shop schedule against all problem constraints.

    Checks:
        1. Operation machine assignment and processing duration accuracy.
        2. Non-overlapping execution intervals on every machine.
        3. Completeness (no missing or duplicate operations).
        4. Sequential precedence order of operations within each job.
        5. Optional consistency between machine_schedules and job_schedules.
    """
    errors: list[str] = []
    scheduled_by_key: dict[tuple[int, int], ScheduledOperation] = {}

    for machine_id, ops in machine_schedules.items():
        if not (0 <= machine_id < instance.num_machines):
            errors.append(f"Invalid machine_id {machine_id}")
            continue

        sorted_ops = sorted(ops, key=lambda s: (s.start_time, s.end_time))

        for sched in sorted_ops:
            errors.extend(check_operation_integrity(instance, machine_id, sched))

            op = sched.operation
            if 0 <= op.job_id < instance.num_jobs and 0 <= op.operation_id < instance.num_machines:
                key = (op.job_id, op.operation_id)
                if key in scheduled_by_key:
                    errors.append(
                        f"Duplicate operation (job={op.job_id}, op={op.operation_id})"
                    )
                else:
                    scheduled_by_key[key] = sched

        errors.extend(check_machine_no_overlap(machine_id, sorted_ops))

    expected_total = instance.total_operations
    if len(scheduled_by_key) != expected_total:
        errors.append(
            f"Expected {expected_total} operations, found {len(scheduled_by_key)}"
        )

    for op in instance.iter_operations():
        if (op.job_id, op.operation_id) not in scheduled_by_key:
            errors.append(f"Missing operation (job={op.job_id}, op={op.operation_id})")

    errors.extend(check_job_precedence(instance, scheduled_by_key))

    if job_schedules is not None:
        errors.extend(
            check_job_schedules_consistency(instance, scheduled_by_key, job_schedules)
        )

    return (len(errors) == 0), errors


def validate_or_raise(
    instance: ProblemInstance,
    machine_schedules: dict[int, list[ScheduledOperation]],
    job_schedules: dict[int, list[ScheduledOperation]] | None = None,
) -> None:
    """Validate a schedule and raise ValidationError if any constraint is broken."""
    is_valid, errors = validate_schedule(instance, machine_schedules, job_schedules)
    if not is_valid:
        raise ValidationError(
            f"Schedule validation failed for '{instance.name}' with {len(errors)} error(s).",
            errors=errors,
        )
