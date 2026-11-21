from __future__ import annotations

from src.models.problem_instance import ProblemInstance
from src.models.scheduled_operation import ScheduledOperation


def check_operation_integrity(
    instance: ProblemInstance,
    machine_id: int,
    sched: ScheduledOperation,
) -> list[str]:
    """Verify a single scheduled operation's identity, machine assignment, and duration."""
    errors: list[str] = []
    op = sched.operation

    machine_valid = 0 <= machine_id < instance.num_machines
    if not machine_valid:
        errors.append(f"Invalid machine_id {machine_id}")

    if not (0 <= op.job_id < instance.num_jobs and 0 <= op.operation_id < instance.num_machines):
        errors.append(
            f"Unknown operation (job={op.job_id}, op={op.operation_id}) "
            f"outside instance bounds ({instance.num_jobs}x{instance.num_machines})"
        )
        return errors

    expected_op = instance.get_operation(op.job_id, op.operation_id)

    if op.machine_id != expected_op.machine_id:
        errors.append(
            f"Job {op.job_id} op {op.operation_id} expects machine {expected_op.machine_id}, "
            f"operation specifies machine {op.machine_id}"
        )

    if machine_valid and machine_id != expected_op.machine_id:
        errors.append(
            f"Job {op.job_id} op {op.operation_id} belongs to machine {expected_op.machine_id}, "
            f"found on machine {machine_id}"
        )

    if op.duration != expected_op.duration:
        errors.append(
            f"Job {op.job_id} op {op.operation_id} definition duration mismatch: "
            f"got {op.duration}, expected {expected_op.duration}"
        )

    if sched.duration != expected_op.duration:
        errors.append(
            f"Job {op.job_id} op {op.operation_id} duration mismatch: "
            f"got {sched.duration}, expected {expected_op.duration}"
        )

    return errors


def check_machine_no_overlap(
    machine_id: int,
    sorted_ops: list[ScheduledOperation],
) -> list[str]:
    """Ensure operations assigned to the same machine do not overlap in time."""
    errors: list[str] = []
    prev_end = 0

    for sched in sorted_ops:
        op = sched.operation
        if sched.start_time < prev_end:
            errors.append(
                f"Overlap on machine {machine_id}: job {op.job_id} op {op.operation_id} "
                f"starts at {sched.start_time} before previous ends at {prev_end}"
            )
        prev_end = max(prev_end, sched.end_time)

    return errors


def check_job_precedence(
    instance: ProblemInstance,
    scheduled_by_key: dict[tuple[int, int], ScheduledOperation],
) -> list[str]:
    """Ensure each operation in a job starts only after its predecessor finishes."""
    errors: list[str] = []

    for job_id in range(instance.num_jobs):
        for op_id in range(1, instance.num_machines):
            prev_op = scheduled_by_key.get((job_id, op_id - 1))
            curr_op = scheduled_by_key.get((job_id, op_id))
            if prev_op and curr_op and curr_op.start_time < prev_op.end_time:
                errors.append(
                    f"Job {job_id} order error: op {op_id} starts at {curr_op.start_time} "
                    f"before op {op_id - 1} ends at {prev_op.end_time}"
                )

    return errors


def check_job_schedules_consistency(
    instance: ProblemInstance,
    scheduled_by_key: dict[tuple[int, int], ScheduledOperation],
    job_schedules: dict[int, list[ScheduledOperation]],
) -> list[str]:
    """Verify job_schedules matches machine_schedules completely and without duplicates."""
    errors: list[str] = []
    seen_keys: set[tuple[int, int]] = set()

    for job_id, j_ops in job_schedules.items():
        if not (0 <= job_id < instance.num_jobs):
            errors.append(f"Invalid job_id {job_id} in job_schedules")
            continue

        for sched in j_ops:
            op = sched.operation
            key = (op.job_id, op.operation_id)

            if op.job_id != job_id:
                errors.append(
                    f"Operation {key} placed under wrong job_id {job_id} in job_schedules"
                )

            if key in seen_keys:
                errors.append(f"Duplicate operation {key} in job_schedules")
                continue
            seen_keys.add(key)

            m_sched = scheduled_by_key.get(key)
            if m_sched is None:
                errors.append(f"Operation {key} missing from machine_schedules")
            elif (
                m_sched.start_time != sched.start_time
                or m_sched.end_time != sched.end_time
                or m_sched.operation != sched.operation
            ):
                errors.append(
                    f"Mismatch for {key} between machine and job schedules"
                )

    for key in scheduled_by_key:
        if key not in seen_keys:
            errors.append(f"Operation {key} missing from job_schedules")

    return errors
