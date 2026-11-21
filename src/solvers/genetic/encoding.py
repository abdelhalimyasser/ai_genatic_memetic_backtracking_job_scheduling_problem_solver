from __future__ import annotations

import random

from src.models.problem_instance import ProblemInstance
from src.models.scheduled_operation import ScheduledOperation


def create_chromosome(instance: ProblemInstance, r: random.Random) -> list[int]:
    """Create a valid OBR chromosome."""
    chromosome = [    # as it is represented as matrix
        job_id
        for job_id in range(instance.num_jobs)
        for _ in range(instance.num_machines)
    ]
    r.shuffle(chromosome)
    return chromosome


def _find_active_slot(machine_ops: list[ScheduledOperation], job_ready_time: int,
                      machine_ready_time: int, duration: int) -> tuple[int, int]:

    earliest_start = max(job_ready_time, machine_ready_time)

    prev_end = 0
    for idx, scheduled in enumerate(machine_ops):
        gap_start = prev_end
        gap_end = scheduled.start_time

        candidate_start = max(gap_start, job_ready_time)
        candidate_end = candidate_start + duration

        if candidate_end <= gap_end and candidate_start < earliest_start:
            return candidate_start, idx

        prev_end = max(prev_end, scheduled.end_time)

    return earliest_start, len(machine_ops)


def decode_active_obr(
    instance: ProblemInstance, chromosome: list[int]
) -> tuple[int, dict[int, list[ScheduledOperation]], dict[int, list[ScheduledOperation]]]:
    """Decode an Active OBR chromosome into feasible machine and job schedules."""
    next_op_idx = [0] * instance.num_jobs   # the next op index [0, 2], j0 -> never done any thing, j1 -> finished 2 next is 3
    job_ready = [0] * instance.num_jobs # when job become ready [7, 8], j0 -> will start after time 7, j1 -> will start after time 8
    machine_ready = [0] * instance.num_machines # when the machine ready, [10, 8], m0 -> will finish after 10, m1 -> will end after 8

    machine_schedules = {
        m: [] for m in range(instance.num_machines)
    }

    job_schedules = {
        j: [] for j in range(instance.num_jobs)
    }

    for job_id in chromosome:
        op_id = next_op_idx[job_id]
        op = instance.get_operation(job_id, op_id)
        m_id = op.machine_id

        start_time, insert_idx = _find_active_slot(
            machine_ops=machine_schedules[m_id],
            job_ready_time=job_ready[job_id],
            machine_ready_time=machine_ready[m_id],
            duration=op.duration,
        )
        end_time = start_time + op.duration

        scheduled = ScheduledOperation(
            operation=op,
            start_time=start_time,
            end_time=end_time,
        )

        machine_schedules[m_id].insert(insert_idx, scheduled)
        job_schedules[job_id].append(scheduled)

        job_ready[job_id] = max(job_ready[job_id], end_time)
        machine_ready[m_id] = max(machine_ready[m_id], end_time)
        next_op_idx[job_id] += 1

    makespan = max(
        (sched.end_time for ops in machine_schedules.values() for sched in ops),
        default=0,
    )

    return makespan, machine_schedules, job_schedules


def evaluate_fitness(instance: ProblemInstance, chromosome: list[int]) -> int:
    makespan, _, _ = decode_active_obr(instance, chromosome)
    return makespan
