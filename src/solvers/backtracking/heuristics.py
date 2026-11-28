"""Search guidance: which operations to branch on, in which order to try
them, and when to give up on a partial schedule (pruning).

These do not change what a correct answer is - they only decide which
branches the backtracking search visits first and which branches it can
safely skip.
"""

from __future__ import annotations

from src.models.operation import Operation
from src.models.problem_instance import ProblemInstance
from src.solvers.backtracking.constraints import available_operations, earliest_start
from src.solvers.backtracking.state import ScheduleState


def order_operations(
    candidates: list[Operation],
    state: ScheduleState,
    instance: ProblemInstance,
) -> list[Operation]:
    """Order the available operations so promising ones are tried first.

    Trying jobs with the most remaining work first tends to find good
    schedules early, and finding good schedules early makes the bound
    below prune much more of the tree.
    """
    def remaining_work(job: int) -> int:
        return sum(op.duration for op in instance.jobs[job][state.next_op[job]:])

    return sorted(candidates, key=lambda op: remaining_work(op.job_id), reverse=True)


def lower_bound(state: ScheduleState, instance: ProblemInstance) -> int:
    """Optimistic estimate of the best makespan reachable from this state.

    Whatever happens next, every job still needs its remaining processing
    time after its ready time, and every machine still needs its remaining
    processing time after it becomes free. The maximum of these is a lower
    bound: no schedule completed from this state can beat it.
    """
    bound = 0

    for job in range(instance.num_jobs):
        remaining = sum(op.duration for op in instance.jobs[job][state.next_op[job]:])
        bound = max(bound, state.job_ready[job] + remaining)

    machine_remaining = [0] * instance.num_machines
    for job in range(instance.num_jobs):
        for op in instance.jobs[job][state.next_op[job]:]:
            machine_remaining[op.machine_id] += op.duration
    for machine in range(instance.num_machines):
        bound = max(bound, state.machine_free[machine] + machine_remaining[machine])

    return bound


def branch_candidates(state: ScheduleState, instance: ProblemInstance) -> list[Operation]:
    """The operations worth branching on at this state, promising ones first.

    Branching on every available operation explodes the tree. Instead we
    look at the operation that could finish earliest: its machine is where
    the next real decision is. Only the operations competing for that same
    machine (each could start before the earliest one finishes) are worth
    branching on - everything else can wait. This still explores every
    active schedule, so an optimal schedule stays reachable, but the tree
    is dramatically smaller.
    """
    candidates = available_operations(state, instance)

    first_machine = None
    first_end = None
    for op in candidates:
        end = earliest_start(state, op.job_id, op.machine_id) + op.duration
        if first_end is None or end < first_end:
            first_end = end
            first_machine = op.machine_id

    conflict = [
        op for op in candidates
        if op.machine_id == first_machine
        and earliest_start(state, op.job_id, op.machine_id) < first_end
    ]
    return order_operations(conflict, state, instance)
