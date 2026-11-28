"""Constraint handling for the Job Shop Scheduling Problem.

Two kinds of constraints must always hold:

1. Precedence constraints: the operations of one job must run in their
   fixed order, so a job can only offer its NEXT unscheduled operation.
2. Machine capacity constraints: a machine runs one operation at a time,
   so an operation cannot start before its machine is free.

Because of the precedence constraint, the only operations that can be
scheduled at any point are the "available" ones: the next operation of
each job that still has work left.
"""

from __future__ import annotations

from src.models.operation import Operation
from src.models.problem_instance import ProblemInstance
from src.solvers.backtracking.state import ScheduleState


def available_operations(state: ScheduleState, instance: ProblemInstance) -> list[Operation]:
    """Return the Operations that can be scheduled now (next op of each busy job)."""
    available = []
    for job in range(instance.num_jobs):
        op_index = state.next_op[job]
        if op_index < len(instance.jobs[job]):
            available.append(instance.jobs[job][op_index])
    return available


def earliest_start(state: ScheduleState, job: int, machine: int) -> int:
    """Earliest feasible start time: both the job and the machine must be free."""
    return max(state.job_ready[job], state.machine_free[machine])


def is_complete(state: ScheduleState, instance: ProblemInstance) -> bool:
    """True when every operation of every job has been scheduled."""
    return len(state.scheduled) == instance.total_operations
