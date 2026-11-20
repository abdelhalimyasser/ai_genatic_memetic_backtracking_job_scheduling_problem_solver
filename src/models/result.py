from __future__ import annotations
from dataclasses import dataclass, field

from src.models.scheduled_operation import ScheduledOperation


@dataclass
class ScheduleResult:
    instance_name: str
    algorithm_name: str
    makespan: int
    is_feasible: bool = True
    runtime_sec: float = 0.0
    machine_schedules: dict[int, list[ScheduledOperation]] = field(default_factory=dict)
    job_schedules: dict[int, list[ScheduledOperation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.makespan < 0:
            raise ValueError("makespan must be non-negative")
        if self.runtime_sec < 0:
            raise ValueError("runtime_sec must be non-negative")
