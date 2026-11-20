from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    job_id: int
    operation_id: int
    machine_id: int
    duration: int

    def __post_init__(self) -> None: # used as validation DTO for each object
        if self.job_id < 0:
            raise ValueError("job_id must be non-negative")
        if self.operation_id < 0:
            raise ValueError("operation_id must be non-negative")
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.duration < 0:
            raise ValueError("duration must be non-negative")
