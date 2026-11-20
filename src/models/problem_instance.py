from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass, field

from src.models.operation import Operation


@dataclass
class ProblemInstance:
    name: str
    num_jobs: int
    num_machines: int
    jobs: list[list[Operation]]
    description: str = ""
    extra_meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("instance name cannot be empty")
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("num_jobs and num_machines must be positive")
        if len(self.jobs) != self.num_jobs:
            raise ValueError(f"Expected {self.num_jobs} jobs, got {len(self.jobs)}")
        for j_idx, job_ops in enumerate(self.jobs):
            if len(job_ops) != self.num_machines:
                raise ValueError(f"Job {j_idx} expected {self.num_machines} operations, got {len(job_ops)}")

    @property
    def total_operations(self) -> int:
        return self.num_jobs * self.num_machines

    def get_operation(self, job_id: int, operation_id: int) -> Operation:
        return self.jobs[job_id][operation_id]

    def iter_operations(self) -> Iterator[Operation]:
        for job_ops in self.jobs:
            for op in job_ops:
                yield op
