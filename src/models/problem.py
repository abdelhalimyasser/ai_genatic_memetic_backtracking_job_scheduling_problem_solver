from dataclasses import dataclass
from typing import List


@dataclass
class Operation:
    job_id: int
    step_index: int
    machine_id: int
    duration: int


@dataclass
class ProblemInstance:
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[List[Operation]]