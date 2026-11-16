from dataclasses import dataclass, field
from typing import List
from src.models.schedule import ScheduledBlock


@dataclass
class SolverResult:
    algorithm_name: str
    best_chromosome: List[int]
    makespan: int
    execution_time: float
    blocks: List[ScheduledBlock] = field(default_factory=list)
    history: List[int] = field(default_factory=list)
    notes: str = ""