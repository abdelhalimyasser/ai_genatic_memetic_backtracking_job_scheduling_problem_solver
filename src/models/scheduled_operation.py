from __future__ import annotations
from dataclasses import dataclass

from src.models import Operation

@dataclass
class ScheduledOperation:
    operation: Operation
    start_time: int
    end_time: int

    def __post_init__(self) -> None:
        if(self.start_time < 0) or (self.end_time < 0):
            raise ValueError("time must be non-negative")

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time
