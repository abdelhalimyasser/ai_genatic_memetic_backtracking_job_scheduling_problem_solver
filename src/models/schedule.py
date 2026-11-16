from dataclasses import dataclass


@dataclass
class ScheduledBlock:
    job_id: int
    operation_index: int
    machine_id: int
    start_time: int
    end_time: int