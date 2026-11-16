from typing import List, Tuple
from src.models import ProblemInstance, ScheduledBlock


def calculate_makespan(chromosome: List[int], problem: ProblemInstance) -> int:
    op_count = [0] * problem.num_jobs
    job_time = [0] * problem.num_jobs
    machine_time = [0] * problem.num_machines

    for job_id in chromosome:
        step = op_count[job_id]
        op = problem.jobs[job_id][step]

        start_time = max(job_time[job_id], machine_time[op.machine_id])
        end_time = start_time + op.duration

        job_time[job_id] = end_time
        machine_time[op.machine_id] = end_time
        op_count[job_id] += 1

    return max(machine_time)


def decode_schedule_details(chromosome: List[int], problem: ProblemInstance) -> Tuple[int, List[ScheduledBlock]]:
    op_count = [0] * problem.num_jobs
    job_time = [0] * problem.num_jobs
    machine_time = [0] * problem.num_machines
    schedule_blocks = []

    for job_id in chromosome:
        step = op_count[job_id]
        op = problem.jobs[job_id][step]

        start_time = max(job_time[job_id], machine_time[op.machine_id])
        end_time = start_time + op.duration

        block = ScheduledBlock(
            job_id=job_id,
            operation_index=step,
            machine_id=op.machine_id,
            start_time=start_time,
            end_time=end_time
        )
        schedule_blocks.append(block)

        job_time[job_id] = end_time
        machine_time[op.machine_id] = end_time
        op_count[job_id] += 1

    return max(machine_time), schedule_blocks