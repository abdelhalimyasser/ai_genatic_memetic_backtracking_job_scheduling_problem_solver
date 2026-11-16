import os
from src.models import Operation, ProblemInstance


def get_default_dataset_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base_dir, "dataset", "jobshop1.txt")


def get_instance_list(filepath=None):
    if filepath is None:
        filepath = get_default_dataset_path()

    names = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("instance "):
                    names.append(line.replace("instance ", ""))
    except FileNotFoundError:
        names = ["ft06", "abz5", "abz6", "abz7", "la01"]
    return names


def load_instance(filepath, instance_name) -> ProblemInstance:
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    target = f"instance {instance_name}"
    start_idx = None
    for i, line in enumerate(lines):
        if line == target:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Instance '{instance_name}' not found in {filepath}")

    dims = lines[start_idx + 3].split()
    num_jobs, num_machines = int(dims[0]), int(dims[1])

    jobs = []
    for j in range(num_jobs):
        raw_nums = list(map(int, lines[start_idx + 4 + j].split()))
        operations = []
        step_idx = 0
        for k in range(0, len(raw_nums), 2):
            op = Operation(
                job_id=j,
                step_index=step_idx,
                machine_id=raw_nums[k],
                duration=raw_nums[k + 1]
            )
            operations.append(op)
            step_idx += 1
        jobs.append(operations)

    return ProblemInstance(
        name=instance_name,
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs
    )