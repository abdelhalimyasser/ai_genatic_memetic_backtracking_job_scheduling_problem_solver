from __future__ import annotations

import random


def job_based_crossover(
    parent1: list[int],
    parent2: list[int],
    num_jobs: int,
    rng: random.Random,
) -> list[int]:

    if len(parent1) != len(parent2):
        raise ValueError("parent1 and parent2 must have the same length")
    if num_jobs <= 0:
        raise ValueError("num_jobs must be positive")
    if not parent1:
        return []

    # Step 1: Choose a non-empty subset of jobs to inherit positions from parent1
    split_point = rng.randrange(1, num_jobs) if num_jobs > 1 else 1
    chosen_jobs = set(rng.sample(range(num_jobs), k=split_point))

    child: list[int | None] = [None] * len(parent1)

    # Step 2: Copy genes belonging to chosen_jobs from parent1 in-place
    for idx, gene in enumerate(parent1):
        if gene in chosen_jobs:
            child[idx] = gene

    # Step 3: Fill remaining slots using genes from parent2 in order
    remaining_iter = (gene for gene in parent2 if gene not in chosen_jobs)
    for idx in range(len(child)):
        if child[idx] is None:
            child[idx] = next(remaining_iter)

    return [gene for gene in child if gene is not None]
