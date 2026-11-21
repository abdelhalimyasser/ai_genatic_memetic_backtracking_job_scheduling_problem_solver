from __future__ import annotations

import random


def swap_mutation(
    chromosome: list[int],
    mutation_rate: float,
    rng: random.Random,
) -> list[int]:
    """Apply swap mutation to an OBR chromosome while preserving job counts."""
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be between 0.0 and 1.0")

    mutated = chromosome[:]
    if len(mutated) < 2 or rng.random() >= mutation_rate:
        return mutated

    i = rng.randrange(len(mutated))
    different_indices = [idx for idx, gene in enumerate(mutated) if gene != mutated[i]]
    if different_indices:
        j = rng.choice(different_indices)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated
