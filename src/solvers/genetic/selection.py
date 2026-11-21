from __future__ import annotations

import random


def tournament_selection(
    population: list[list[int]],
    fitnesses: list[int],
    rng: random.Random,
    tournament_size: int = 3,
) -> list[int]:
    """Select one parent via tournament selection,
    select random ex:3, then select the lower one then it will be the first parent,
    repeat for the second parent, .... 
    """
    if not population:
        raise ValueError("population cannot be empty")
    if len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have the same length")

    k = min(max(1, tournament_size), len(population))
    contenders = rng.sample(range(len(population)), k=k)
    winner_idx = min(contenders, key=lambda idx: fitnesses[idx])

    return population[winner_idx][:]
