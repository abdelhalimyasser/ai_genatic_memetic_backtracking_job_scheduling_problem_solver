from __future__ import annotations

import random

from src.models.problem_instance import ProblemInstance
from src.solvers.genetic.encoding import evaluate_fitness
from src.solvers.memetic.neighborhood import generate_neighbor


def tabu_search(
    instance: ProblemInstance,
    chromosome: list[int],
    rng: random.Random,
    iterations: int = 20,
    neighbors_per_iter: int = 10,
    tenure: int = 7,
) -> tuple[list[int], int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if neighbors_per_iter <= 0:
        raise ValueError("neighbors_per_iter must be positive")
    if tenure < 0:
        raise ValueError("tenure must be non-negative")

    if not chromosome:
        return [], 0

    current_chromosome = chromosome[:]
    current_fitness = evaluate_fitness(instance, current_chromosome)

    best_chromosome = current_chromosome[:]
    best_fitness = current_fitness

    tabu_list: list[tuple[int, int]] = []

    # Tabu search iterations
    for _ in range(iterations):
        allowed_candidates: list[tuple[int, list[int], tuple[int, int]]] = []

        # Sample candidate neighbors
        for _ in range(neighbors_per_iter):
            neighbor, move = generate_neighbor(current_chromosome, rng)
            if neighbor is None or move is None:
                continue

            fitness = evaluate_fitness(instance, neighbor)
            is_tabu = move in tabu_list

            # Aspiration criterion: allow if not tabu or strictly better than best_fitness
            if (not is_tabu) or (fitness < best_fitness):
                allowed_candidates.append((fitness, neighbor, move))

        # If all sampled neighbors are rejected, skip this iteration
        if not allowed_candidates:
            continue

        # Pick the best allowed neighbor (lowest makespan)
        best_candidate = min(allowed_candidates, key=lambda item: item[0])
        cand_fitness, cand_chromosome, cand_move = best_candidate

        # Move to the chosen neighbor (even if worse than current)
        current_chromosome = cand_chromosome
        current_fitness = cand_fitness

        # Update tabu list with the new move
        if tenure > 0:
            tabu_list.append(cand_move)
            if len(tabu_list) > tenure:
                tabu_list.pop(0)

        # Update global best solution found so far
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_chromosome = current_chromosome[:]

    return best_chromosome, best_fitness
