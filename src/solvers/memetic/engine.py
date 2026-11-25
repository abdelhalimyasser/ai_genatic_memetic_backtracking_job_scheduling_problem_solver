from __future__ import annotations

import random
import time

from src.core.validator import validate_schedule
from src.models.problem_instance import ProblemInstance
from src.models.result import ScheduleResult
from src.solvers.genetic.crossover import job_based_crossover
from src.solvers.genetic.encoding import (
    create_chromosome,
    decode_active_obr,
    evaluate_fitness,
)
from src.solvers.genetic.mutation import swap_mutation
from src.solvers.genetic.selection import tournament_selection
from src.solvers.memetic.tabu_search import tabu_search


def run_memetic_algorithm(
    instance: ProblemInstance,
    population_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    elitism_count: int = 1,
    local_search_rate: float = 0.3,
    tabu_iterations: int = 20,
    neighbors_per_iter: int = 10,
    tabu_tenure: int = 7,
    seed: int | None = None,
) -> ScheduleResult:
    # Step 1: Validate parameters
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generations < 0:
        raise ValueError("generations must be non-negative")
    if not (0.0 <= crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be between 0.0 and 1.0")
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be between 0.0 and 1.0")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    if not (0 <= elitism_count <= population_size):
        raise ValueError("elitism_count must be between 0 and population_size")
    if not (0.0 <= local_search_rate <= 1.0):
        raise ValueError("local_search_rate must be between 0.0 and 1.0")
    if tabu_iterations <= 0:
        raise ValueError("tabu_iterations must be positive")
    if neighbors_per_iter <= 0:
        raise ValueError("neighbors_per_iter must be positive")
    if tabu_tenure < 0:
        raise ValueError("tabu_tenure must be non-negative")

    start_time = time.perf_counter()
    rng = random.Random(seed)

    # Step 2: Initialize population and evaluate initial fitness
    population = [create_chromosome(instance, rng) for _ in range(population_size)]
    fitnesses = [evaluate_fitness(instance, chrom) for chrom in population]

    best_idx = min(range(population_size), key=lambda idx: fitnesses[idx])
    best_chromosome = population[best_idx][:]
    best_fitness = fitnesses[best_idx]

    # Step 3: Evolution loop across generations
    for _ in range(generations):
        next_population: list[list[int]] = []
        next_fitnesses: list[int] = []

        # Elitism: carry over the best individuals directly
        if elitism_count > 0:
            elite_indices = sorted(
                range(population_size), key=lambda idx: fitnesses[idx]
            )[:elitism_count]
            for idx in elite_indices:
                next_population.append(population[idx][:])
                next_fitnesses.append(fitnesses[idx])

        while len(next_population) < population_size:
            parent1 = tournament_selection(
                population, fitnesses, rng, tournament_size=tournament_size
            )
            parent2 = tournament_selection(
                population, fitnesses, rng, tournament_size=tournament_size
            )

            # Crossover
            if rng.random() < crossover_rate:
                child = job_based_crossover(parent1, parent2, instance.num_jobs, rng)
            else:
                child = parent1[:]

            # Mutation
            child = swap_mutation(child, mutation_rate, rng)

            # Local Search (Tabu Search): apply with probability local_search_rate
            if rng.random() < local_search_rate:
                child, child_fitness = tabu_search(
                    instance=instance,
                    chromosome=child,
                    rng=rng,
                    iterations=tabu_iterations,
                    neighbors_per_iter=neighbors_per_iter,
                    tenure=tabu_tenure,
                )
            else:
                child_fitness = evaluate_fitness(instance, child)

            next_population.append(child)
            next_fitnesses.append(child_fitness)

        population = next_population
        fitnesses = next_fitnesses

        # Track the best individual found so far
        gen_best_idx = min(range(population_size), key=lambda idx: fitnesses[idx])
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx][:]

    # Step 4: Final schedule decoding and validation
    makespan, machine_schedules, job_schedules = decode_active_obr(
        instance, best_chromosome
    )
    is_feasible, _ = validate_schedule(instance, machine_schedules, job_schedules)

    return ScheduleResult(
        instance_name=instance.name,
        algorithm_name="Memetic Algorithm",
        makespan=makespan,
        is_feasible=is_feasible,
        runtime_sec=time.perf_counter() - start_time,
        machine_schedules=machine_schedules,
        job_schedules=job_schedules,
    )
