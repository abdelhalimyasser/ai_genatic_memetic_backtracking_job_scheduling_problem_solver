from __future__ import annotations

import random
import time

from src.core.validator import validate_schedule
from src.models.problem_instance import ProblemInstance
from src.models.result import ScheduleResult
from src.solvers.genetic.crossover import job_based_crossover
from src.solvers.genetic.encoding import create_chromosome, decode_active_obr, evaluate_fitness
from src.solvers.genetic.mutation import swap_mutation
from src.solvers.genetic.selection import tournament_selection


def run_genetic_algorithm(
    instance: ProblemInstance,
    population_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    elitism_count: int = 1,
    seed: int | None = None,
) -> ScheduleResult:
    """Run a standard Genetic Algorithm to minimize makespan for a JSP instance."""
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

    start_time = time.perf_counter()
    rng = random.Random(seed)

    population = [create_chromosome(instance, rng) for _ in range(population_size)]
    fitnesses = [evaluate_fitness(instance, chrom) for chrom in population]

    best_idx = min(range(population_size), key=lambda idx: fitnesses[idx])
    best_chromosome = population[best_idx][:]
    best_fitness = fitnesses[best_idx]

    for _ in range(generations):
        next_population: list[list[int]] = []
        if elitism_count > 0:
            elite_indices = sorted(range(population_size), key=lambda idx: fitnesses[idx])[
                :elitism_count
            ]
            next_population.extend(population[idx][:] for idx in elite_indices)

        while len(next_population) < population_size:
            parent1 = tournament_selection(
                population, fitnesses, rng, tournament_size=tournament_size
            )
            parent2 = tournament_selection(
                population, fitnesses, rng, tournament_size=tournament_size
            )

            if rng.random() < crossover_rate:
                child = job_based_crossover(parent1, parent2, instance.num_jobs, rng)
            else:
                child = parent1[:]

            child = swap_mutation(child, mutation_rate, rng)
            next_population.append(child)

        population = next_population
        fitnesses = [evaluate_fitness(instance, chrom) for chrom in population]

        gen_best_idx = min(range(population_size), key=lambda idx: fitnesses[idx])
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx][:]

    makespan, machine_schedules, job_schedules = decode_active_obr(instance, best_chromosome)
    is_feasible, _ = validate_schedule(instance, machine_schedules, job_schedules)

    return ScheduleResult(
        instance_name=instance.name,
        algorithm_name="Genetic Algorithm",
        makespan=makespan,
        is_feasible=is_feasible,
        runtime_sec=time.perf_counter() - start_time,
        machine_schedules=machine_schedules,
        job_schedules=job_schedules,
    )
