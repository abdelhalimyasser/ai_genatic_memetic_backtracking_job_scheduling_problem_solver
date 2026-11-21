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


def run_genetic_algorithm( instance: ProblemInstance, population_size: int = 50, generations: int = 100,
                           crossover_rate: float = 0.8, mutation_rate: float = 0.1, tournament_size: int = 3,
                           elitism_count: int = 1, seed: int | None = None) -> ScheduleResult:
    """Run the Genetic Algorithm to solve a Job Shop Scheduling instance.

    Steps:
1. Validate all algorithm hyperparameters (population size, generations, rates, elitism).
2. Initialize a random population of chromosomes and evaluate their fitness.
3. Track the best chromosome and lowest makespan across generations.
4. Evolve each generation using elitism, tournament selection, job-based crossover, and swap mutation.
5. Decode and validate the best chromosome, then return a complete ScheduleResult.
    """
    # make sure all parameters are valid
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

    start_time = time.perf_counter() # start a timer (stopwatch)
    rng = random.Random(seed) # set up the random generator
  
    population = [create_chromosome(instance, rng) for _ in range(population_size)] # create the initial population
    fitnesses = [evaluate_fitness(instance, chrom) for chrom in population] # evaluate fitness for each chromosome
    # find and save the best solution so far
    best_idx = min(range(population_size), key=lambda idx: fitnesses[idx])
    best_chromosome = population[best_idx][:]
    best_fitness = fitnesses[best_idx]

    for _ in range(generations):
        next_population: list[list[int]] = []
        if elitism_count > 0: # keep the best chromosomes for the next generation
            elite_indices = sorted(range(population_size), key=lambda idx: fitnesses[idx])[:elitism_count]
            next_population.extend(population[idx][:] for idx in elite_indices)
        # fill the rest of the new population
        while len(next_population) < population_size: 
            parent1 = tournament_selection(population, fitnesses, rng, tournament_size=tournament_size)
            
            parent2 = tournament_selection(population, fitnesses, rng, tournament_size=tournament_size)
            # apply crossover or just copy parent1
            if rng.random() < crossover_rate:
                child = job_based_crossover(parent1, parent2, instance.num_jobs, rng)
            else:
                child = parent1[:]

            child = swap_mutation(child, mutation_rate, rng) # mutate the child
            next_population.append(child)

        population = next_population
        fitnesses = [evaluate_fitness(instance, chrom) for chrom in population] # recalculate fitness for the new population

        gen_best_idx = min(range(population_size), key=lambda idx: fitnesses[idx])# get the best in this generation
        # update the overall best if we found a better one
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx][:]

    makespan, machine_schedules, job_schedules = decode_active_obr(instance, best_chromosome) # decode the best chromosome into a schedule
    is_feasible, _ = validate_schedule(instance, machine_schedules, job_schedules) # check if the final schedule is valid

    return ScheduleResult(
        instance_name=instance.name,
        algorithm_name="Genetic Algorithm",
        makespan=makespan,
        is_feasible=is_feasible,
        runtime_sec=time.perf_counter() - start_time,
        machine_schedules=machine_schedules,
        job_schedules=job_schedules,
    )
