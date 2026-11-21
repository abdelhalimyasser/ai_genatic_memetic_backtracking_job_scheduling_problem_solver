from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import pytest

from src.core.parser import load_instance
from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance
from src.solvers.genetic import (
    create_chromosome,
    decode_active_obr,
    evaluate_fitness,
    job_based_crossover,
    run_genetic_algorithm,
    swap_mutation,
    tournament_selection,
)

DATASETS_DIR = Path(__file__).resolve().parents[3] / "dataset"
JOB1_PATH = DATASETS_DIR / "jobshop1.txt"
JOB2_PATH = DATASETS_DIR / "jobshop2.txt"


def _make_3x3_instance() -> ProblemInstance:
    return ProblemInstance(
        name="mini3x3",
        num_jobs=3,
        num_machines=3,
        jobs=[
            [Operation(0, 0, 0, 2), Operation(0, 1, 1, 3), Operation(0, 2, 2, 4)],
            [Operation(1, 0, 1, 5), Operation(1, 1, 2, 1), Operation(1, 2, 0, 2)],
            [Operation(2, 0, 2, 3), Operation(2, 1, 0, 4), Operation(2, 2, 1, 2)],
        ],
    )


def _assert_population_valid(
    instance: ProblemInstance,
    population: list[list[int]],
    fitnesses: list[int],
) -> None:
    expected_counts = {j: instance.num_machines for j in range(instance.num_jobs)}
    for chrom, fitness in zip(population, fitnesses):
        assert len(chrom) == instance.total_operations
        assert Counter(chrom) == expected_counts
        decoded_makespan, ms, js = decode_active_obr(instance, chrom)
        assert fitness == decoded_makespan
        assert validate_schedule(instance, ms, js) == (True, [])


class TestRealOperatorPipelineIntegration:
    @pytest.mark.parametrize("use_ft06", [False, True], ids=["mini3x3", "ft06"])
    def test_chromosome_validity_and_monotonic_best_across_generations(
        self, use_ft06: bool
    ) -> None:
        instance = (
            load_instance(JOB1_PATH, "ft06", format_type="job1")
            if use_ft06
            else _make_3x3_instance()
        )
        pop_size = 10
        generations = 8
        elitism_count = 1
        crossover_rate = 0.8
        mutation_rate = 0.2
        tournament_size = 3
        rng = random.Random(42)

        population = [create_chromosome(instance, rng) for _ in range(pop_size)]
        fitnesses = [evaluate_fitness(instance, chrom) for chrom in population]
        _assert_population_valid(instance, population, fitnesses)

        best_idx = min(range(pop_size), key=lambda idx: fitnesses[idx])
        best_chrom = population[best_idx][:]
        best_fitness = fitnesses[best_idx]

        for _ in range(generations):
            elite_indices = sorted(range(pop_size), key=lambda idx: fitnesses[idx])[
                :elitism_count
            ]
            next_population = [population[idx][:] for idx in elite_indices]

            while len(next_population) < pop_size:
                p1 = tournament_selection(
                    population, fitnesses, rng, tournament_size=tournament_size
                )
                p2 = tournament_selection(
                    population, fitnesses, rng, tournament_size=tournament_size
                )
                child = (
                    job_based_crossover(p1, p2, instance.num_jobs, rng)
                    if rng.random() < crossover_rate
                    else p1[:]
                )
                child = swap_mutation(child, mutation_rate, rng)
                next_population.append(child)

            population = next_population
            fitnesses = [evaluate_fitness(instance, chrom) for chrom in population]
            _assert_population_valid(instance, population, fitnesses)

            gen_best_idx = min(range(pop_size), key=lambda idx: fitnesses[idx])
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chrom = population[gen_best_idx][:]

            assert best_fitness <= min(fitnesses) or best_fitness < min(fitnesses)

        final_makespan, ms, js = decode_active_obr(instance, best_chrom)
        assert final_makespan == best_fitness
        assert validate_schedule(instance, ms, js) == (True, [])

    def test_engine_matches_unmocked_operator_replay(self) -> None:
        instance = _make_3x3_instance()
        pop_size = 6
        generations = 5
        crossover_rate = 0.8
        mutation_rate = 0.1
        tournament_size = 3
        elitism_count = 1
        seed = 99

        engine_res = run_genetic_algorithm(
            instance,
            population_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            elitism_count=elitism_count,
            seed=seed,
        )

        rng = random.Random(seed)
        population = [create_chromosome(instance, rng) for _ in range(pop_size)]
        fitnesses = [evaluate_fitness(instance, c) for c in population]
        best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        best_chrom = population[best_idx][:]
        best_fit = fitnesses[best_idx]

        for _ in range(generations):
            elites = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elitism_count]
            next_pop = [population[i][:] for i in elites]
            while len(next_pop) < pop_size:
                p1 = tournament_selection(population, fitnesses, rng, tournament_size=tournament_size)
                p2 = tournament_selection(population, fitnesses, rng, tournament_size=tournament_size)
                child = (
                    job_based_crossover(p1, p2, instance.num_jobs, rng)
                    if rng.random() < crossover_rate
                    else p1[:]
                )
                child = swap_mutation(child, mutation_rate, rng)
                next_pop.append(child)
            population = next_pop
            fitnesses = [evaluate_fitness(instance, c) for c in population]
            gen_best = min(range(pop_size), key=lambda i: fitnesses[i])
            if fitnesses[gen_best] < best_fit:
                best_fit = fitnesses[gen_best]
                best_chrom = population[gen_best][:]

        replay_makespan, replay_ms, replay_js = decode_active_obr(instance, best_chrom)
        assert engine_res.makespan == replay_makespan
        assert engine_res.machine_schedules == replay_ms
        assert engine_res.job_schedules == replay_js

    def test_active_left_shift_finds_optimal_makespan_on_gap_instance(self) -> None:
        inst = ProblemInstance(
            name="gap2x2",
            num_jobs=2,
            num_machines=2,
            jobs=[
                [Operation(0, 0, 0, 5), Operation(0, 1, 1, 3)],
                [Operation(1, 0, 1, 4), Operation(1, 1, 0, 2)],
            ],
        )
        res = run_genetic_algorithm(inst, population_size=8, generations=10, seed=1)
        assert res.is_feasible is True
        assert res.makespan == 8
        assert validate_schedule(inst, res.machine_schedules, res.job_schedules) == (True, [])


class TestRealDatasetEndToEndIntegration:
    def test_ft06_evolves_and_schedules_all_operations(self) -> None:
        ft06 = load_instance(JOB1_PATH, "ft06", format_type="job1")
        initial = run_genetic_algorithm(ft06, population_size=20, generations=0, seed=42)
        evolved = run_genetic_algorithm(ft06, population_size=20, generations=40, seed=42)

        assert initial.is_feasible is True
        assert evolved.is_feasible is True
        assert validate_schedule(ft06, initial.machine_schedules, initial.job_schedules) == (True, [])
        assert validate_schedule(ft06, evolved.machine_schedules, evolved.job_schedules) == (True, [])
        assert evolved.makespan >= 55
        assert evolved.makespan <= initial.makespan
        assert sum(len(ops) for ops in evolved.machine_schedules.values()) == 36
        assert sum(len(ops) for ops in evolved.job_schedules.values()) == 36

    def test_la01_rectangular_benchmark_feasible(self) -> None:
        la01 = load_instance(JOB1_PATH, "la01", format_type="job1")
        res = run_genetic_algorithm(la01, population_size=12, generations=8, seed=13)
        assert res.is_feasible is True
        assert res.makespan >= 666
        assert validate_schedule(la01, res.machine_schedules, res.job_schedules) == (True, [])

    def test_orb07_zero_duration_benchmark_feasible(self) -> None:
        orb07 = load_instance(JOB1_PATH, "orb07", format_type="job1")
        res = run_genetic_algorithm(orb07, population_size=12, generations=8, seed=7)
        assert res.is_feasible is True
        assert validate_schedule(orb07, res.machine_schedules, res.job_schedules) == (True, [])

    def test_ta01_jobshop2_benchmark_feasible(self) -> None:
        ta01 = load_instance(JOB2_PATH, "ta01", format_type="job2")
        res = run_genetic_algorithm(ta01, population_size=6, generations=3, seed=5)
        assert res.is_feasible is True
        assert validate_schedule(ta01, res.machine_schedules, res.job_schedules) == (True, [])
