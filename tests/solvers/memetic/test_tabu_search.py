from __future__ import annotations

from collections import Counter
import random

import pytest

from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance
from src.solvers.genetic import create_chromosome, decode_active_obr, evaluate_fitness
from src.solvers.memetic.tabu_search import tabu_search


def _make_tiny_instance() -> ProblemInstance:
    """Create a small 3x3 synthetic instance for fast test execution."""
    return ProblemInstance(
        name="tiny3x3",
        num_jobs=3,
        num_machines=3,
        jobs=[
            [Operation(0, 0, 0, 2), Operation(0, 1, 1, 3), Operation(0, 2, 2, 4)],
            [Operation(1, 0, 1, 5), Operation(1, 1, 2, 1), Operation(1, 2, 0, 2)],
            [Operation(2, 0, 2, 3), Operation(2, 1, 0, 4), Operation(2, 2, 1, 2)],
        ],
    )


class TestTabuSearchBehavior:
    def test_result_fitness_less_than_or_equal_to_starting_fitness(self) -> None:
        instance = _make_tiny_instance()

        for seed in range(5):
            rng = random.Random(seed)
            chrom = create_chromosome(instance, rng)
            start_fitness = evaluate_fitness(instance, chrom)

            best_chrom, best_fitness = tabu_search(
                instance,
                chrom,
                random.Random(seed + 100),
                iterations=10,
                neighbors_per_iter=6,
                tenure=4,
            )

            assert best_fitness <= start_fitness
            assert evaluate_fitness(instance, best_chrom) == best_fitness

    def test_result_is_valid_chromosome_and_feasible_schedule(self) -> None:
        instance = _make_tiny_instance()
        chrom = create_chromosome(instance, random.Random(42))
        expected_counts = {j: instance.num_machines for j in range(instance.num_jobs)}

        best_chrom, best_fitness = tabu_search(
            instance,
            chrom,
            random.Random(123),
            iterations=12,
            neighbors_per_iter=6,
            tenure=5,
        )

        assert len(best_chrom) == instance.total_operations
        assert Counter(best_chrom) == expected_counts

        makespan, machine_schedules, job_schedules = decode_active_obr(instance, best_chrom)
        assert makespan == best_fitness
        is_valid, errors = validate_schedule(instance, machine_schedules, job_schedules)
        assert is_valid is True
        assert errors == []

    def test_original_chromosome_not_modified(self) -> None:
        instance = _make_tiny_instance()
        chrom = create_chromosome(instance, random.Random(7))
        snapshot = chrom[:]

        tabu_search(instance, chrom, random.Random(99), iterations=8, neighbors_per_iter=5)

        assert chrom == snapshot

    def test_deterministic_for_same_seed(self) -> None:
        instance = _make_tiny_instance()
        chrom = create_chromosome(instance, random.Random(11))

        res1_chrom, res1_fit = tabu_search(
            instance, chrom, random.Random(55), iterations=10, neighbors_per_iter=5, tenure=3
        )
        res2_chrom, res2_fit = tabu_search(
            instance, chrom, random.Random(55), iterations=10, neighbors_per_iter=5, tenure=3
        )

        assert res1_chrom == res2_chrom
        assert res1_fit == res2_fit


class TestTabuSearchValidationAndEdgeCases:
    def test_empty_chromosome_returns_empty(self) -> None:
        instance = _make_tiny_instance()
        res_chrom, res_fit = tabu_search(instance, [], random.Random(0))
        assert res_chrom == []
        assert res_fit == 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"iterations": 0}, "iterations"),
            ({"iterations": -5}, "iterations"),
            ({"neighbors_per_iter": 0}, "neighbors_per_iter"),
            ({"neighbors_per_iter": -2}, "neighbors_per_iter"),
            ({"tenure": -1}, "tenure"),
            ({"tenure": -10}, "tenure"),
        ],
    )
    def test_invalid_parameters_raise_value_error(
        self, kwargs: dict[str, int], match: str
    ) -> None:
        instance = _make_tiny_instance()
        chrom = create_chromosome(instance, random.Random(0))

        with pytest.raises(ValueError, match=match):
            tabu_search(instance, chrom, random.Random(0), **kwargs)
