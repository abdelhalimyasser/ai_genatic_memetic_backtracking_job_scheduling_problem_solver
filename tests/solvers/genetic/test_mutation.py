from __future__ import annotations

from collections import Counter
import random

import pytest

from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance
from src.solvers.genetic.encoding import decode_active_obr
from src.solvers.genetic.mutation import swap_mutation


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


class TestSwapMutationBehavior:
    @pytest.mark.parametrize(
        ("num_jobs", "num_machines"),
        [(2, 2), (3, 3), (6, 6)],
    )
    def test_preserves_length_and_job_counts_across_seeds(
        self, num_jobs: int, num_machines: int
    ) -> None:
        base = [j for j in range(num_jobs) for _ in range(num_machines)]
        expected_counts = Counter(base)

        for seed in range(25):
            mutated = swap_mutation(base, mutation_rate=1.0, rng=random.Random(seed))
            assert len(mutated) == len(base)
            assert Counter(mutated) == expected_counts

    def test_mutated_chromosome_decodes_to_valid_schedule(self) -> None:
        inst = _make_3x3_instance()
        chrom = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        for seed in range(10):
            mutated = swap_mutation(chrom, mutation_rate=1.0, rng=random.Random(seed))
            _, ms, js = decode_active_obr(inst, mutated)
            assert validate_schedule(inst, ms, js) == (True, [])

    def test_does_not_mutate_input_in_place(self) -> None:
        original = [0, 1, 2, 0, 1, 2]
        snapshot = original[:]

        mutated_high = swap_mutation(original, mutation_rate=1.0, rng=random.Random(42))
        assert original == snapshot
        assert mutated_high is not original

        mutated_zero = swap_mutation(original, mutation_rate=0.0, rng=random.Random(42))
        assert original == snapshot
        assert mutated_zero is not original

    def test_mutation_rate_zero_never_changes_chromosome(self) -> None:
        chromosome = [0, 1, 2, 3, 0, 1, 2, 3]
        for seed in range(30):
            mutated = swap_mutation(chromosome, mutation_rate=0.0, rng=random.Random(seed))
            assert mutated == chromosome

    def test_mutation_rate_one_swaps_exactly_two_distinct_genes(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        for seed in range(25):
            mutated = swap_mutation(chromosome, mutation_rate=1.0, rng=random.Random(seed))
            diffs = [
                (idx, orig, new)
                for idx, (orig, new) in enumerate(zip(chromosome, mutated))
                if orig != new
            ]
            assert len(diffs) == 2
            assert diffs[0][1] == diffs[1][2]
            assert diffs[0][2] == diffs[1][1]
            assert diffs[0][1] != diffs[0][2]

    def test_intermediate_mutation_rate_mixes_mutated_and_unchanged(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        rng = random.Random(123)
        results = [swap_mutation(chromosome, mutation_rate=0.5, rng=rng) for _ in range(40)]
        assert any(res == chromosome for res in results)
        assert any(res != chromosome for res in results)

    def test_deterministic_for_same_seed(self) -> None:
        chromosome = [0, 1, 2, 3, 0, 1, 2, 3]
        first = swap_mutation(chromosome, mutation_rate=0.75, rng=random.Random(99))
        second = swap_mutation(chromosome, mutation_rate=0.75, rng=random.Random(99))
        assert first == second


class TestSwapMutationEdgeCasesAndValidation:
    def test_edge_cases_empty_single_two_and_all_identical(self) -> None:
        assert swap_mutation([], mutation_rate=1.0, rng=random.Random(0)) == []
        assert swap_mutation([0], mutation_rate=1.0, rng=random.Random(0)) == [0]

        identical = [0, 0, 0]
        res_identical = swap_mutation(identical, mutation_rate=1.0, rng=random.Random(0))
        assert res_identical == [0, 0, 0]
        assert res_identical is not identical

        assert swap_mutation([0, 1], mutation_rate=1.0, rng=random.Random(0)) == [1, 0]

    @pytest.mark.parametrize(
        "bad_rate",
        [-0.1, -1.0, 1.1, 2.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_invalid_mutation_rate_raises(self, bad_rate: float) -> None:
        with pytest.raises(ValueError, match="mutation_rate"):
            swap_mutation([0, 1], mutation_rate=bad_rate, rng=random.Random(0))
