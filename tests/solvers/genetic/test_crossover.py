from __future__ import annotations

from collections import Counter
import random

import pytest

from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance
from src.solvers.genetic.crossover import job_based_crossover
from src.solvers.genetic.encoding import create_chromosome, decode_active_obr


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


class TestJobBasedCrossoverCorrectness:
    @pytest.mark.parametrize(
        ("num_jobs", "num_machines"),
        [(2, 2), (3, 3), (2, 3), (3, 2), (6, 6)],
    )
    def test_preserves_length_and_job_counts(
        self, num_jobs: int, num_machines: int
    ) -> None:
        base = [j for j in range(num_jobs) for _ in range(num_machines)]
        expected = Counter(base)

        for seed in range(10):
            rng = random.Random(seed)
            p1 = base[:]
            p2 = base[:]
            rng.shuffle(p1)
            rng.shuffle(p2)

            child = job_based_crossover(p1, p2, num_jobs=num_jobs, rng=random.Random(seed + 100))
            assert len(child) == len(p1)
            assert Counter(child) == expected

    def test_jbx_preserves_chosen_job_positions_and_remaining_order(self) -> None:
        parent1 = [0, 1, 2, 0, 2, 1, 1, 0, 2]
        parent2 = [2, 1, 0, 1, 2, 0, 0, 2, 1]
        num_jobs = 3

        for seed in range(15):
            mirror_rng = random.Random(seed)
            split_point = mirror_rng.randrange(1, num_jobs)
            chosen_jobs = set(mirror_rng.sample(range(num_jobs), k=split_point))

            child = job_based_crossover(parent1, parent2, num_jobs, random.Random(seed))

            for idx, gene in enumerate(parent1):
                if gene in chosen_jobs:
                    assert child[idx] == gene

            expected_remaining = [g for g in parent2 if g not in chosen_jobs]
            actual_remaining = [g for g in child if g not in chosen_jobs]
            assert actual_remaining == expected_remaining

    def test_offspring_decodes_to_valid_schedule(self) -> None:
        inst = _make_3x3_instance()
        for seed in range(10):
            p1 = create_chromosome(inst, random.Random(seed))
            p2 = create_chromosome(inst, random.Random(seed + 50))
            child = job_based_crossover(p1, p2, inst.num_jobs, random.Random(seed + 100))
            _, ms, js = decode_active_obr(inst, child)
            assert validate_schedule(inst, ms, js) == (True, [])

    def test_does_not_modify_parents_and_is_deterministic(self) -> None:
        p1 = [0, 1, 2, 0, 1, 2]
        p2 = [2, 1, 0, 2, 1, 0]
        p1_snap = p1[:]
        p2_snap = p2[:]

        c1 = job_based_crossover(p1, p2, num_jobs=3, rng=random.Random(42))
        c2 = job_based_crossover(p1, p2, num_jobs=3, rng=random.Random(42))

        assert p1 == p1_snap
        assert p2 == p2_snap
        assert c1 == c2
        assert c1 is not p1
        assert c1 is not p2


class TestJobBasedCrossoverEdgeCasesAndValidation:
    def test_identical_parents_produce_identical_copy(self) -> None:
        parent = [0, 2, 1, 0, 2, 1]
        child = job_based_crossover(parent, parent[:], num_jobs=3, rng=random.Random(7))
        assert child == parent
        assert child is not parent

    def test_single_job_instance(self) -> None:
        p1 = [0, 0, 0]
        p2 = [0, 0, 0]
        child = job_based_crossover(p1, p2, num_jobs=1, rng=random.Random(0))
        assert child == [0, 0, 0]
        assert child is not p1
        assert child is not p2

    def test_empty_parents_return_empty_list(self) -> None:
        assert job_based_crossover([], [], num_jobs=2, rng=random.Random(0)) == []

    @pytest.mark.parametrize(
        ("p1", "p2", "num_jobs", "match"),
        [
            ([0, 1], [0, 1, 0], 2, "same length"),
            ([0, 1, 0], [0, 1], 2, "same length"),
            ([0, 1], [1, 0], 0, "positive"),
            ([0, 1], [1, 0], -2, "positive"),
        ],
    )
    def test_invalid_inputs_raise_value_error(
        self,
        p1: list[int],
        p2: list[int],
        num_jobs: int,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            job_based_crossover(p1, p2, num_jobs=num_jobs, rng=random.Random(0))
