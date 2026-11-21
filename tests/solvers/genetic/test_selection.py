from __future__ import annotations

from collections import Counter
import random

import pytest

from src.solvers.genetic.selection import tournament_selection


class TestTournamentSelectionBehavior:
    def test_returns_copy_without_mutating_population(self) -> None:
        population = [[0, 1], [1, 0], [0, 0]]
        fitnesses = [15, 10, 20]
        snapshot = [chrom[:] for chrom in population]

        selected = tournament_selection(
            population, fitnesses, random.Random(42), tournament_size=3
        )
        assert selected == [1, 0]
        assert selected is not population[1]

        selected[0] = 999
        assert population == snapshot

    def test_deterministic_for_same_seed(self) -> None:
        population = [[i, i + 1] for i in range(6)]
        fitnesses = [30, 12, 25, 8, 19, 40]

        first = tournament_selection(
            population, fitnesses, random.Random(77), tournament_size=3
        )
        second = tournament_selection(
            population, fitnesses, random.Random(77), tournament_size=3
        )
        assert first == second

    def test_prefers_lower_makespan_with_distinct_fitnesses(self) -> None:
        population = [[0], [1], [2], [3], [4]]
        fitnesses = [10, 20, 30, 40, 50]
        rng = random.Random(123)

        counts = Counter(
            tournament_selection(population, fitnesses, rng, tournament_size=3)[0]
            for _ in range(200)
        )
        assert counts[0] > counts[2]
        assert counts[1] > counts[3]
        assert counts[4] == 0

    def test_tied_fitnesses_handled_validly(self) -> None:
        population = [[0], [1], [2], [3]]
        all_tied = [10, 10, 10, 10]
        rng = random.Random(5)
        picks = {
            tournament_selection(population, all_tied, rng, tournament_size=2)[0]
            for _ in range(30)
        }
        assert picks.issubset({0, 1, 2, 3})
        assert len(picks) > 1

        partial_ties = [5, 20, 5, 30]
        for seed in range(15):
            chosen = tournament_selection(
                population, partial_ties, random.Random(seed), tournament_size=4
            )
            assert chosen in ([0], [2])

    def test_extreme_fitness_values(self) -> None:
        population = [[0], [1], [2], [3]]
        fitnesses = [0, 1, 10**6, 10**9]
        rng = random.Random(99)

        counts = Counter(
            tournament_selection(population, fitnesses, rng, tournament_size=2)[0]
            for _ in range(100)
        )
        assert counts[0] > counts[2]
        assert counts[3] == 0


class TestTournamentSelectionEdgeCasesAndValidation:
    def test_tournament_size_one_can_select_any_individual(self) -> None:
        population = [[0], [1], [2], [3]]
        fitnesses = [10, 20, 30, 40]
        rng = random.Random(11)

        picks = {
            tournament_selection(population, fitnesses, rng, tournament_size=1)[0]
            for _ in range(60)
        }
        assert picks == {0, 1, 2, 3}

    def test_tournament_size_equal_or_greater_than_population_picks_global_best(
        self,
    ) -> None:
        population = [[0], [1], [2], [3]]
        fitnesses = [25, 12, 40, 18]

        for size in (len(population), len(population) + 5):
            for seed in range(10):
                chosen = tournament_selection(
                    population, fitnesses, random.Random(seed), tournament_size=size
                )
                assert chosen == [1]

    @pytest.mark.parametrize("bad_size", [0, -1, -5])
    def test_non_positive_tournament_size_clamps_to_one(self, bad_size: int) -> None:
        population = [[0, 1], [1, 0]]
        fitnesses = [10, 20]
        chosen = tournament_selection(
            population, fitnesses, random.Random(0), tournament_size=bad_size
        )
        assert chosen in population

    def test_single_element_population(self) -> None:
        population = [[0, 1, 0, 1]]
        chosen = tournament_selection(
            population, [12], random.Random(0), tournament_size=3
        )
        assert chosen == [0, 1, 0, 1]
        assert chosen is not population[0]

    @pytest.mark.parametrize(
        ("population", "fitnesses", "match"),
        [
            ([], [], "empty"),
            ([[0], [1]], [10], "same length"),
            ([[0]], [10, 20], "same length"),
        ],
    )
    def test_invalid_inputs_raise_value_error(
        self,
        population: list[list[int]],
        fitnesses: list[int],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            tournament_selection(population, fitnesses, random.Random(0))
