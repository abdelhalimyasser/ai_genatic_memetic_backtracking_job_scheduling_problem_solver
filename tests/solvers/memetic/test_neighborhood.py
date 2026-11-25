from __future__ import annotations

from collections import Counter
import random

import pytest

from src.solvers.memetic.neighborhood import (
    generate_neighbor,
    insert_neighbor,
    swap_neighbor,
)


class TestSwapNeighbor:
    def test_preserves_length_and_gene_counts(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        expected_counts = Counter(chromosome)

        for seed in range(20):
            rng = random.Random(seed)
            neighbor, move = swap_neighbor(chromosome, rng)

            assert neighbor is not None
            assert move is not None
            assert len(neighbor) == len(chromosome)
            assert Counter(neighbor) == expected_counts

    def test_original_chromosome_not_modified(self) -> None:
        chromosome = [0, 1, 0, 1]
        snapshot = chromosome[:]
        rng = random.Random(42)

        neighbor, move = swap_neighbor(chromosome, rng)

        assert chromosome == snapshot
        assert neighbor is not chromosome

    def test_move_format_and_correct_swap(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        rng = random.Random(7)

        neighbor, move = swap_neighbor(chromosome, rng)

        assert neighbor is not None
        assert move is not None
        i, j = move
        assert i < j
        assert chromosome[i] != chromosome[j]
        assert neighbor[i] == chromosome[j]
        assert neighbor[j] == chromosome[i]

    def test_edge_cases_empty_single_and_identical(self) -> None:
        rng = random.Random(0)
        assert swap_neighbor([], rng) == (None, None)
        assert swap_neighbor([0], rng) == (None, None)
        assert swap_neighbor([1, 1, 1], rng) == (None, None)

    def test_deterministic_with_same_seed(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        res1 = swap_neighbor(chromosome, random.Random(99))
        res2 = swap_neighbor(chromosome, random.Random(99))
        assert res1 == res2


class TestInsertNeighbor:
    def test_preserves_length_and_gene_counts(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        expected_counts = Counter(chromosome)

        for seed in range(20):
            rng = random.Random(seed)
            neighbor, move = insert_neighbor(chromosome, rng)

            assert neighbor is not None
            assert move is not None
            assert len(neighbor) == len(chromosome)
            assert Counter(neighbor) == expected_counts

    def test_original_chromosome_not_modified(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        snapshot = chromosome[:]
        rng = random.Random(42)

        neighbor, move = insert_neighbor(chromosome, rng)

        assert chromosome == snapshot
        assert neighbor is not chromosome

    def test_move_format_and_bounds(self) -> None:
        chromosome = [0, 1, 2, 3]
        rng = random.Random(15)

        neighbor, move = insert_neighbor(chromosome, rng)

        assert neighbor is not None
        assert move is not None
        i, j = move
        assert 0 <= i < len(chromosome)
        assert 0 <= j < len(chromosome)
        assert i != j

    def test_edge_cases_empty_and_single(self) -> None:
        rng = random.Random(0)
        assert insert_neighbor([], rng) == (None, None)
        assert insert_neighbor([0], rng) == (None, None)

    def test_deterministic_with_same_seed(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        res1 = insert_neighbor(chromosome, random.Random(77))
        res2 = insert_neighbor(chromosome, random.Random(77))
        assert res1 == res2


class TestGenerateNeighbor:
    def test_preserves_length_and_gene_counts(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        expected_counts = Counter(chromosome)

        for seed in range(30):
            rng = random.Random(seed)
            neighbor, move = generate_neighbor(chromosome, rng)

            assert neighbor is not None
            assert move is not None
            assert len(neighbor) == len(chromosome)
            assert Counter(neighbor) == expected_counts

    def test_original_chromosome_never_modified(self) -> None:
        chromosome = [0, 1, 2, 0, 1, 2]
        snapshot = chromosome[:]

        for seed in range(15):
            rng = random.Random(seed)
            generate_neighbor(chromosome, rng)
            assert chromosome == snapshot
