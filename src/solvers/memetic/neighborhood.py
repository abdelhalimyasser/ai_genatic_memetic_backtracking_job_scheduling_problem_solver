from __future__ import annotations

import random


def swap_neighbor(
        chromosome: list[int], rng: random.Random
) -> tuple[list[int], tuple[int, int]] | tuple[None, None]:
    if len(chromosome) < 2:
        return None, None

    i = rng.randrange(len(chromosome))

    different_indices = [
        idx for idx, gene in enumerate(chromosome) if gene != chromosome[i]
    ]
    if not different_indices:
        return None, None

    j = rng.choice(different_indices)

    neighbor = chromosome[:]
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

    move = (min(i, j), max(i, j))
    return neighbor, move


def insert_neighbor(
        chromosome: list[int], rng: random.Random
) -> tuple[list[int], tuple[int, int]] | tuple[None, None]:
    if len(chromosome) < 2:
        return None, None

    i = rng.randrange(len(chromosome))

    other_indices = [idx for idx in range(len(chromosome)) if idx != i]
    j = rng.choice(other_indices)

    neighbor = chromosome[:]
    gene = neighbor.pop(i)
    neighbor.insert(j, gene)

    move = (i, j)
    return neighbor, move


def generate_neighbor(
        chromosome: list[int], rng: random.Random
) -> tuple[list[int], tuple[int, int]] | tuple[None, None]:
    if rng.random() < 0.5:
        return swap_neighbor(chromosome, rng)
    return insert_neighbor(chromosome, rng)
