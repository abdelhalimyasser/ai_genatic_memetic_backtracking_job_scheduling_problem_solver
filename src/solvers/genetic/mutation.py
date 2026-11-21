from __future__ import annotations

import random


def swap_mutation(chromosome: list[int], mutation_rate: float, rng: random.Random) -> list[int]:
    """Apply swap mutation to an operation-based chromosome.

    Steps:
1. Validate that mutation_rate is within [0.0, 1.0].
2. Copy the chromosome and skip mutation if length is under 2 or rate does not trigger.
3. Select a random index and find all indices with a different gene value.
4. Swap the selected gene with a randomly chosen different gene.
5. Return the resulting chromosome.
    """
    # make sure the mutation rate is valid
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be between 0.0 and 1.0")
    # copy the chromosome so we don't lose the original
    mutated = chromosome[:] 
    # return the chromosome if its length is too short or mutation didn't trigger
    if len(mutated) < 2 or rng.random() >= mutation_rate: 
        return mutated
    # pick a random index from the chromosome
    i = rng.randrange(len(mutated)) 
    # find indices with a different value
    different_indices = [idx for idx, gene in enumerate(mutated) if gene != mutated[i]]
    if different_indices:
        j = rng.choice(different_indices) # pick one from the list of different indices to swap with
        mutated[i], mutated[j] = mutated[j], mutated[i] # swap them 

    return mutated
