from __future__ import annotations

from unittest.mock import patch

import pytest

import src.solvers.genetic as genetic_pkg
import src.solvers.genetic.engine as engine_mod
from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance, ScheduleResult
from src.solvers.genetic import run_genetic_algorithm


def _make_instance(
    name: str, jobs_spec: list[list[tuple[int, int]]]
) -> ProblemInstance:
    jobs = [
        [
            Operation(job_id=j_id, operation_id=op_id, machine_id=m_id, duration=dur)
            for op_id, (m_id, dur) in enumerate(job_ops)
        ]
        for j_id, job_ops in enumerate(jobs_spec)
    ]
    return ProblemInstance(
        name=name,
        num_jobs=len(jobs_spec),
        num_machines=len(jobs_spec[0]),
        jobs=jobs,
    )


def _make_3x3_instance() -> ProblemInstance:
    return _make_instance(
        "mini3x3",
        [
            [(0, 2), (1, 3), (2, 4)],
            [(1, 5), (2, 1), (0, 2)],
            [(2, 3), (0, 4), (1, 2)],
        ],
    )


class TestPublicExports:
    def test_expected_symbols_exported(self) -> None:
        assert genetic_pkg.__version__ == "1.0.0"
        expected = {
            "create_chromosome",
            "decode_active_obr",
            "evaluate_fitness",
            "tournament_selection",
            "job_based_crossover",
            "swap_mutation",
            "run_genetic_algorithm",
        }
        assert set(genetic_pkg.__all__) == expected
        for name in expected:
            assert hasattr(genetic_pkg, name)


class TestGeneticAlgorithmEngine:
    @pytest.mark.parametrize(
        "instance",
        [
            _make_instance("1x1", [[(0, 5)]]),
            _make_instance("1x3", [[(0, 3), (2, 4), (1, 2)]]),
            _make_instance("3x1", [[(0, 2)], [(0, 5)], [(0, 1)]]),
            _make_instance("2x3", [[(0, 2), (1, 3), (2, 1)], [(2, 4), (0, 2), (1, 5)]]),
            _make_instance("3x2", [[(0, 3), (1, 2)], [(1, 4), (0, 1)], [(0, 2), (1, 3)]]),
            _make_instance("2x2", [[(0, 3), (1, 4)], [(1, 2), (0, 5)]]),
            _make_3x3_instance(),
            _make_instance("zeros", [[(0, 0), (1, 2)], [(1, 0), (0, 0)]]),
        ],
        ids=["1x1", "1x3", "3x1", "2x3", "3x2", "2x2", "3x3", "zeros"],
    )
    def test_produces_feasible_validated_result_on_synthetic_instances(
        self, instance: ProblemInstance
    ) -> None:
        result = run_genetic_algorithm(
            instance,
            population_size=10,
            generations=12,
            seed=42,
        )
        assert isinstance(result, ScheduleResult)
        assert result.instance_name == instance.name
        assert result.algorithm_name == "Genetic Algorithm"
        assert result.is_feasible is True
        assert result.runtime_sec >= 0.0
        assert validate_schedule(
            instance, result.machine_schedules, result.job_schedules
        ) == (True, [])
        expected_makespan = max(
            (s.end_time for ops in result.machine_schedules.values() for s in ops),
            default=0,
        )
        assert result.makespan == expected_makespan

    def test_coordinates_all_components_with_expected_call_counts(self) -> None:
        inst = _make_instance("2x2", [[(0, 3), (1, 4)], [(1, 2), (0, 5)]])
        pop_size = 6
        gens = 3
        elitism = 2
        offspring_per_gen = pop_size - elitism

        with (
            patch.object(
                engine_mod, "create_chromosome", wraps=engine_mod.create_chromosome
            ) as mock_create,
            patch.object(
                engine_mod, "evaluate_fitness", wraps=engine_mod.evaluate_fitness
            ) as mock_eval,
            patch.object(
                engine_mod, "tournament_selection", wraps=engine_mod.tournament_selection
            ) as mock_select,
            patch.object(
                engine_mod, "job_based_crossover", wraps=engine_mod.job_based_crossover
            ) as mock_xover,
            patch.object(
                engine_mod, "swap_mutation", wraps=engine_mod.swap_mutation
            ) as mock_mutate,
            patch.object(
                engine_mod, "decode_active_obr", wraps=engine_mod.decode_active_obr
            ) as mock_decode,
        ):
            run_genetic_algorithm(
                inst,
                population_size=pop_size,
                generations=gens,
                crossover_rate=1.0,
                mutation_rate=0.5,
                elitism_count=elitism,
                seed=11,
            )

            assert mock_create.call_count == pop_size
            assert mock_eval.call_count == pop_size * (gens + 1)
            assert mock_select.call_count == 2 * offspring_per_gen * gens
            assert mock_xover.call_count == offspring_per_gen * gens
            assert mock_mutate.call_count == offspring_per_gen * gens
            assert mock_decode.call_count == 1

        with (
            patch.object(
                engine_mod, "job_based_crossover", wraps=engine_mod.job_based_crossover
            ) as mock_xover_zero,
            patch.object(
                engine_mod, "tournament_selection", wraps=engine_mod.tournament_selection
            ) as mock_select_zero_gen,
        ):
            run_genetic_algorithm(
                inst,
                population_size=pop_size,
                generations=gens,
                crossover_rate=0.0,
                elitism_count=elitism,
                seed=11,
            )
            assert mock_xover_zero.call_count == 0
            assert mock_select_zero_gen.call_count == 2 * offspring_per_gen * gens

        with patch.object(
            engine_mod, "tournament_selection", wraps=engine_mod.tournament_selection
        ) as mock_select_none:
            run_genetic_algorithm(
                inst, population_size=pop_size, generations=0, seed=11
            )
            assert mock_select_none.call_count == 0

    def test_best_solution_tracking_with_and_without_elitism(self) -> None:
        inst = _make_3x3_instance()

        r0 = run_genetic_algorithm(inst, population_size=12, generations=0, elitism_count=1, seed=7)
        r5 = run_genetic_algorithm(inst, population_size=12, generations=5, elitism_count=1, seed=7)
        r20 = run_genetic_algorithm(inst, population_size=12, generations=20, elitism_count=1, seed=7)
        assert r20.makespan <= r5.makespan <= r0.makespan

        r0_no_elite = run_genetic_algorithm(
            inst, population_size=12, generations=0, elitism_count=0, seed=7
        )
        r15_no_elite = run_genetic_algorithm(
            inst, population_size=12, generations=15, elitism_count=0, seed=7
        )
        assert r15_no_elite.makespan <= r0_no_elite.makespan

    def test_deterministic_with_same_seed(self) -> None:
        inst = _make_3x3_instance()
        res1 = run_genetic_algorithm(inst, population_size=12, generations=10, seed=123)
        res2 = run_genetic_algorithm(inst, population_size=12, generations=10, seed=123)

        assert res1.makespan == res2.makespan
        assert res1.machine_schedules == res2.machine_schedules
        assert res1.job_schedules == res2.job_schedules

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"population_size": 6, "generations": 0, "seed": 5},
            {"population_size": 6, "generations": 5, "elitism_count": 0, "seed": 5},
            {"population_size": 6, "generations": 5, "elitism_count": 3, "seed": 5},
            {"population_size": 6, "generations": 5, "elitism_count": 6, "seed": 5},
            {"population_size": 1, "generations": 3, "seed": 5},
            {"population_size": 6, "generations": 4, "tournament_size": 1, "seed": 5},
            {"population_size": 2, "generations": 4, "tournament_size": 5, "seed": 5},
            {"population_size": 6, "generations": 4, "crossover_rate": 0.0, "mutation_rate": 0.0, "seed": 5},
            {"population_size": 6, "generations": 4, "crossover_rate": 1.0, "mutation_rate": 1.0, "seed": 5},
        ],
    )
    def test_valid_parameter_configurations(self, kwargs: dict[str, int | float]) -> None:
        inst = _make_3x3_instance()
        res = run_genetic_algorithm(inst, **kwargs)  # type: ignore[arg-type]
        assert res.is_feasible is True
        assert validate_schedule(inst, res.machine_schedules, res.job_schedules) == (True, [])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"population_size": 0}, "population_size"),
            ({"population_size": -5}, "population_size"),
            ({"generations": -1}, "generations"),
            ({"generations": -10}, "generations"),
            ({"crossover_rate": -0.1}, "crossover_rate"),
            ({"crossover_rate": 1.1}, "crossover_rate"),
            ({"crossover_rate": float("nan")}, "crossover_rate"),
            ({"mutation_rate": -0.1}, "mutation_rate"),
            ({"mutation_rate": 1.1}, "mutation_rate"),
            ({"mutation_rate": float("nan")}, "mutation_rate"),
            ({"tournament_size": 0}, "tournament_size"),
            ({"tournament_size": -2}, "tournament_size"),
            ({"elitism_count": -1}, "elitism_count"),
            ({"population_size": 5, "elitism_count": 6}, "elitism_count"),
        ],
    )
    def test_invalid_parameters_raise_value_error(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        inst = _make_instance("2x2", [[(0, 3), (1, 4)], [(1, 2), (0, 5)]])
        with pytest.raises(ValueError, match=match):
            run_genetic_algorithm(inst, **kwargs)  # type: ignore[arg-type]
