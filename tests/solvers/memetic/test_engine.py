from __future__ import annotations

import pytest

import src.solvers.memetic as memetic_pkg
from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance, ScheduleResult
from src.solvers.memetic import run_memetic_algorithm


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
        assert memetic_pkg.__version__ == "1.0.0"
        expected = {
            "generate_neighbor",
            "insert_neighbor",
            "run_memetic_algorithm",
            "swap_neighbor",
            "tabu_search",
        }
        assert set(memetic_pkg.__all__) == expected
        for name in expected:
            assert hasattr(memetic_pkg, name)


class TestMemeticAlgorithmEngine:
    @pytest.mark.parametrize(
        "instance",
        [
            _make_instance("1x1", [[(0, 5)]]),
            _make_instance("2x2", [[(0, 3), (1, 4)], [(1, 2), (0, 5)]]),
            _make_3x3_instance(),
        ],
        ids=["1x1", "2x2", "3x3"],
    )
    def test_produces_feasible_validated_result(
        self, instance: ProblemInstance
    ) -> None:
        result = run_memetic_algorithm(
            instance,
            population_size=6,
            generations=5,
            local_search_rate=0.5,
            tabu_iterations=5,
            neighbors_per_iter=4,
            tabu_tenure=3,
            seed=42,
        )

        assert isinstance(result, ScheduleResult)
        assert result.instance_name == instance.name
        assert result.algorithm_name == "Memetic Algorithm"
        assert result.is_feasible is True
        assert result.runtime_sec >= 0.0

        is_valid, errors = validate_schedule(
            instance, result.machine_schedules, result.job_schedules
        )
        assert is_valid is True
        assert errors == []

        expected_makespan = max(
            (s.end_time for ops in result.machine_schedules.values() for s in ops),
            default=0,
        )
        assert result.makespan == expected_makespan

    def test_deterministic_for_same_seed(self) -> None:
        instance = _make_3x3_instance()
        res1 = run_memetic_algorithm(
            instance,
            population_size=8,
            generations=5,
            local_search_rate=0.4,
            tabu_iterations=5,
            neighbors_per_iter=4,
            tabu_tenure=3,
            seed=123,
        )
        res2 = run_memetic_algorithm(
            instance,
            population_size=8,
            generations=5,
            local_search_rate=0.4,
            tabu_iterations=5,
            neighbors_per_iter=4,
            tabu_tenure=3,
            seed=123,
        )

        assert res1.makespan == res2.makespan
        assert res1.machine_schedules == res2.machine_schedules
        assert res1.job_schedules == res2.job_schedules

    def test_best_solution_tracking_monotonic(self) -> None:
        instance = _make_3x3_instance()
        r0 = run_memetic_algorithm(
            instance, population_size=8, generations=0, seed=7
        )
        r5 = run_memetic_algorithm(
            instance,
            population_size=8,
            generations=5,
            local_search_rate=0.5,
            tabu_iterations=5,
            neighbors_per_iter=4,
            seed=7,
        )
        assert r5.makespan <= r0.makespan

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"population_size": 6, "generations": 0, "seed": 1},
            {"population_size": 6, "generations": 3, "elitism_count": 0, "seed": 1},
            {"population_size": 6, "generations": 3, "elitism_count": 3, "seed": 1},
            {"population_size": 6, "generations": 3, "local_search_rate": 0.0, "seed": 1},
            {"population_size": 6, "generations": 3, "local_search_rate": 1.0, "tabu_iterations": 3, "seed": 1},
            {"population_size": 6, "generations": 3, "crossover_rate": 0.0, "mutation_rate": 0.0, "seed": 1},
            {"population_size": 6, "generations": 3, "tabu_tenure": 0, "seed": 1},
        ],
    )
    def test_valid_parameter_configurations(
        self, kwargs: dict[str, int | float]
    ) -> None:
        instance = _make_3x3_instance()
        res = run_memetic_algorithm(instance, **kwargs)  # type: ignore[arg-type]
        assert res.is_feasible is True
        assert validate_schedule(instance, res.machine_schedules, res.job_schedules) == (True, [])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"population_size": 0}, "population_size"),
            ({"population_size": -3}, "population_size"),
            ({"generations": -1}, "generations"),
            ({"crossover_rate": -0.1}, "crossover_rate"),
            ({"crossover_rate": 1.2}, "crossover_rate"),
            ({"mutation_rate": -0.1}, "mutation_rate"),
            ({"mutation_rate": 1.5}, "mutation_rate"),
            ({"tournament_size": 0}, "tournament_size"),
            ({"elitism_count": -1}, "elitism_count"),
            ({"population_size": 4, "elitism_count": 5}, "elitism_count"),
            ({"local_search_rate": -0.1}, "local_search_rate"),
            ({"local_search_rate": 1.1}, "local_search_rate"),
            ({"tabu_iterations": 0}, "tabu_iterations"),
            ({"tabu_iterations": -5}, "tabu_iterations"),
            ({"neighbors_per_iter": 0}, "neighbors_per_iter"),
            ({"neighbors_per_iter": -1}, "neighbors_per_iter"),
            ({"tabu_tenure": -1}, "tabu_tenure"),
        ],
    )
    def test_invalid_parameters_raise_value_error(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        instance = _make_instance("2x2", [[(0, 3), (1, 4)], [(1, 2), (0, 5)]])
        with pytest.raises(ValueError, match=match):
            run_memetic_algorithm(instance, **kwargs)  # type: ignore[arg-type]
