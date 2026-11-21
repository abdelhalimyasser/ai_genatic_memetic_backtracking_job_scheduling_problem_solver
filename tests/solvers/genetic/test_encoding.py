from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import pytest

from src.core.parser import load_instance
from src.core.validator import validate_schedule
from src.models import Operation, ProblemInstance, ScheduledOperation
from src.solvers.genetic.encoding import (
    _find_active_slot,
    create_chromosome,
    decode_active_obr,
    evaluate_fitness,
)

DATASETS_DIR = Path(__file__).resolve().parents[3] / "dataset"
JOB1_PATH = DATASETS_DIR / "jobshop1.txt"


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


class TestCreateChromosome:
    @pytest.mark.parametrize(
        "instance",
        [
            _make_instance("1x1", [[(0, 4)]]),
            _make_instance("1x4", [[(0, 2), (2, 3), (1, 1), (3, 5)]]),
            _make_instance("4x1", [[(0, 2)], [(0, 4)], [(0, 1)], [(0, 3)]]),
            _make_instance("2x3", [[(0, 2), (1, 3), (2, 1)], [(2, 4), (0, 2), (1, 5)]]),
            _make_instance("3x2", [[(0, 3), (1, 2)], [(1, 4), (0, 1)], [(0, 2), (1, 3)]]),
            _make_instance(
                "3x3",
                [
                    [(0, 2), (1, 3), (2, 4)],
                    [(1, 5), (2, 1), (0, 2)],
                    [(2, 3), (0, 4), (1, 2)],
                ],
            ),
        ],
        ids=["1x1", "1x4", "4x1", "2x3", "3x2", "3x3"],
    )
    def test_chromosome_length_and_exact_job_counts(
        self, instance: ProblemInstance
    ) -> None:
        expected_counts = {
            j_id: instance.num_machines for j_id in range(instance.num_jobs)
        }
        for seed in range(5):
            chrom = create_chromosome(instance, random.Random(seed))
            assert len(chrom) == instance.total_operations
            assert Counter(chrom) == expected_counts

    def test_chromosome_creation_ft06_and_determinism(self) -> None:
        ft06 = load_instance(JOB1_PATH, "ft06", format_type="job1")
        chrom1 = create_chromosome(ft06, random.Random(42))
        chrom2 = create_chromosome(ft06, random.Random(42))
        assert chrom1 == chrom2
        assert len(chrom1) == ft06.total_operations
        assert Counter(chrom1) == {j: ft06.num_machines for j in range(ft06.num_jobs)}

    def test_produces_diverse_permutations_across_seeds(self) -> None:
        inst = _make_instance(
            "3x3",
            [
                [(0, 2), (1, 3), (2, 4)],
                [(1, 5), (2, 1), (0, 2)],
                [(2, 3), (0, 4), (1, 2)],
            ],
        )
        perms = {tuple(create_chromosome(inst, random.Random(s))) for s in range(10)}
        assert len(perms) > 1


class TestActiveObrDecoding:
    def test_left_shift_into_early_gap_before_first_operation(self) -> None:
        inst = _make_instance(
            "early_gap_2x2",
            [
                [(0, 5), (1, 3)],
                [(1, 4), (0, 2)],
            ],
        )
        makespan, machine_schedules, job_schedules = decode_active_obr(
            inst, [0, 0, 1, 1]
        )

        assert validate_schedule(inst, machine_schedules, job_schedules) == (True, [])
        assert makespan == 8
        assert [
            (s.operation.job_id, s.operation.operation_id, s.start_time, s.end_time)
            for s in machine_schedules[1]
        ] == [(1, 0, 0, 4), (0, 1, 5, 8)]
        assert [
            (s.operation.job_id, s.operation.operation_id, s.start_time, s.end_time)
            for s in machine_schedules[0]
        ] == [(0, 0, 0, 5), (1, 1, 5, 7)]

    def test_left_shift_into_interior_gap_between_operations(self) -> None:
        inst = _make_instance(
            "interior_gap_3x3",
            [
                [(1, 2), (0, 1), (2, 1)],
                [(0, 7), (1, 3), (2, 1)],
                [(2, 3), (1, 3), (0, 1)],
            ],
        )
        chrom = [0, 1, 1, 2, 2, 0, 0, 1, 2]
        makespan, machine_schedules, job_schedules = decode_active_obr(inst, chrom)

        assert validate_schedule(inst, machine_schedules, job_schedules) == (True, [])
        assert [
            (s.operation.job_id, s.operation.operation_id, s.start_time, s.end_time)
            for s in machine_schedules[1]
        ] == [(0, 0, 0, 2), (2, 1, 3, 6), (1, 1, 7, 10)]
        assert makespan == 11

    def test_appends_at_end_when_interior_gap_too_small(self) -> None:
        inst = _make_instance(
            "no_fit_gap_3x3",
            [
                [(1, 2), (0, 1), (2, 1)],
                [(0, 7), (1, 3), (2, 1)],
                [(2, 3), (1, 5), (0, 1)],
            ],
        )
        chrom = [0, 1, 1, 2, 2, 0, 0, 1, 2]
        makespan, machine_schedules, job_schedules = decode_active_obr(inst, chrom)

        assert validate_schedule(inst, machine_schedules, job_schedules) == (True, [])
        assert [
            (s.operation.job_id, s.operation.operation_id, s.start_time, s.end_time)
            for s in machine_schedules[1]
        ] == [(0, 0, 0, 2), (1, 1, 7, 10), (2, 1, 10, 15)]
        assert makespan == 16

    def test_find_active_slot_boundary_cases(self) -> None:
        op0 = Operation(0, 0, 0, 3)
        op1 = Operation(1, 0, 0, 3)

        assert _find_active_slot([], job_ready_time=4, machine_ready_time=0, duration=2) == (4, 0)

        # Exact-fit early gap [0, 5] before op starting at 5
        m_ops_early = [ScheduledOperation(op0, 5, 8)]
        assert _find_active_slot(m_ops_early, job_ready_time=2, machine_ready_time=8, duration=3) == (2, 0)

        # Exact-fit interior gap [3, 7] between [0, 3] and [7, 10]
        m_ops_mid = [ScheduledOperation(op0, 0, 3), ScheduledOperation(op1, 7, 10)]
        assert _find_active_slot(m_ops_mid, job_ready_time=4, machine_ready_time=10, duration=3) == (4, 1)

        # Gap blocked by job_ready_time -> appends at end
        assert _find_active_slot(m_ops_mid, job_ready_time=5, machine_ready_time=10, duration=3) == (10, 2)

        # Zero-duration operation fitting in early gap vs already at earliest_start
        assert _find_active_slot(m_ops_early, job_ready_time=0, machine_ready_time=8, duration=0) == (0, 0)
        assert _find_active_slot(m_ops_early, job_ready_time=8, machine_ready_time=8, duration=0) == (8, 1)


class TestEvaluateFitnessAndEdgeInstances:
    def test_evaluate_fitness_matches_decoded_makespan(self) -> None:
        inst = _make_instance(
            "3x3",
            [
                [(0, 2), (1, 3), (2, 4)],
                [(1, 5), (2, 1), (0, 2)],
                [(2, 3), (0, 4), (1, 2)],
            ],
        )
        for seed in range(10):
            chrom = create_chromosome(inst, random.Random(seed))
            makespan, machine_schedules, job_schedules = decode_active_obr(inst, chrom)
            assert evaluate_fitness(inst, chrom) == makespan
            assert validate_schedule(inst, machine_schedules, job_schedules) == (True, [])

    def test_single_job_single_and_multi_machine_instances(self) -> None:
        inst_1x1 = _make_instance("1x1", [[(0, 7)]])
        m1, ms1, js1 = decode_active_obr(inst_1x1, [0])
        assert m1 == 7
        assert evaluate_fitness(inst_1x1, [0]) == 7
        assert validate_schedule(inst_1x1, ms1, js1) == (True, [])

        inst_1x4 = _make_instance("1x4", [[(2, 3), (0, 4), (3, 2), (1, 5)]])
        m4, ms4, js4 = decode_active_obr(inst_1x4, [0, 0, 0, 0])
        assert m4 == 14
        assert evaluate_fitness(inst_1x4, [0, 0, 0, 0]) == 14
        assert validate_schedule(inst_1x4, ms4, js4) == (True, [])

    def test_multi_job_single_machine_instance(self) -> None:
        inst_4x1 = _make_instance("4x1", [[(0, 2)], [(0, 4)], [(0, 1)], [(0, 3)]])
        chrom = [2, 0, 3, 1]
        makespan, machine_schedules, job_schedules = decode_active_obr(inst_4x1, chrom)
        assert makespan == 10
        assert [s.operation.job_id for s in machine_schedules[0]] == chrom
        assert validate_schedule(inst_4x1, machine_schedules, job_schedules) == (True, [])

    @pytest.mark.parametrize(
        "inst",
        [
            _make_instance("2x3", [[(0, 2), (1, 3), (2, 1)], [(2, 4), (0, 2), (1, 5)]]),
            _make_instance("3x2", [[(0, 3), (1, 2)], [(1, 4), (0, 1)], [(0, 2), (1, 3)]]),
        ],
        ids=["2x3", "3x2"],
    )
    def test_rectangular_instances_decode_validly(self, inst: ProblemInstance) -> None:
        for seed in range(5):
            chrom = create_chromosome(inst, random.Random(seed))
            makespan, ms, js = decode_active_obr(inst, chrom)
            assert evaluate_fitness(inst, chrom) == makespan
            assert validate_schedule(inst, ms, js) == (True, [])

    def test_zero_duration_synthetic_and_orb07_instances(self) -> None:
        all_zero = _make_instance("all_zero", [[(0, 0), (1, 0)], [(1, 0), (0, 0)]])
        m_zero, ms_zero, js_zero = decode_active_obr(all_zero, [0, 1, 0, 1])
        assert m_zero == 0
        assert validate_schedule(all_zero, ms_zero, js_zero) == (True, [])

        mixed_zero = _make_instance("mixed_zero", [[(0, 0), (1, 2)], [(1, 0), (0, 0)]])
        m_mix, ms_mix, js_mix = decode_active_obr(mixed_zero, [1, 0, 1, 0])
        assert m_mix == 2
        assert validate_schedule(mixed_zero, ms_mix, js_mix) == (True, [])

        orb07 = load_instance(JOB1_PATH, "orb07", format_type="job1")
        for seed in range(3):
            chrom = create_chromosome(orb07, random.Random(seed))
            makespan, ms, js = decode_active_obr(orb07, chrom)
            assert evaluate_fitness(orb07, chrom) == makespan
            assert validate_schedule(orb07, ms, js) == (True, [])

    def test_ft06_decoded_chromosomes_pass_validator(self) -> None:
        ft06 = load_instance(JOB1_PATH, "ft06", format_type="job1")
        for seed in range(5):
            chrom = create_chromosome(ft06, random.Random(seed))
            makespan, ms, js = decode_active_obr(ft06, chrom)
            assert makespan >= 55
            assert evaluate_fitness(ft06, chrom) == makespan
            assert validate_schedule(ft06, ms, js) == (True, [])
