from __future__ import annotations

from pathlib import Path

import pytest

import src.core.validator as validator_pkg
from src.core.parser import load_instance
from src.core.validator import ValidationError, validate_or_raise, validate_schedule
from src.models import Operation, ProblemInstance, ScheduledOperation

DATASETS_DIR = Path(__file__).resolve().parents[3] / "dataset"
JOB1_PATH = DATASETS_DIR / "jobshop1.txt"
JOB2_PATH = DATASETS_DIR / "jobshop2.txt"


def _make_2x2_instance() -> ProblemInstance:
    return ProblemInstance(
        name="mini2x2",
        num_jobs=2,
        num_machines=2,
        jobs=[
            [Operation(0, 0, 0, 3), Operation(0, 1, 1, 4)],
            [Operation(1, 0, 1, 2), Operation(1, 1, 0, 5)],
        ],
    )


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


def _build_valid_schedule(
    instance: ProblemInstance,
) -> tuple[dict[int, list[ScheduledOperation]], dict[int, list[ScheduledOperation]]]:
    """Construct a deterministic valid schedule respecting precedence and machine capacity."""
    machine_schedules: dict[int, list[ScheduledOperation]] = {
        m: [] for m in range(instance.num_machines)
    }
    job_schedules: dict[int, list[ScheduledOperation]] = {
        j: [] for j in range(instance.num_jobs)
    }
    job_ready = [0] * instance.num_jobs
    machine_ready = [0] * instance.num_machines

    for op_id in range(instance.num_machines):
        for job_id in range(instance.num_jobs):
            op = instance.get_operation(job_id, op_id)
            start = max(job_ready[job_id], machine_ready[op.machine_id])
            end = start + op.duration
            sched = ScheduledOperation(operation=op, start_time=start, end_time=end)
            machine_schedules[op.machine_id].append(sched)
            job_schedules[job_id].append(sched)
            job_ready[job_id] = end
            machine_ready[op.machine_id] = end

    return machine_schedules, job_schedules


def _assert_invalid(
    instance: ProblemInstance,
    machine_schedules: dict[int, list[ScheduledOperation]],
    job_schedules: dict[int, list[ScheduledOperation]] | None = None,
    *expected_substrings: str,
) -> list[str]:
    is_valid, errors = validate_schedule(instance, machine_schedules, job_schedules)
    assert is_valid is False
    assert errors
    for needle in expected_substrings:
        assert any(needle in err for err in errors), (
            f"Expected {needle!r} in errors: {errors}"
        )

    with pytest.raises(ValidationError) as exc_info:
        validate_or_raise(instance, machine_schedules, job_schedules)
    assert exc_info.value.errors == errors
    assert instance.name in exc_info.value.message
    return errors


class TestPublicExports:
    def test_expected_symbols_exported(self) -> None:
        assert validator_pkg.__version__ == "1.0.0"
        expected = {"ValidationError", "validate_schedule", "validate_or_raise"}
        assert set(validator_pkg.__all__) == expected
        for name in expected:
            assert hasattr(validator_pkg, name)


class TestValidSchedules:
    def test_valid_2x2_and_3x3_with_and_without_job_schedules(self) -> None:
        for inst in (_make_2x2_instance(), _make_3x3_instance()):
            machine_schedules, job_schedules = _build_valid_schedule(inst)
            assert validate_schedule(inst, machine_schedules) == (True, [])
            assert validate_schedule(inst, machine_schedules, job_schedules) == (True, [])
            validate_or_raise(inst, machine_schedules)
            validate_or_raise(inst, machine_schedules, job_schedules)

    def test_unsorted_machine_schedules_are_accepted(self) -> None:
        inst = _make_2x2_instance()
        machine_schedules, job_schedules = _build_valid_schedule(inst)
        reversed_machines = {m: list(reversed(ops)) for m, ops in machine_schedules.items()}
        assert validate_schedule(inst, reversed_machines, job_schedules) == (True, [])
        validate_or_raise(inst, reversed_machines, job_schedules)

    def test_synthetic_zero_duration_schedule_is_valid(self) -> None:
        inst = ProblemInstance(
            name="zeros",
            num_jobs=2,
            num_machines=2,
            jobs=[
                [Operation(0, 0, 0, 0), Operation(0, 1, 1, 2)],
                [Operation(1, 0, 1, 0), Operation(1, 1, 0, 0)],
            ],
        )
        machine_schedules = {
            0: [
                ScheduledOperation(inst.get_operation(0, 0), 0, 0),
                ScheduledOperation(inst.get_operation(1, 1), 0, 0),
            ],
            1: [
                ScheduledOperation(inst.get_operation(1, 0), 0, 0),
                ScheduledOperation(inst.get_operation(0, 1), 0, 2),
            ],
        }
        assert validate_schedule(inst, machine_schedules) == (True, [])


class TestConstraintViolations:
    def test_empty_and_missing_operations(self) -> None:
        inst = _make_2x2_instance()
        _assert_invalid(
            inst,
            {},
            None,
            "Expected 4 operations, found 0",
            "Missing operation (job=0, op=0)",
        )

        machine_schedules, _ = _build_valid_schedule(inst)
        machine_schedules[1].pop()
        _assert_invalid(
            inst,
            machine_schedules,
            None,
            "Expected 4 operations, found 3",
            "Missing operation",
        )

    def test_duplicate_operation_same_and_different_machines(self) -> None:
        inst = _make_2x2_instance()
        m_sched, _ = _build_valid_schedule(inst)
        dup_op = ScheduledOperation(inst.get_operation(0, 0), 10, 13)
        m_sched[0].append(dup_op)
        _assert_invalid(inst, m_sched, None, "Duplicate operation (job=0, op=0)")

        m_sched2, _ = _build_valid_schedule(inst)
        m_sched2[1].append(ScheduledOperation(inst.get_operation(0, 0), 20, 23))
        _assert_invalid(
            inst,
            m_sched2,
            None,
            "Duplicate operation (job=0, op=0)",
            "belongs to machine 0, found on machine 1",
        )

    def test_machine_overlap_even_when_unsorted(self) -> None:
        inst = _make_2x2_instance()
        machine_schedules = {
            0: [
                ScheduledOperation(inst.get_operation(1, 1), 2, 7),
                ScheduledOperation(inst.get_operation(0, 0), 0, 3),
            ],
            1: [
                ScheduledOperation(inst.get_operation(1, 0), 0, 2),
                ScheduledOperation(inst.get_operation(0, 1), 3, 7),
            ],
        }
        _assert_invalid(inst, machine_schedules, None, "Overlap on machine 0")

    def test_job_precedence_violation(self) -> None:
        inst = _make_2x2_instance()
        machine_schedules = {
            0: [
                ScheduledOperation(inst.get_operation(0, 0), 3, 6),
                ScheduledOperation(inst.get_operation(1, 1), 6, 11),
            ],
            1: [
                ScheduledOperation(inst.get_operation(1, 0), 0, 2),
                ScheduledOperation(inst.get_operation(0, 1), 2, 6),
            ],
        }
        _assert_invalid(inst, machine_schedules, None, "Job 0 order error")

    def test_wrong_machine_assignment(self) -> None:
        inst = _make_2x2_instance()
        machine_schedules = {
            0: [
                ScheduledOperation(inst.get_operation(0, 0), 0, 3),
                ScheduledOperation(inst.get_operation(1, 0), 3, 5),
                ScheduledOperation(inst.get_operation(1, 1), 5, 10),
            ],
            1: [
                ScheduledOperation(inst.get_operation(0, 1), 3, 7),
            ],
        }
        _assert_invalid(
            inst,
            machine_schedules,
            None,
            "Job 1 op 0 belongs to machine 1, found on machine 0",
        )

    @pytest.mark.parametrize(
        ("start_time", "end_time"),
        [(0, 2), (0, 5), (5, 2)],
    )
    def test_duration_mismatches_including_negative(
        self, start_time: int, end_time: int
    ) -> None:
        inst = _make_2x2_instance()
        m_sched, _ = _build_valid_schedule(inst)
        m_sched[0][0] = ScheduledOperation(inst.get_operation(0, 0), start_time, end_time)
        _assert_invalid(inst, m_sched, None, "duration mismatch")

    def test_forged_operation_machine_and_duration_detected(self) -> None:
        inst = _make_2x2_instance()
        m_sched, _ = _build_valid_schedule(inst)
        m_sched[0][0] = ScheduledOperation(Operation(0, 0, 0, 9), 0, 9)
        _assert_invalid(inst, m_sched, None, "definition duration mismatch")

        m_sched2, _ = _build_valid_schedule(inst)
        m_sched2[0].pop(0)
        m_sched2[1].append(ScheduledOperation(Operation(0, 0, 1, 3), 10, 13))
        _assert_invalid(
            inst,
            m_sched2,
            None,
            "expects machine 0, operation specifies machine 1",
            "belongs to machine 0, found on machine 1",
        )

    def test_invalid_machine_key_populated_and_empty(self) -> None:
        inst = _make_2x2_instance()
        m_sched, _ = _build_valid_schedule(inst)
        m_sched[5] = []
        _assert_invalid(inst, m_sched, None, "Invalid machine_id 5")

        m_sched2, _ = _build_valid_schedule(inst)
        moved = m_sched2[1].pop()
        m_sched2[-1] = [moved]
        _assert_invalid(
            inst,
            m_sched2,
            None,
            "Invalid machine_id -1",
            "Expected 4 operations, found 3",
            "Missing operation",
        )

    def test_unknown_operation_cannot_mask_missing_valid_operation(self) -> None:
        inst = _make_2x2_instance()
        m_sched, _ = _build_valid_schedule(inst)
        m_sched[0].pop()
        m_sched[0].append(ScheduledOperation(Operation(9, 0, 0, 2), 20, 22))
        _assert_invalid(
            inst,
            m_sched,
            None,
            "Unknown operation (job=9, op=0)",
            "Expected 4 operations, found 3",
            "Missing operation (job=1, op=1)",
        )

    def test_job_schedules_inconsistencies(self) -> None:
        inst = _make_2x2_instance()
        m_sched, j_sched = _build_valid_schedule(inst)

        _assert_invalid(inst, m_sched, {}, "missing from job_schedules")

        j_missing = {0: j_sched[0]}
        _assert_invalid(inst, m_sched, j_missing, "(1, 0) missing from job_schedules")

        _, j_mismatch = _build_valid_schedule(inst)
        j_mismatch[0][0] = ScheduledOperation(inst.get_operation(0, 0), 1, 4)
        _assert_invalid(inst, m_sched, j_mismatch, "Mismatch for (0, 0)")

        _, j_wrong_key = _build_valid_schedule(inst)
        j_wrong_key[0].append(j_wrong_key[1].pop(0))
        _assert_invalid(
            inst,
            m_sched,
            j_wrong_key,
            "Operation (1, 0) placed under wrong job_id 0",
        )

        _, j_dup = _build_valid_schedule(inst)
        j_dup[0].append(j_dup[0][0])
        _assert_invalid(inst, m_sched, j_dup, "Duplicate operation (0, 0)")

        _, j_bad_id = _build_valid_schedule(inst)
        j_bad_id[3] = []
        _assert_invalid(inst, m_sched, j_bad_id, "Invalid job_id 3 in job_schedules")

        m_partial, j_full = _build_valid_schedule(inst)
        m_partial[0].pop()
        _assert_invalid(
            inst,
            m_partial,
            j_full,
            "Operation (1, 1) missing from machine_schedules",
        )

    def test_multiple_simultaneous_violations_collected(self) -> None:
        inst = _make_2x2_instance()
        machine_schedules = {
            0: [
                ScheduledOperation(inst.get_operation(0, 0), 5, 8),
                ScheduledOperation(inst.get_operation(1, 1), 6, 10),
            ],
            1: [
                ScheduledOperation(inst.get_operation(0, 1), 0, 4),
            ],
        }
        errors = _assert_invalid(
            inst,
            machine_schedules,
            None,
            "duration mismatch",
            "Overlap on machine 0",
            "Expected 4 operations, found 3",
            "Missing operation (job=1, op=0)",
            "Job 0 order error",
        )
        assert len(errors) >= 5


class TestRealDatasetIntegration:
    @pytest.mark.parametrize(
        ("path", "fmt", "name"),
        [
            (JOB1_PATH, "job1", "ft06"),
            (JOB1_PATH, "job1", "orb07"),
            (JOB2_PATH, "job2", "ta01"),
        ],
    )
    def test_valid_schedules_for_benchmark_instances(
        self, path: Path, fmt: str, name: str
    ) -> None:
        inst = load_instance(path, name, format_type=fmt)
        machine_schedules, job_schedules = _build_valid_schedule(inst)
        assert validate_schedule(inst, machine_schedules, job_schedules) == (True, [])
        validate_or_raise(inst, machine_schedules, job_schedules)

    def test_mutated_ft06_schedule_fails_validation(self) -> None:
        ft06 = load_instance(JOB1_PATH, "ft06", format_type="job1")
        machine_schedules, job_schedules = _build_valid_schedule(ft06)

        # Shift job 3 op 1 to start inside job 0 op 1 on machine 0 and before job 3 op 0 finishes
        bad_op = ft06.get_operation(3, 1)
        m_list = machine_schedules[bad_op.machine_id]
        idx = next(i for i, s in enumerate(m_list) if s.operation == bad_op)
        overlap_start = m_list[0].start_time + 1
        m_list[idx] = ScheduledOperation(
            bad_op, overlap_start, overlap_start + bad_op.duration
        )

        _assert_invalid(
            ft06,
            machine_schedules,
            job_schedules,
            f"Overlap on machine {bad_op.machine_id}",
            f"Job {bad_op.job_id} order error",
            f"Mismatch for ({bad_op.job_id}, {bad_op.operation_id})",
        )
