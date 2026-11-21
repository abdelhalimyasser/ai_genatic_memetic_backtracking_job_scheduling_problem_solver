from __future__ import annotations

import pytest

from src.core.validator.validation_errors import ValidationError
from src.core.validator.validation_rules import (
    check_job_precedence,
    check_job_schedules_consistency,
    check_machine_no_overlap,
    check_operation_integrity,
)
from src.models import Operation, ProblemInstance, ScheduledOperation


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


class TestValidationError:
    def test_default_errors_wraps_message(self) -> None:
        err = ValidationError("broken schedule")
        assert str(err) == "broken schedule"
        assert err.message == "broken schedule"
        assert err.errors == ["broken schedule"]

    def test_explicit_errors_and_empty_list(self) -> None:
        caller_errors = ["err1", "err2"]
        err = ValidationError("failed", errors=caller_errors)
        assert err.errors == ["err1", "err2"]

        caller_errors.append("mutated")
        assert err.errors == ["err1", "err2"]

        empty_err = ValidationError("none", errors=[])
        assert empty_err.errors == []


class TestCheckOperationIntegrity:
    def test_valid_operation_and_zero_duration(self) -> None:
        inst = ProblemInstance(
            name="zero_op",
            num_jobs=1,
            num_machines=2,
            jobs=[[Operation(0, 0, 0, 0), Operation(0, 1, 1, 3)]],
        )
        assert check_operation_integrity(inst, 0, ScheduledOperation(inst.get_operation(0, 0), 0, 0)) == []
        assert check_operation_integrity(inst, 1, ScheduledOperation(inst.get_operation(0, 1), 0, 3)) == []

    @pytest.mark.parametrize("bad_machine", [-1, 2, 99])
    def test_invalid_machine_id(self, bad_machine: int) -> None:
        inst = _make_2x2_instance()
        sched = ScheduledOperation(inst.get_operation(0, 0), 0, 3)
        errors = check_operation_integrity(inst, bad_machine, sched)
        assert errors == [f"Invalid machine_id {bad_machine}"]

    @pytest.mark.parametrize(
        ("job_id", "op_id"),
        [(2, 0), (0, 2), (5, 5)],
    )
    def test_unknown_out_of_bounds_operation(self, job_id: int, op_id: int) -> None:
        inst = _make_2x2_instance()
        forged = Operation(job_id=job_id, operation_id=op_id, machine_id=0, duration=3)
        errors = check_operation_integrity(inst, 0, ScheduledOperation(forged, 0, 3))
        assert len(errors) == 1
        assert "Unknown operation" in errors[0]

    def test_unknown_operation_with_invalid_machine_reports_both(self) -> None:
        inst = _make_2x2_instance()
        forged = Operation(job_id=3, operation_id=0, machine_id=0, duration=3)
        errors = check_operation_integrity(inst, -1, ScheduledOperation(forged, 0, 3))
        assert len(errors) == 2
        assert any("Invalid machine_id" in e for e in errors)
        assert any("Unknown operation" in e for e in errors)

    def test_wrong_machine_placement(self) -> None:
        inst = _make_2x2_instance()
        sched = ScheduledOperation(inst.get_operation(0, 0), 0, 3)
        errors = check_operation_integrity(inst, 1, sched)
        assert len(errors) == 1
        assert "belongs to machine 0, found on machine 1" in errors[0]

    def test_forged_operation_machine_and_duration(self) -> None:
        inst = _make_2x2_instance()
        forged_machine_op = Operation(job_id=0, operation_id=0, machine_id=1, duration=3)
        errs_m = check_operation_integrity(inst, 1, ScheduledOperation(forged_machine_op, 0, 3))
        assert any("expects machine 0, operation specifies machine 1" in e for e in errs_m)
        assert any("belongs to machine 0, found on machine 1" in e for e in errs_m)

        forged_dur_op = Operation(job_id=0, operation_id=0, machine_id=0, duration=9)
        errs_d = check_operation_integrity(inst, 0, ScheduledOperation(forged_dur_op, 0, 9))
        assert any("definition duration mismatch" in e for e in errs_d)
        assert any("duration mismatch: got 9, expected 3" in e for e in errs_d)

    @pytest.mark.parametrize(
        ("start_time", "end_time"),
        [(0, 2), (0, 5), (5, 2)],
    )
    def test_scheduled_duration_mismatch(self, start_time: int, end_time: int) -> None:
        inst = _make_2x2_instance()
        sched = ScheduledOperation(inst.get_operation(0, 0), start_time, end_time)
        errors = check_operation_integrity(inst, 0, sched)
        assert len(errors) == 1
        assert "duration mismatch" in errors[0]


class TestCheckMachineNoOverlap:
    def test_empty_single_touching_gapped_and_zero_duration_valid(self) -> None:
        assert check_machine_no_overlap(0, []) == []

        op0 = Operation(0, 0, 0, 0)
        op1 = Operation(1, 0, 0, 5)
        op2 = Operation(0, 1, 0, 0)
        op3 = Operation(1, 1, 0, 3)

        assert check_machine_no_overlap(0, [ScheduledOperation(op1, 0, 5)]) == []
        assert (
            check_machine_no_overlap(
                0,
                [
                    ScheduledOperation(op0, 0, 0),
                    ScheduledOperation(op1, 0, 5),
                    ScheduledOperation(op2, 5, 5),
                    ScheduledOperation(op3, 5, 8),
                ],
            )
            == []
        )
        assert (
            check_machine_no_overlap(
                0,
                [ScheduledOperation(op1, 0, 5), ScheduledOperation(op3, 7, 10)],
            )
            == []
        )

    def test_partial_overlap_nested_containment_and_zero_duration_inside(self) -> None:
        op_a = Operation(0, 0, 0, 10)
        op_b = Operation(1, 0, 0, 3)
        op_c = Operation(0, 1, 0, 2)
        op_z = Operation(1, 1, 0, 0)

        partial = check_machine_no_overlap(
            0, [ScheduledOperation(op_a, 0, 10), ScheduledOperation(op_b, 8, 11)]
        )
        assert len(partial) == 1
        assert "Overlap on machine 0" in partial[0]

        nested = check_machine_no_overlap(
            0,
            [
                ScheduledOperation(op_a, 0, 10),
                ScheduledOperation(op_b, 2, 5),
                ScheduledOperation(op_c, 6, 8),
            ],
        )
        assert len(nested) == 2

        zero_inside = check_machine_no_overlap(
            0, [ScheduledOperation(op_a, 0, 10), ScheduledOperation(op_z, 5, 5)]
        )
        assert len(zero_inside) == 1
        assert "starts at 5 before previous ends at 10" in zero_inside[0]


class TestCheckJobPrecedence:
    def test_touching_gapped_zero_duration_and_missing_safe(self) -> None:
        inst = ProblemInstance(
            name="prec",
            num_jobs=2,
            num_machines=3,
            jobs=[
                [Operation(0, 0, 0, 3), Operation(0, 1, 1, 0), Operation(0, 2, 2, 4)],
                [Operation(1, 0, 1, 2), Operation(1, 1, 2, 5), Operation(1, 2, 0, 1)],
            ],
        )
        valid_by_key = {
            (0, 0): ScheduledOperation(inst.get_operation(0, 0), 0, 3),
            (0, 1): ScheduledOperation(inst.get_operation(0, 1), 3, 3),
            (0, 2): ScheduledOperation(inst.get_operation(0, 2), 3, 7),
            (1, 0): ScheduledOperation(inst.get_operation(1, 0), 0, 2),
            (1, 1): ScheduledOperation(inst.get_operation(1, 1), 4, 9),
            (1, 2): ScheduledOperation(inst.get_operation(1, 2), 9, 10),
        }
        assert check_job_precedence(inst, valid_by_key) == []

        partial_by_key = {(0, 0): valid_by_key[(0, 0)], (0, 2): valid_by_key[(0, 2)]}
        assert check_job_precedence(inst, partial_by_key) == []

    def test_single_and_multi_job_precedence_violations(self) -> None:
        inst = _make_2x2_instance()
        bad_by_key = {
            (0, 0): ScheduledOperation(inst.get_operation(0, 0), 0, 3),
            (0, 1): ScheduledOperation(inst.get_operation(0, 1), 2, 6),
            (1, 0): ScheduledOperation(inst.get_operation(1, 0), 2, 4),
            (1, 1): ScheduledOperation(inst.get_operation(1, 1), 3, 8),
        }
        errors = check_job_precedence(inst, bad_by_key)
        assert len(errors) == 2
        assert "Job 0 order error" in errors[0]
        assert "Job 1 order error" in errors[1]


class TestCheckJobSchedulesConsistency:
    def _valid_maps(
        self, inst: ProblemInstance
    ) -> tuple[dict[tuple[int, int], ScheduledOperation], dict[int, list[ScheduledOperation]]]:
        s00 = ScheduledOperation(inst.get_operation(0, 0), 0, 3)
        s01 = ScheduledOperation(inst.get_operation(0, 1), 3, 7)
        s10 = ScheduledOperation(inst.get_operation(1, 0), 0, 2)
        s11 = ScheduledOperation(inst.get_operation(1, 1), 3, 8)
        by_key = {(0, 0): s00, (0, 1): s01, (1, 0): s10, (1, 1): s11}
        job_schedules = {0: [s00, s01], 1: [s10, s11]}
        return by_key, job_schedules

    def test_exact_match_is_valid(self) -> None:
        inst = _make_2x2_instance()
        by_key, job_schedules = self._valid_maps(inst)
        assert check_job_schedules_consistency(inst, by_key, job_schedules) == []

    def test_empty_and_partial_job_schedules(self) -> None:
        inst = _make_2x2_instance()
        by_key, job_schedules = self._valid_maps(inst)

        empty_errs = check_job_schedules_consistency(inst, by_key, {})
        assert len(empty_errs) == 4
        assert all("missing from job_schedules" in e for e in empty_errs)

        partial_errs = check_job_schedules_consistency(inst, by_key, {0: job_schedules[0]})
        assert len(partial_errs) == 2
        assert any("(1, 0) missing from job_schedules" in e for e in partial_errs)
        assert any("(1, 1) missing from job_schedules" in e for e in partial_errs)

    def test_time_and_operation_mismatches(self) -> None:
        inst = _make_2x2_instance()
        by_key, job_schedules = self._valid_maps(inst)

        job_schedules[0][1] = ScheduledOperation(inst.get_operation(0, 1), 4, 8)
        time_errs = check_job_schedules_consistency(inst, by_key, job_schedules)
        assert any("Mismatch for (0, 1)" in e for e in time_errs)

        _, job_schedules2 = self._valid_maps(inst)
        forged_op = Operation(job_id=0, operation_id=1, machine_id=0, duration=4)
        job_schedules2[0][1] = ScheduledOperation(forged_op, 3, 7)
        op_errs = check_job_schedules_consistency(inst, by_key, job_schedules2)
        assert any("Mismatch for (0, 1)" in e for e in op_errs)

    def test_wrong_job_key_duplicate_invalid_job_id_and_missing_from_machine(self) -> None:
        inst = _make_2x2_instance()
        by_key, job_schedules = self._valid_maps(inst)

        job_schedules[0].append(job_schedules[1][0])
        dup_wrong_errs = check_job_schedules_consistency(inst, by_key, job_schedules)
        assert any("placed under wrong job_id 0" in e for e in dup_wrong_errs)
        assert any("Duplicate operation (1, 0) in job_schedules" in e for e in dup_wrong_errs)

        _, job_schedules2 = self._valid_maps(inst)
        job_schedules2[2] = [ScheduledOperation(inst.get_operation(0, 0), 0, 3)]
        bad_id_errs = check_job_schedules_consistency(inst, by_key, job_schedules2)
        assert any("Invalid job_id 2 in job_schedules" in e for e in bad_id_errs)

        _, job_schedules3 = self._valid_maps(inst)
        job_schedules3[-1] = []
        neg_id_errs = check_job_schedules_consistency(inst, by_key, job_schedules3)
        assert any("Invalid job_id -1 in job_schedules" in e for e in neg_id_errs)

        del by_key[(1, 1)]
        _, job_schedules4 = self._valid_maps(inst)
        missing_m_errs = check_job_schedules_consistency(inst, by_key, job_schedules4)
        assert missing_m_errs == ["Operation (1, 1) missing from machine_schedules"]
