from __future__ import annotations

from pathlib import Path

import pytest

from src.core.parser import ParserError, list_jobshop1_instances, parse_jobshop1
from src.models import Operation, ProblemInstance

DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "jobshop1.txt"


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestRealJobshop1Dataset:
    @pytest.fixture(scope="class")
    @classmethod
    def all_instances(cls) -> dict[str, ProblemInstance]:
        return parse_jobshop1(DATASET_PATH)

    def test_parses_all_82_instances(self, all_instances: dict[str, ProblemInstance]) -> None:
        assert len(all_instances) == 82
        assert "abz5" in all_instances
        assert "ft06" in all_instances
        assert "orb07" in all_instances
        assert "yn4" in all_instances

    def test_ft06_exact_structure(self, all_instances: dict[str, ProblemInstance]) -> None:
        ft06 = all_instances["ft06"]
        assert ft06.name == "ft06"
        assert ft06.num_jobs == 6
        assert ft06.num_machines == 6
        assert ft06.total_operations == 36
        assert ft06.description == "Fisher and Thompson 6x6 instance, alternate name (mt06)"

        expected_pairs = [
            [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
            [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
            [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
            [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
            [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
            [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
        ]
        for job_id, row in enumerate(expected_pairs):
            for op_id, (machine_id, duration) in enumerate(row):
                op = ft06.get_operation(job_id, op_id)
                assert op == Operation(
                    job_id=job_id,
                    operation_id=op_id,
                    machine_id=machine_id,
                    duration=duration,
                )

    def test_orb07_allows_zero_duration_operation(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        orb07 = all_instances["orb07"]
        op = orb07.get_operation(9, 9)
        assert op.machine_id == 0
        assert op.duration == 0

    def test_structural_invariants_across_all_instances(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        for name, inst in all_instances.items():
            assert inst.name == name
            assert inst.num_jobs > 0
            assert inst.num_machines > 0
            assert len(inst.jobs) == inst.num_jobs
            assert inst.total_operations == inst.num_jobs * inst.num_machines
            expected_machines = list(range(inst.num_machines))

            for job_id, job_ops in enumerate(inst.jobs):
                assert len(job_ops) == inst.num_machines
                machines: list[int] = []
                for op_id, op in enumerate(job_ops):
                    assert op.job_id == job_id
                    assert op.operation_id == op_id
                    assert 0 <= op.machine_id < inst.num_machines
                    assert op.duration >= 0
                    machines.append(op.machine_id)
                assert sorted(machines) == expected_machines

    def test_list_jobshop1_instances_matches_keys_and_order(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        names = list_jobshop1_instances(DATASET_PATH)
        assert names == list(all_instances.keys())
        assert len(names) == 82

    def test_selective_loading(self, all_instances: dict[str, ProblemInstance]) -> None:
        subset = parse_jobshop1(DATASET_PATH, instance_names=["ft06", "la01"])
        assert set(subset.keys()) == {"ft06", "la01"}
        assert subset["ft06"] == all_instances["ft06"]
        assert subset["la01"] == all_instances["la01"]

    def test_selective_loading_empty_returns_empty_dict(self) -> None:
        assert parse_jobshop1(DATASET_PATH, instance_names=[]) == {}

    def test_selective_loading_rejects_single_string(self) -> None:
        with pytest.raises(ParserError, match="iterable"):
            parse_jobshop1(DATASET_PATH, instance_names="ft06")

    def test_selective_loading_missing_name_raises(self) -> None:
        with pytest.raises(ParserError, match="not_an_instance"):
            parse_jobshop1(DATASET_PATH, instance_names=["ft06", "not_an_instance"])


class TestSyntheticJobshop1Cases:
    def test_accepts_str_and_path_with_separators_tabs_and_zero_duration(
        self, tmp_path: Path
    ) -> None:
        content = (
            "preamble line\n"
            "instance\tmini\n"
            "+++++\n"
            "-----\n"
            "=====\n"
            "\n"
            "Tiny 2x2 instance\n"
            "2 2\n"
            "0 0 1 5\n"
            "1 3 0 7\n"
        )
        file_path = _write_text(tmp_path / "mini.txt", content)

        from_path = parse_jobshop1(file_path)
        from_str = parse_jobshop1(str(file_path))

        assert from_path == from_str
        assert list_jobshop1_instances(file_path) == ["mini"]
        inst = from_path["mini"]
        assert inst.description == "Tiny 2x2 instance"
        assert inst.get_operation(0, 0).duration == 0
        assert inst.get_operation(1, 1).machine_id == 0

    def test_missing_file_and_directory_errors(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(ParserError, match="File not found"):
            parse_jobshop1(missing)
        with pytest.raises(ParserError, match="File not found"):
            list_jobshop1_instances(missing)

        with pytest.raises(ParserError, match="not a file"):
            parse_jobshop1(tmp_path)
        with pytest.raises(ParserError, match="not a file"):
            list_jobshop1_instances(tmp_path)

    def test_invalid_utf8_raises_parser_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad_encoding.txt"
        bad_file.write_bytes(b"\xff\xfe\xfd")

        with pytest.raises(ParserError, match="Failed to read"):
            parse_jobshop1(bad_file)
        with pytest.raises(ParserError, match="Failed to read"):
            parse_jobshop1(bad_file, instance_names=[])
        with pytest.raises(ParserError, match="Failed to read"):
            list_jobshop1_instances(bad_file)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "just some comments\nwithout any instances\n",
        ],
    )
    def test_no_instances_found_raises(self, tmp_path: Path, content: str) -> None:
        file_path = _write_text(tmp_path / "empty.txt", content)
        with pytest.raises(ParserError, match="No instances found"):
            parse_jobshop1(file_path)
        with pytest.raises(ParserError, match="No instances found"):
            list_jobshop1_instances(file_path)

    @pytest.mark.parametrize(
        "header",
        [
            "instance\n",
            "instance   \n",
            "instance one two\n",
        ],
    )
    def test_malformed_instance_header_raises(self, tmp_path: Path, header: str) -> None:
        file_path = _write_text(tmp_path / "bad_header.txt", header)
        with pytest.raises(ParserError, match="Invalid instance header"):
            parse_jobshop1(file_path)
        with pytest.raises(ParserError, match="Invalid instance header"):
            list_jobshop1_instances(file_path)

    def test_duplicate_instance_name_raises(self, tmp_path: Path) -> None:
        content = (
            "instance dup\n"
            "First\n"
            "1 1\n"
            "0 5\n"
            "instance dup\n"
            "Second\n"
            "1 1\n"
            "0 6\n"
        )
        file_path = _write_text(tmp_path / "dup.txt", content)
        with pytest.raises(ParserError, match="Duplicate instance"):
            parse_jobshop1(file_path)
        with pytest.raises(ParserError, match="Duplicate instance"):
            list_jobshop1_instances(file_path)

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("instance a\n+++\n", "Missing description"),
            ("instance a\ndesc\n", "Missing job/machine counts"),
            ("instance a\ndesc\n2\n", "Invalid dimensions"),
            ("instance a\ndesc\n2 2 2\n", "Invalid dimensions"),
            ("instance a\ndesc\ntwo 2\n", "Non-integer dimensions"),
            ("instance a\ndesc\n0 2\n", "Dimensions must be positive"),
            ("instance a\ndesc\n2 -1\n", "Dimensions must be positive"),
            ("instance a\ndesc\n2 2\n0 1 1 2\n", "Missing job 1 data"),
            ("instance a\ndesc\n1 2\n0 1 1\n", "Expected 4 numbers"),
            ("instance a\ndesc\n1 2\n0 1 1 x\n", "Non-integer value"),
            ("instance a\ndesc\n1 2\n-1 5 1 3\n", "Invalid machine_id"),
            ("instance a\ndesc\n1 2\n0 5 2 3\n", "Invalid machine_id"),
            ("instance a\ndesc\n1 2\n0 -5 1 3\n", "Negative duration"),
        ],
    )
    def test_invalid_instance_bodies_raise_parser_error(
        self, tmp_path: Path, content: str, match: str
    ) -> None:
        file_path = _write_text(tmp_path / "invalid.txt", content)
        with pytest.raises(ParserError, match=match):
            parse_jobshop1(file_path)
