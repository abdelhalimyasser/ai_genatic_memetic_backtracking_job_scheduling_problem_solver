from __future__ import annotations

from pathlib import Path

import pytest

from src.core.parser import ParserError, list_jobshop2_instances, parse_jobshop2
from src.core.parser.parse_jobshop2 import _generate_taillard_instance, _unif
from src.models import ProblemInstance

DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "jobshop2.txt"


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestTaillardPRNGAndVerification:
    def test_official_taillard_4x4_verification_vector(self) -> None:
        # From dataset/jobshop2.txt verification block (lines 177-187, 220-226)
        inst = _generate_taillard_instance(
            name="verify",
            time_seed=1166510396,
            machine_seed=164000672,
            num_jobs=4,
            num_machines=4,
        )
        assert inst.num_jobs == 4
        assert inst.num_machines == 4
        assert inst.description == "Taillard 4x4 instance (verify)"

        expected_machines = [
            [2, 0, 3, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
            [0, 2, 1, 3],
        ]
        expected_durations = [
            [54, 34, 61, 2],
            [9, 15, 89, 70],
            [38, 19, 28, 87],
            [95, 34, 7, 29],
        ]

        for job_id in range(4):
            assert [op.machine_id for op in inst.jobs[job_id]] == expected_machines[job_id]
            assert [op.duration for op in inst.jobs[job_id]] == expected_durations[job_id]

    def test_unif_deterministic_and_bounds(self) -> None:
        state_a = [840612802]
        state_b = [840612802]
        seq_a = [_unif(state_a, 1, 99) for _ in range(25)]
        seq_b = [_unif(state_b, 1, 99) for _ in range(25)]
        assert seq_a == seq_b
        assert state_a == state_b
        assert all(1 <= val <= 99 for val in seq_a)

        fixed_state = [12345]
        assert _unif(fixed_state, 7, 7) == 7

    @pytest.mark.parametrize(
        ("state", "low", "high"),
        [
            ([], 1, 10),
            ([0], 1, 10),
            ([-5], 1, 10),
            ([2147483647], 1, 10),
            ([12345], 10, 5),
        ],
    )
    def test_unif_invalid_inputs_raise_value_error(
        self, state: list[int], low: int, high: int
    ) -> None:
        with pytest.raises(ValueError):
            _unif(state, low, high)

    @pytest.mark.parametrize(
        ("time_seed", "machine_seed", "num_jobs", "num_machines"),
        [
            (123, 456, 0, 4),
            (123, 456, 4, -1),
            (0, 456, 4, 4),
            (-1, 456, 4, 4),
            (2147483647, 456, 4, 4),
            (123, 0, 4, 4),
            (123, -10, 4, 4),
            (123, 2147483647, 4, 4),
        ],
    )
    def test_generate_taillard_instance_invalid_args_raise(
        self, time_seed: int, machine_seed: int, num_jobs: int, num_machines: int
    ) -> None:
        with pytest.raises(ParserError):
            _generate_taillard_instance("ta01", time_seed, machine_seed, num_jobs, num_machines)


class TestRealJobshop2Dataset:
    @pytest.fixture(scope="class")
    @classmethod
    def all_instances(cls) -> dict[str, ProblemInstance]:
        return parse_jobshop2(DATASET_PATH)

    def test_parses_all_80_instances_and_normalizes_ta42(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        expected_names = [f"ta{i:02d}" for i in range(1, 81)]
        assert list(all_instances.keys()) == expected_names
        assert "ta4@" not in all_instances
        assert "ta42" in all_instances

    def test_dimensions_across_all_8_taillard_groups(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        expected_dims = [
            (1, 10, 15, 15),
            (11, 20, 20, 15),
            (21, 30, 20, 20),
            (31, 40, 30, 15),
            (41, 50, 30, 20),
            (51, 60, 50, 15),
            (61, 70, 50, 20),
            (71, 80, 100, 20),
        ]
        for start, end, nj, nm in expected_dims:
            for idx in range(start, end + 1):
                inst = all_instances[f"ta{idx:02d}"]
                assert inst.num_jobs == nj
                assert inst.num_machines == nm
                assert inst.total_operations == nj * nm

    def test_determinism_and_structural_invariants(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        second_run = parse_jobshop2(DATASET_PATH, instance_names=["ta01", "ta42", "ta80"])
        assert second_run["ta01"] == all_instances["ta01"]
        assert second_run["ta42"] == all_instances["ta42"]
        assert second_run["ta80"] == all_instances["ta80"]

        for name, inst in all_instances.items():
            assert inst.name == name
            expected_machines = list(range(inst.num_machines))
            for job_id, job_ops in enumerate(inst.jobs):
                assert len(job_ops) == inst.num_machines
                machines: list[int] = []
                for op_id, op in enumerate(job_ops):
                    assert op.job_id == job_id
                    assert op.operation_id == op_id
                    assert 1 <= op.duration <= 99
                    machines.append(op.machine_id)
                assert sorted(machines) == expected_machines

    def test_list_jobshop2_instances_matches_keys_and_order(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        names = list_jobshop2_instances(DATASET_PATH)
        assert names == list(all_instances.keys())
        assert len(names) == 80

    def test_selective_loading_edge_cases(
        self, all_instances: dict[str, ProblemInstance]
    ) -> None:
        subset = parse_jobshop2(str(DATASET_PATH), instance_names=["ta01", "ta42"])
        assert set(subset.keys()) == {"ta01", "ta42"}
        assert subset["ta01"] == all_instances["ta01"]
        assert subset["ta42"] == all_instances["ta42"]

        assert parse_jobshop2(DATASET_PATH, instance_names=[]) == {}

        with pytest.raises(ParserError, match="iterable"):
            parse_jobshop2(DATASET_PATH, instance_names="ta01")

        with pytest.raises(ParserError, match="ta999"):
            parse_jobshop2(DATASET_PATH, instance_names=["ta01", "ta999"])


class TestSyntheticJobshop2Errors:
    def test_missing_file_directory_and_bad_encoding(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.txt"
        with pytest.raises(ParserError, match="File not found"):
            parse_jobshop2(missing)
        with pytest.raises(ParserError, match="File not found"):
            list_jobshop2_instances(missing)

        with pytest.raises(ParserError, match="not a file"):
            parse_jobshop2(tmp_path)
        with pytest.raises(ParserError, match="not a file"):
            list_jobshop2_instances(tmp_path)

        bad_file = tmp_path / "bad.txt"
        bad_file.write_bytes(b"\x80\x81\x82")
        with pytest.raises(ParserError, match="Failed to read"):
            parse_jobshop2(bad_file)
        with pytest.raises(ParserError, match="Failed to read"):
            parse_jobshop2(bad_file, instance_names=[])
        with pytest.raises(ParserError, match="Failed to read"):
            list_jobshop2_instances(bad_file)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "Header text without seeds\n4 jobs 4 machines\n",
        ],
    )
    def test_no_seeds_found_raises(self, tmp_path: Path, content: str) -> None:
        file_path = _write_text(tmp_path / "no_seeds.txt", content)
        with pytest.raises(ParserError, match="No Taillard seeds found"):
            parse_jobshop2(file_path)
        with pytest.raises(ParserError, match="No Taillard seeds found"):
            list_jobshop2_instances(file_path)

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("12345 67890 ta01\n", "precedes any dimensions header"),
            ("0 jobs 5 machines\n123 456 ta01\n", "Dimensions must be positive"),
            ("5 jobs 0 machines\n123 456 ta01\n", "Dimensions must be positive"),
            ("4 jobs 4 machines\n0 456 ta01\n", "Invalid time_seed"),
            ("4 jobs 4 machines\n-5 456 ta01\n", "Invalid time_seed"),
            ("4 jobs 4 machines\n2147483647 456 ta01\n", "Invalid time_seed"),
            ("4 jobs 4 machines\n123 0 ta01\n", "Invalid machine_seed"),
            ("4 jobs 4 machines\n123 -1 ta01\n", "Invalid machine_seed"),
            ("4 jobs 4 machines\n123 2147483648 ta01\n", "Invalid machine_seed"),
            (
                "4 jobs 4 machines\n123 456 ta01\n789 111 ta01\n",
                "Duplicate instance 'ta01'",
            ),
            (
                "4 jobs 4 machines\n123 456 ta42\n789 111 ta4@\n",
                "Duplicate instance 'ta42'",
            ),
        ],
    )
    def test_invalid_seed_and_header_entries_raise(
        self, tmp_path: Path, content: str, match: str
    ) -> None:
        file_path = _write_text(tmp_path / "invalid_j2.txt", content)
        with pytest.raises(ParserError, match=match):
            parse_jobshop2(file_path)
        with pytest.raises(ParserError, match=match):
            list_jobshop2_instances(file_path)
