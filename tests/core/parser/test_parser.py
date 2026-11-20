from __future__ import annotations

from pathlib import Path

import pytest

import src.core.parser as parser_pkg
from src.core.parser import (
    ParserError,
    list_instances,
    load_instance,
    load_instances,
)

DATASETS_DIR = Path(__file__).resolve().parents[3] / "dataset"
JOB1_PATH = DATASETS_DIR / "jobshop1.txt"
JOB2_PATH = DATASETS_DIR / "jobshop2.txt"


class TestPublicExports:
    def test_all_expected_symbols_exported(self) -> None:
        expected = {
            "load_instances",
            "load_instance",
            "list_instances",
            "parse_jobshop1",
            "parse_jobshop2",
            "list_jobshop1_instances",
            "list_jobshop2_instances",
            "ParserError",
        }
        assert set(parser_pkg.__all__) == expected
        for symbol in expected:
            assert hasattr(parser_pkg, symbol)


class TestLoadInstancesAndAutoDetection:
    def test_load_instances_job1_default_and_normalized(self) -> None:
        default_loaded = load_instances(JOB1_PATH, instance_names=["ft06"])
        normalized_loaded = load_instances(
            JOB1_PATH, format_type=" JOB1 ", instance_names=["ft06"]
        )
        assert set(default_loaded.keys()) == {"ft06"}
        assert default_loaded == normalized_loaded

    def test_load_instances_job2_and_normalized(self) -> None:
        loaded = load_instances(JOB2_PATH, format_type="Job2", instance_names=["ta01", "ta42"])
        assert set(loaded.keys()) == {"ta01", "ta42"}
        assert loaded["ta01"].num_jobs == 15
        assert loaded["ta42"].num_jobs == 30

    def test_load_instances_auto_detects_both_formats(self) -> None:
        j1 = load_instances(JOB1_PATH, format_type="auto", instance_names=["ft06"])
        j2 = load_instances(JOB2_PATH, format_type="AUTO", instance_names=["ta01"])
        assert j1["ft06"].num_jobs == 6
        assert j2["ta01"].num_jobs == 15

    def test_load_instance_single_from_job1_job2_and_auto(self) -> None:
        ft06 = load_instance(JOB1_PATH, "ft06")
        ft06_auto = load_instance(JOB1_PATH, "ft06", format_type="auto")
        ta01 = load_instance(JOB2_PATH, "ta01", format_type="job2")
        ta42 = load_instance(JOB2_PATH, "ta42", format_type="auto")

        assert ft06 == ft06_auto
        assert ft06.name == "ft06"
        assert ta01.name == "ta01"
        assert ta42.name == "ta42"
        assert ta42.num_jobs == 30 and ta42.num_machines == 20

    def test_list_instances_for_job1_job2_and_auto(self) -> None:
        j1_names = list_instances(JOB1_PATH, format_type="job1")
        j1_auto = list_instances(JOB1_PATH, format_type="auto")
        j2_names = list_instances(JOB2_PATH, format_type="job2")
        j2_auto = list_instances(JOB2_PATH, format_type="auto")

        assert j1_names == j1_auto
        assert len(j1_names) == 82
        assert j2_names == j2_auto
        assert len(j2_names) == 80


class TestFacadeErrorHandling:
    @pytest.mark.parametrize("bad_format", ["job3", "", "   ", None, 123])
    def test_unsupported_format_type_raises(self, bad_format: object) -> None:
        with pytest.raises(ParserError, match="Unsupported format type"):
            load_instances(JOB1_PATH, format_type=bad_format)  # type: ignore[arg-type]
        with pytest.raises(ParserError, match="Unsupported format type"):
            list_instances(JOB1_PATH, format_type=bad_format)  # type: ignore[arg-type]

    def test_auto_detection_errors(self, tmp_path: Path) -> None:
        unknown_file = tmp_path / "unknown.txt"
        unknown_file.write_text("some random text\nno headers here\n", encoding="utf-8")
        with pytest.raises(ParserError, match="Unable to detect dataset format"):
            load_instances(unknown_file, format_type="auto")
        with pytest.raises(ParserError, match="Unable to detect dataset format"):
            list_instances(unknown_file, format_type="auto")

        with pytest.raises(ParserError, match="File not found"):
            load_instances(tmp_path / "missing.txt", format_type="auto")
        with pytest.raises(ParserError, match="not a file"):
            load_instances(tmp_path, format_type="auto")

        bad_enc = tmp_path / "bad_enc.txt"
        bad_enc.write_bytes(b"\xff\xfe")
        with pytest.raises(ParserError, match="Failed to read"):
            load_instances(bad_enc, format_type="auto")

    @pytest.mark.parametrize("bad_name", ["", "   ", None, 42])
    def test_load_instance_invalid_name_raises(self, bad_name: object) -> None:
        with pytest.raises(ParserError, match="non-empty string"):
            load_instance(JOB1_PATH, bad_name)  # type: ignore[arg-type]

    def test_load_instance_nonexistent_name_raises(self) -> None:
        with pytest.raises(ParserError, match="not_found_inst"):
            load_instance(JOB1_PATH, "not_found_inst", format_type="job1")
        with pytest.raises(ParserError, match="not_found_inst"):
            load_instance(JOB2_PATH, "not_found_inst", format_type="job2")
